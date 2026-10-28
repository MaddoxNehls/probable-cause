"""Experiment runner: ``pc eval run --config experiments/<name>.yaml``.

Two modes:
* ``retrieval``: every config in a grid, every answerable question; no LLM (query plans
  come from benchmark/plans.jsonl, written once by ``pc bench rewrite``).
* ``end_to_end``: a few configs on a stratified subset, with generation, verification and
  an LLM judge. Retrieval runs sequentially; generation fans out over worker threads.
"""

import itertools
import json
import statistics
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from probable_cause.eval import metrics
from probable_cause.eval.dataset import Item, read_items
from probable_cause.eval.judge import judge_answers
from probable_cause.llm import LLM
from probable_cause.rag.filters import QueryPlan
from probable_cause.rag.pipeline import complete, prepare
from probable_cause.rag.retrieve import RetrievalConfig, Retriever

KS = (5, 10, 20)
TOP_EVENTS = 20


@dataclass(frozen=True)
class Experiment:
    name: str
    mode: str
    configs: list[RetrievalConfig]
    subset: dict[str, int] | None
    questions: Path
    plans: Path


def load_experiment(path: Path) -> Experiment:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "grid" in spec:
        keys = list(spec["grid"])
        configs = [
            RetrievalConfig(**dict(zip(keys, values, strict=True)))
            for values in itertools.product(*(spec["grid"][k] for k in keys))
        ]
    else:
        configs = [RetrievalConfig(**c) for c in spec["configs"]]
    return Experiment(
        name=spec["name"],
        mode=spec.get("mode", "retrieval"),
        configs=configs,
        subset=spec.get("subset"),
        questions=Path(spec.get("questions", "benchmark/questions.jsonl")),
        plans=Path(spec.get("plans", "benchmark/plans.jsonl")),
    )


def load_plans(path: Path) -> dict[str, QueryPlan]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {r["id"]: QueryPlan.model_validate(r["plan"]) for r in rows}


def stratified_subset(items: list[Item], sizes: dict[str, int]) -> list[Item]:
    """Deterministic: the first n items of each type, with unanswerable ones kept."""
    out: list[Item] = []
    for kind, n in sizes.items():
        pool = [i for i in items if i.type == kind]
        if kind == "analytical":
            unanswerable = [i for i in pool if not i.answerable]
            answerable = [i for i in pool if i.answerable]
            take_u = min(len(unanswerable), max(2, n * 3 // 8))
            pool = answerable[: n - take_u] + unanswerable[:take_u]
        else:
            pool = pool[:: max(1, len(pool) // n)][:n]
        out += pool
    return out


def retrieval_metrics(events: list[str], gold: list[str]) -> dict[str, float]:
    row = {f"recall@{k}": metrics.recall_at_k(events, gold, k) for k in KS}
    row["mrr"] = metrics.reciprocal_rank(events, gold)
    row["ndcg@10"] = metrics.ndcg_at_k(events, gold, 10)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean metrics overall and per question type; latency percentiles."""
    out: dict[str, Any] = {}
    for scope in ("all", "lookup", "structured", "analytical"):
        scoped = [r for r in rows if scope == "all" or r["type"] == scope]
        if not scoped:
            continue
        # Citation metrics exist only for answered questions; each mean is over the rows
        # that have the metric, and its count is reported as n_<metric> when it differs.
        keys = list(dict.fromkeys(k for r in scoped for k in r["metrics"]))
        out[scope] = {"n": len(scoped)}
        for key in keys:
            values = [r["metrics"][key] for r in scoped if key in r["metrics"]]
            out[scope][key] = statistics.fmean(values)
            if len(values) != len(scoped):
                out[scope][f"n_{key}"] = len(values)
    latencies = sorted(r["latency_ms"] for r in rows)
    out["latency_ms"] = {
        "p50": statistics.median(latencies),
        "p95": latencies[int(0.95 * (len(latencies) - 1))],
        "mean": statistics.fmean(latencies),
    }
    return out


def run_retrieval(
    experiment: Experiment, retriever: Retriever, out_dir: Path
) -> dict[str, dict[str, Any]]:
    items = [i for i in read_items(experiment.questions) if i.answerable]
    plans = load_plans(experiment.plans)
    summaries: dict[str, dict[str, Any]] = {}
    with (out_dir / "results.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for config in experiment.configs:
            rows = []
            for item in tqdm(items, desc=config.name, leave=False):
                started = time.perf_counter()
                result = retriever.retrieve(item.question, config, plans.get(item.id))
                latency = (time.perf_counter() - started) * 1000
                events = result.events[:TOP_EVENTS]
                row = {
                    "config": config.name, "id": item.id, "type": item.type,
                    "events": events, "latency_ms": latency,
                    "filtered_events": result.filtered_events,
                    "metrics": retrieval_metrics(events, item.gold_events),
                }  # fmt: skip
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
            summaries[config.name] = summarize(rows)
            all_ = summaries[config.name]["all"]
            print(f"{config.name:40} R@10={all_['recall@10']:.3f} MRR={all_['mrr']:.3f}")
    return summaries


def run_end_to_end(
    experiment: Experiment, retriever: Retriever, llm: LLM, judge: LLM, out_dir: Path,
    workers: int = 4,
) -> dict[str, dict[str, Any]]:  # fmt: skip
    items = read_items(experiment.questions)
    if experiment.subset:
        items = stratified_subset(items, experiment.subset)
    plans = load_plans(experiment.plans)
    summaries: dict[str, dict[str, Any]] = {}
    with (out_dir / "results.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for config in experiment.configs:
            prepared = [prepare(i.question, config, retriever, plans.get(i.id)) for i in items]
            calls_before, cost_before = llm.usage.calls, llm.usage.cost_usd
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(
                    tqdm(pool.map(lambda p: complete(p, llm), prepared), total=len(prepared),
                         desc=config.name)
                )  # fmt: skip
            verdicts = judge_answers(judge, items, results, tag=f"judge/{config.name}")
            rows = []
            for item, result, verdict in zip(items, results, verdicts, strict=True):
                v = result.verification
                row: dict[str, Any] = {
                    "config": config.name, "id": item.id, "type": item.type,
                    "question": item.question, "gold_answer": item.gold_answer,
                    "answerable": item.answerable, "answer": result.answer,
                    "draft": result.draft, "refused": result.refused,
                    "refusal_reason": result.refusal_reason,
                    "top_similarity": result.retrieval.top_similarity,
                    "context_events": _context_events(result.context_keys),
                    "latency_ms": sum(result.timings_ms.values()),
                    "timings_ms": result.timings_ms,
                    "verdict": verdict,
                    "metrics": {
                        "correct": float(verdict["correct"]),
                        **retrieval_metrics(result.retrieval.events[:TOP_EVENTS], item.gold_events),
                        **({"citation_validity": v.citation_validity,
                            "citation_precision": v.citation_precision,
                            "supported_fraction": v.supported_fraction,
                            "gold_citation_rate": _gold_citation_rate(v.citations, item)}
                           if v else {}),
                    },
                }  # fmt: skip
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary = summarize(rows)
            summary["refusal_rate"] = statistics.fmean(r["refused"] for r in rows)
            summary["generation_calls"] = llm.usage.calls - calls_before
            summary["cost_usd"] = llm.usage.cost_usd - cost_before
            summary["cost_per_question_usd"] = summary["cost_usd"] / len(rows)
            summaries[config.name] = summary
            print(f"{config.name:40} correct={summary['all']['correct']:.3f}")
    return summaries


def _context_events(keys: list[str]) -> list[str]:
    return list(dict.fromkeys(k.split(":")[0] for k in keys))


def _gold_citation_rate(citations: list[str], item: Item) -> float:
    if not citations or not item.gold_events:
        return 0.0
    gold = set(item.gold_events)
    return sum(c.split(":")[0] in gold for c in citations) / len(citations)


def write_summary(
    out_dir: Path, summaries: dict[str, dict[str, Any]], extra: dict[str, Any]
) -> None:
    payload = {"configs": summaries, **extra}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def iter_results(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
