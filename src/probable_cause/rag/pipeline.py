"""The full question-answering pipeline: rewrite -> retrieve -> (refuse?) -> generate -> verify."""

import time
from dataclasses import dataclass, field
from typing import Any

from probable_cause import tracing
from probable_cause.llm import LLM
from probable_cause.rag.filters import QueryPlan
from probable_cause.rag.generate import (
    CONTEXT_WORD_LIMIT,
    INSUFFICIENT,
    EventInfo,
    generate,
)
from probable_cause.rag.retrieve import (
    RetrievalConfig,
    RetrievalResult,
    RetrievedChunk,
    Retriever,
)
from probable_cause.rag.rewrite import rewrite
from probable_cause.rag.verify import Verification, llm_recheck, verify

# Below this cosine similarity between the query and the best context chunk, retrieval is
# treated as too weak to answer from (calibrated on the benchmark; see docs/results.md).
MIN_SIMILARITY = 0.66
# An answer whose sentences are mostly unsupported is withheld.
MIN_SUPPORTED = 0.5


@dataclass
class AskResult:
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None
    retrieval: RetrievalResult
    context_keys: list[str]
    verification: Verification | None
    draft: str | None = None  # the generated answer when it was withheld
    timings_ms: dict[str, float] = field(default_factory=dict)
    llm_calls: int = 0
    cost_usd: float = 0.0


def event_info(retriever: Retriever, ev_ids: list[str]) -> dict[str, EventInfo]:
    if not ev_ids:
        return {}
    rows = retriever.conn.execute(
        "SELECT e.ev_id, e.ntsb_no, e.ev_date::text, concat_ws(', ', e.city, e.state, e.country), "
        "coalesce(string_agg(DISTINCT concat_ws(' ', a.make, a.model), '; '), 'unknown aircraft'), "
        "e.highest_injury FROM events e LEFT JOIN aircraft a USING (ev_id) "
        "WHERE e.ev_id = ANY(%s) GROUP BY e.ev_id",
        (ev_ids,),
    ).fetchall()
    return {
        str(r[0]): EventInfo(str(r[0]), _opt(r[1]), str(r[2]), str(r[3]), str(r[4]), _opt(r[5]))
        for r in rows
    }


def _opt(value: object) -> str | None:
    return None if value is None else str(value)


def shown_text(context_text: str) -> str:
    words = context_text.split()
    return " ".join(words[:CONTEXT_WORD_LIMIT])


@dataclass
class Prepared:
    """Everything up to (not including) generation: cheap, deterministic, DB-bound."""

    question: str
    config: RetrievalConfig
    retrieval: RetrievalResult
    context: list[RetrievedChunk]
    shown: dict[str, str]  # citation key -> text shown to the model
    events: dict[str, EventInfo]
    refusal_reason: str | None
    timings_ms: dict[str, float]


def prepare(
    question: str, config: RetrievalConfig, retriever: Retriever, plan: QueryPlan | None
) -> Prepared:
    timings: dict[str, float] = {}
    started = time.perf_counter()
    retrieval = retriever.retrieve(question, config, plan)
    context = retrieval.context(config.context_chunks)
    retrieval.top_similarity = retriever.score_similarity(
        retrieval.text_query, [c.chunk_id for c in context]
    )
    timings["retrieve"] = (time.perf_counter() - started) * 1000
    shown: dict[str, str] = {}
    for chunk in context:
        previous = shown.get(chunk.citation_key)
        text = shown_text(chunk.context_text)
        shown[chunk.citation_key] = f"{previous}\n{text}" if previous else text
    refusal = None
    # The similarity gate only applies to purely semantic retrieval: when structured filters
    # selected the events, "how many fatal Part 121 accidents in 2018?" legitimately has low
    # cosine similarity to every narrative, yet the filtered events are exactly the evidence.
    filtered = bool(retrieval.filtered_events)
    if not context:
        refusal = "no matching records"
    elif not filtered and retrieval.top_similarity < MIN_SIMILARITY:
        refusal = (
            f"weak retrieval (best similarity {retrieval.top_similarity:.2f} < {MIN_SIMILARITY})"
        )
    return Prepared(
        question, config, retrieval, context, shown,
        event_info(retriever, [c.ev_id for c in context]), refusal, timings,
    )  # fmt: skip


def complete(prepared: Prepared, llm: LLM, *, llm_verify: bool = False) -> AskResult:
    """Generate and verify, or refuse. Safe to call from worker threads."""
    timings = dict(prepared.timings_ms)
    result = AskResult(
        question=prepared.question,
        answer=INSUFFICIENT,
        refused=True,
        refusal_reason=prepared.refusal_reason,
        retrieval=prepared.retrieval,
        context_keys=list(prepared.shown),
        verification=None,
        timings_ms=timings,
    )
    if prepared.refusal_reason:
        return result
    started = time.perf_counter()
    answer = generate(llm, prepared.question, prepared.context, prepared.events)
    timings["generate"] = (time.perf_counter() - started) * 1000
    if answer.insufficient_evidence:
        result.refusal_reason = "the model found the excerpts insufficient"
        return result
    started = time.perf_counter()
    with tracing.span("verify"):
        verification = verify(answer.answer, prepared.shown)
        if llm_verify:
            llm_recheck(llm, verification, prepared.shown)
    timings["verify"] = (time.perf_counter() - started) * 1000
    result.verification = verification
    if verification.supported_fraction < MIN_SUPPORTED:
        result.refusal_reason = (
            f"only {verification.supported_fraction:.0%} of sentences are supported "
            "by their citations"
        )
        result.draft = answer.answer
    else:
        result.answer, result.refused = answer.answer, False
    return result


def ask(
    question: str,
    config: RetrievalConfig,
    retriever: Retriever,
    llm: LLM,
    *,
    plan: QueryPlan | None = None,
    llm_verify: bool = False,
) -> AskResult:
    calls_before, cost_before = llm.usage.calls, llm.usage.cost_usd
    with tracing.run("pc ask", input=question, tags=[config.name]) as run:
        rewrite_ms = 0.0
        if config.rewrite and plan is None:
            started = time.perf_counter()
            plan = rewrite(llm, question)
            rewrite_ms = (time.perf_counter() - started) * 1000
        prepared = prepare(question, config, retriever, plan)
        if rewrite_ms:
            prepared.timings_ms["rewrite"] = rewrite_ms
        result = complete(prepared, llm, llm_verify=llm_verify)
        result.llm_calls = llm.usage.calls - calls_before
        result.cost_usd = llm.usage.cost_usd - cost_before
        run.set_output({"answer": result.answer, "refused": result.refused})
    return result


def debug_payload(result: AskResult) -> dict[str, Any]:
    r = result.retrieval
    return {
        "rewritten_query": r.text_query,
        "plan": r.plan.model_dump() if r.plan else None,
        "filtered_events": r.filtered_events,
        "top_similarity": round(r.top_similarity, 4),
        "timings_ms": {**r.timings_ms, **result.timings_ms},
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "key": c.citation_key,
                "heading": c.heading,
                "score": round(c.score, 5),
                "ranks": c.ranks,
                "scores": {k: round(v, 5) for k, v in c.scores.items()},
                "in_context": c.citation_key in result.context_keys,
                "text": c.match_text[:500],
            }
            for c in r.chunks[:20]
        ],
        "rerank": [
            {
                "chunk_id": c.chunk_id,
                "before": c.ranks.get("rrf") or c.ranks.get("bm25") or c.ranks.get("dense"),
                "after": c.ranks["rerank"],
                "score": c.scores["rerank"],
            }
            for c in r.chunks
            if "rerank" in c.ranks
        ],
    }
