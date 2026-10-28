"""`pc` command-line interface."""

import io
import json
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from probable_cause import db
from probable_cause.config import load_settings
from probable_cause.eval import dataset, metrics, report, runner
from probable_cause.ingest import download, load, mdb, pdfs, stats
from probable_cause.llm import llm_from_env
from probable_cause.rag import index
from probable_cause.rag.chunking import STRATEGIES, Strategy
from probable_cause.rag.retrieve import Retriever
from probable_cause.rag.rewrite import rewrite_batch
from probable_cause.search import search as run_search

app = typer.Typer(no_args_is_help=True, add_completion=False)
ingest_app = typer.Typer(no_args_is_help=True, help="Download, load, and profile NTSB data.")
db_app = typer.Typer(no_args_is_help=True, help="Database maintenance.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(db_app, name="db")


@app.callback()
def main() -> None:
    """Probable Cause: benchmarked RAG over NTSB aviation accident investigations."""
    # Narratives are full of curly quotes and degree signs; Windows consoles default to cp1252.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")


@db_app.command("migrate")
def migrate() -> None:
    """Apply pending SQL migrations."""
    settings = load_settings()
    with db.connect(settings.database_url) as conn:
        applied = db.migrate(conn)
    typer.echo(f"applied: {', '.join(applied) or 'nothing (up to date)'}")


@ingest_app.command("download")
def ingest_download() -> None:
    """Fetch avall.zip and reference PDFs from data.ntsb.gov/avdata.

    Cached by listing date + size and verified by SHA-256.
    """
    settings = load_settings()
    for name, status in download.sync(settings.downloads_dir):
        typer.echo(f"{name}: {status}")


@ingest_app.command("load")
def ingest_load(
    force: Annotated[
        bool, typer.Option(help="Reload even if this snapshot is already loaded.")
    ] = False,
) -> None:
    """Extract avall.mdb, export tables via mdbtools, normalize, and load into Postgres."""
    settings = load_settings()
    manifest = download.read_manifest(settings.downloads_dir)
    entry = manifest.get("avall.zip")
    if entry is None:
        raise typer.BadParameter("avall.zip not downloaded yet; run `pc ingest download` first")
    archive = settings.downloads_dir / "avall.zip"
    if download.sha256_file(archive) != entry.sha256:
        raise typer.BadParameter("avall.zip does not match its manifest checksum; re-download it")

    mdb_path = download.extract_mdb(archive, settings.extracted_dir, entry.sha256)
    csv_dir = settings.csv_dir / f"avall-{entry.sha256[:12]}"
    typer.echo(f"exporting tables from {mdb_path} -> {csv_dir}")
    mdb.export_tables(mdb_path, csv_dir)

    snapshot = load.Snapshot(entry.name, entry.published, entry.size, entry.sha256)
    with db.connect(settings.database_url) as conn:
        db.migrate(conn)
        report = load.load_snapshot(conn, csv_dir, snapshot, force=force)
    if report.skipped:
        typer.echo("snapshot already loaded (use --force to reload)")
        return
    for table, count in report.loaded.items():
        typer.echo(f"{table:12} {count:>8,}")
    for label, counts in (
        ("duplicates", report.duplicates_dropped),
        ("orphans", report.orphans_dropped),
    ):
        dropped = {k: v for k, v in counts.items() if v}
        typer.echo(f"{label} dropped: {dropped or 'none'}")


@ingest_app.command("pdfs")
def ingest_pdfs(
    reports: Annotated[int, typer.Option(help="CAROL final-report PDFs to sample.")] = 15,
    dockets: Annotated[int, typer.Option(help="Docket documents to sample.")] = 10,
    ocr: Annotated[
        bool, typer.Option(help="OCR pages without a text layer (Docker tesseract).")
    ] = True,
) -> None:
    """Fetch a small, deterministic PDF sample and extract its text into `documents`."""
    settings = load_settings()
    engine: pdfs.OcrEngine = pdfs.DockerTesseract() if ocr else pdfs.NoOcr()
    with db.connect(settings.database_url) as conn:
        result = pdfs.run_sample(
            conn, settings.pdf_dir, n_reports=reports, n_dockets=dockets, ocr=engine
        )
    for target, ex in result.stored:
        q = ex.issues
        typer.echo(
            f"{target.ntsb_no:12} {target.source:12} {ex.method:8} pages={ex.pages:<3} "
            f"empty={len(ex.empty_pages):<3} chars={q['chars']:<7} "
            f"garbled={q['garbled_word_ratio']}"
        )
    for problem in result.problems:
        typer.echo(f"skipped: {problem}")


@ingest_app.command("stats")
def ingest_stats(
    write: Annotated[Path | None, typer.Option(help="Also write the report to this file.")] = None,
) -> None:
    """Print the data-quality report (Markdown)."""
    settings = load_settings()
    with db.connect(settings.database_url) as conn:
        markdown = stats.render_markdown(stats.collect(conn))
    typer.echo(markdown)
    if write is not None:
        write.write_text(markdown, encoding="utf-8")


@app.command("search")
def search(
    query: Annotated[str, typer.Argument(help="Web-search style query, e.g. 'icing \"part 91\"'.")],
    year_from: Annotated[int | None, typer.Option(help="Earliest event year.")] = None,
    year_to: Annotated[int | None, typer.Option(help="Latest event year.")] = None,
    limit: Annotated[int, typer.Option(help="Events to return.")] = 10,
    any_term: Annotated[bool, typer.Option("--any", help="Match any term instead of all.")] = False,
) -> None:
    """Full-text search over probable-cause, analysis, and factual narratives."""
    settings = load_settings()
    with db.connect(settings.database_url) as conn:
        hits = run_search(
            conn, query, year_from=year_from, year_to=year_to, limit=limit, match_any=any_term
        )
    if not hits:
        typer.echo("no matches (try --any)")
    for i, hit in enumerate(hits, 1):
        typer.echo(
            f"{i:>2}. {hit.ev_id}  {hit.ntsb_no or '-':11} {hit.ev_date}  "
            f"{hit.highest_injury or '-':8} [{hit.section}] score={hit.score:.3f}"
        )
        typer.echo(f"    {hit.snippet}")
        typer.echo(f"    {hit.source_url}")


index_app = typer.Typer(no_args_is_help=True, help="Chunk, embed, and index the corpus.")
app.add_typer(index_app, name="index")

StrategyOption = Annotated[
    list[str] | None, typer.Option("--strategy", "-s", help="Repeatable; default: all three.")
]


def _strategies(values: list[str] | None) -> list[Strategy]:
    chosen = values or list(STRATEGIES)
    unknown = [v for v in chosen if v not in STRATEGIES]
    if unknown:
        raise typer.BadParameter(f"unknown strategy {unknown}; choose from {STRATEGIES}")
    return [s for s in STRATEGIES if s in chosen]


@index_app.command("chunk")
def index_chunk(strategy: StrategyOption = None) -> None:
    """(Re)build chunks for each strategy from the documents table."""
    settings = load_settings()
    with db.connect(settings.database_url) as conn:
        db.migrate(conn)
        for s in _strategies(strategy):
            started = time.perf_counter()
            n = index.build_chunks(conn, s)
            typer.echo(f"{s}: {n:,} chunks in {time.perf_counter() - started:.0f}s")


@index_app.command("embed")
def index_embed(strategy: StrategyOption = None) -> None:
    """Embed chunks that have no vector yet. Safe to interrupt and re-run."""
    settings = load_settings()
    with db.connect(settings.database_url) as conn:
        for s in _strategies(strategy):
            report = index.embed_pending(conn, s)
            rate = report.embedded / report.seconds if report.seconds else 0.0
            typer.echo(f"{s}: embedded {report.embedded:,} in {report.seconds:.0f}s ({rate:.1f}/s)")


@index_app.command("build")
def index_build(
    strategy: StrategyOption = None,
    hnsw: Annotated[bool, typer.Option(help="Build the pgvector HNSW index.")] = True,
    bm25: Annotated[bool, typer.Option(help="Build the BM25 index.")] = True,
) -> None:
    """Build the HNSW and BM25 indexes."""
    settings = load_settings()
    with db.connect(settings.database_url) as conn:
        for s in _strategies(strategy):
            if hnsw:
                seconds = index.build_hnsw(conn, s)
                typer.echo(f"{s}: HNSW built in {seconds:.0f}s")
            if not bm25:
                continue
            started = time.perf_counter()
            n = index.build_bm25(conn, s, settings.index_dir)
            typer.echo(f"{s}: BM25 over {n:,} chunks in {time.perf_counter() - started:.0f}s")


bench_app = typer.Typer(no_args_is_help=True, help="Build the benchmark dataset.")
app.add_typer(bench_app, name="bench")
BENCH_DIR = Path("benchmark")


@bench_app.command("lookup")
def bench_lookup(per_year: int = 20, first_year: int = 2018, last_year: int = 2025) -> None:
    """Generate lookup questions with the LLM (cached) and filter out leaky ones."""
    settings = load_settings()
    llm = llm_from_env()
    with db.connect(settings.database_url) as conn:
        items, rejected = dataset.generate_lookup(
            llm, conn, per_year=per_year, years=range(first_year, last_year + 1)
        )
    dataset.write_items(BENCH_DIR / "sources" / "lookup.jsonl", items)
    (BENCH_DIR / "sources" / "lookup_rejected.json").write_text(
        json.dumps(rejected, indent=1, default=str) + "\n", encoding="utf-8"
    )
    typer.echo(f"kept {len(items)}, rejected {len(rejected)}; LLM calls {llm.usage.calls}")


@bench_app.command("structured")
def bench_structured(per_template: int = 24) -> None:
    """Generate templated structured questions with SQL ground truth."""
    settings = load_settings()
    with db.connect(settings.database_url) as conn:
        items = dataset.generate_structured(conn, per_template)
    dataset.write_items(BENCH_DIR / "sources" / "structured.jsonl", items)
    typer.echo(f"{len(items)} structured questions")


@bench_app.command("assemble")
def bench_assemble() -> None:
    """Combine all sources (and analytical.yaml) into benchmark/questions.jsonl."""
    settings = load_settings()
    items = dataset.read_items(BENCH_DIR / "sources" / "lookup.jsonl")
    items += dataset.read_items(BENCH_DIR / "sources" / "structured.jsonl")
    with db.connect(settings.database_url) as conn:
        items += dataset.load_analytical(conn, BENCH_DIR / "analytical.yaml")
    dataset.write_items(BENCH_DIR / "questions.jsonl", dataset.number(items))
    counts = {t: sum(i.type == t for i in items) for t in ("lookup", "structured", "analytical")}
    typer.echo(f"{len(items)} questions: {counts}")


@bench_app.command("rewrite")
def bench_rewrite(batch_size: int = 25) -> None:
    """Rewrite every benchmark question into a query plan (batched, cached LLM calls)."""
    llm = llm_from_env()
    items = dataset.read_items(BENCH_DIR / "questions.jsonl")
    plans: dict[str, Any] = {}
    for start in range(0, len(items), batch_size):
        batch = {i.id: i.question for i in items[start : start + batch_size]}
        for qid, plan in rewrite_batch(llm, batch, tag=f"rewrite/{start}").items():
            plans[qid] = plan.model_dump()
    with (BENCH_DIR / "plans.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps({"id": item.id, "plan": plans[item.id]}) + "\n")
    typer.echo(f"{len(plans)} plans; LLM calls {llm.usage.calls}")


eval_app = typer.Typer(no_args_is_help=True, help="Run experiments and report results.")
app.add_typer(eval_app, name="eval")
RUNS_DIR = BENCH_DIR / "runs"


@eval_app.command("run")
def eval_run(
    config: Annotated[Path, typer.Option(help="Experiment YAML in experiments/.")],
    workers: int = 4,
) -> None:
    """Run a retrieval grid or an end-to-end experiment."""
    settings = load_settings()
    experiment = runner.load_experiment(config)
    out_dir = RUNS_DIR / experiment.name
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with db.connect(settings.database_url) as conn:
        retriever = Retriever(conn, settings.index_dir)
        if experiment.mode == "retrieval":
            summaries = runner.run_retrieval(experiment, retriever, out_dir)
            extra: dict[str, Any] = {}
        else:
            llm = llm_from_env()
            judge = llm_from_env()
            summaries = runner.run_end_to_end(
                experiment, retriever, llm, judge, out_dir, workers=workers
            )
            extra = {"generation_usage": vars(llm.usage), "judge_usage": vars(judge.usage)}
    extra["wall_clock_s"] = time.perf_counter() - started
    runner.write_summary(out_dir, summaries, extra)
    typer.echo(f"wrote {out_dir} in {extra['wall_clock_s']:.0f}s")


@eval_app.command("report")
def eval_report(out: Path = Path("docs/img")) -> None:
    """Print result tables (Markdown) and write the charts."""
    grid = report.load_summary(RUNS_DIR / "retrieval-grid")["configs"]
    out.mkdir(parents=True, exist_ok=True)
    report.grid_chart(grid, out / "retrieval-grid.png")
    compared = ["fixed+bm25", "section+hybrid+rerank+rewrite", "parent_child+hybrid+rerank+rewrite"]
    report.by_type_chart(grid, compared, out / "recall-by-type.png")
    typer.echo("## Recall@10 grid\n\n" + report.grid_table(grid))
    typer.echo("\n## MRR grid\n\n" + report.grid_table(grid, "mrr"))
    typer.echo("\n## Top configs\n\n" + report.ranked_table(grid, top=12))


@eval_app.command("calibrate")
def eval_calibrate(labels: Path = BENCH_DIR / "human_labels.jsonl") -> None:
    """Agreement between the LLM judge and hand labels on the same answers."""
    verdicts = {
        (row["config"], row["id"]): row["verdict"]
        for run in ("end-to-end-baseline", "end-to-end-best")
        for row in runner.iter_results(RUNS_DIR / run / "results.jsonl")
    }
    pairs = []
    for line in labels.read_text(encoding="utf-8").splitlines():
        human = json.loads(line)
        verdict = verdicts[(human["config"], human["id"])]
        if verdict["by"] == "llm":
            pairs.append((bool(human["correct"]), bool(verdict["correct"]), human))
    agree = sum(h == j for h, j, _ in pairs)
    kappa = metrics.cohens_kappa([h for h, _, _ in pairs], [j for _, j, _ in pairs])
    typer.echo(f"items {len(pairs)}  agreement {agree}/{len(pairs)} = {agree / len(pairs):.3f}  "
               f"Cohen's kappa {kappa:.3f}")  # fmt: skip
    typer.echo(f"human correct rate {sum(h for h, _, _ in pairs) / len(pairs):.3f}, "
               f"judge correct rate {sum(j for _, j, _ in pairs) / len(pairs):.3f}")  # fmt: skip
    for h, j, item in pairs:
        if h != j:
            typer.echo(
                f"disagree {item['config']} {item['id']}: human={h} judge={j} ({item['note']})"
            )
