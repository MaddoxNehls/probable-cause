"""`pc` command-line interface."""

import io
import sys
from pathlib import Path
from typing import Annotated

import typer

from probable_cause import db
from probable_cause.config import load_settings
from probable_cause.ingest import download, load, mdb, pdfs, stats
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
