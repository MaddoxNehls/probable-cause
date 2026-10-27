"""Report-PDF path for a small sample: fetch, extract text, OCR the pages that need it.

Two PDF sources exist:
* CAROL final reports, generated on demand from the CAROL key embedded in post-2020
  ev_ids. Born-digital, so the text layer is usually good.
* Public docket documents (data.ntsb.gov/Docket) for older events. These are often
  scans of typed or handwritten forms with no text layer at all.
"""

import hashlib
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urljoin

import httpx
import psycopg
from psycopg.types.json import Jsonb
from pypdf import PdfReader
from pypdf.errors import PyPdfError as PdfError

from probable_cause.ingest import mdb

type Conn = psycopg.Connection[tuple[object, ...]]

DOCKET_BASE = "https://data.ntsb.gov/Docket/"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) probable-cause/0.1 research ingest"
REQUEST_INTERVAL_S = 1.0
MIN_PAGE_CHARS = 25  # below this a page is treated as having no usable text layer
PAGE_BREAK = "\n\f\n"  # form feed between pages keeps page boundaries recoverable

GARBLED_PAGE_RATIO = 0.10  # above this a text layer is presumed to be bad embedded OCR
_EDGE_PUNCT = re.compile(r"^[\W_]+|[\W_]+$")
# Shapes that real English words almost never have: case flips mid-word ("TransPOrtation"),
# digits inside words ("T1m", "1ts"), runs of dots, or stray symbols.
_SUSPECT = re.compile(r"[a-z][A-Z]|[A-Za-z]\d[a-z]|\d[a-z]{2}|\.\.|[^\w'\u2019\-(),.:]")
_DOCKET_LINK = re.compile(r'href="(?P<href>/Docket/Document/docBLOB\?ID=\d+[^"]*)"')
_FILE_NAME = re.compile(r"FileName=(?P<name>[^&\"]+)")
# Prefer the pilot/operator report (often handwritten, the hardest OCR case), then any
# typed specialist factual report.
DOCKET_PREFERENCES = (
    re.compile(r"6120\.1(?!\d)"),  # not 6120.15, the boilerplate wreckage-release form
    re.compile(r"^(?!Figures).*Factual Report", re.I),
)


class OcrEngine(Protocol):
    name: str

    def ocr_pages(self, pdf: Path, pages: Sequence[int]) -> dict[int, str]:
        """Return OCR text keyed by 1-based page number."""
        ...


class NoOcr:
    """Used with --no-ocr: pages without a text layer stay empty and are flagged."""

    name = "none"

    def ocr_pages(self, pdf: Path, pages: Sequence[int]) -> dict[int, str]:
        return {}


class DockerTesseract:
    """Tesseract in the sandboxed tools image, so no local install (or admin rights) is needed."""

    name = "tesseract-docker"

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi

    def ocr_pages(self, pdf: Path, pages: Sequence[int]) -> dict[int, str]:
        if not pages:
            return {}
        mdb.ensure_tools_image()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            page_list = " ".join(str(p) for p in pages)
            script = (
                f"set -e; for p in {page_list}; do "
                f"pdftoppm -f $p -l $p -r {self.dpi} -gray -png -singlefile "
                f"/in/{pdf.name} /tmp/page; "
                "tesseract /tmp/page.png /out/page-$p >/dev/null 2>&1; done"
            )
            mdb.sandboxed_run({pdf.parent: ("/in", True), out: ("/out", False)}, script)
            return {
                p: (out / f"page-{p}.txt").read_text(encoding="utf-8")
                for p in pages
                if (out / f"page-{p}.txt").exists()
            }


@dataclass
class Extraction:
    text: str
    method: str
    pages: int
    empty_pages: list[int] = field(default_factory=list)
    garbled_pages: list[int] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)
    issues: dict[str, float | int] = field(default_factory=dict)


def garbled_ratio(text: str) -> float:
    """Share of lowercase-bearing words with OCR-damage shapes. Clean prose scores < 0.03."""
    words = [
        core
        for token in re.split(r"[\s/]+", text)
        if len(core := _EDGE_PUNCT.sub("", token)) >= 2 and any(c.islower() for c in core)
    ]
    if not words:
        return 0.0
    bad = sum(1 for w in words if _SUSPECT.search(w.removeprefix("Mc").removeprefix("iP")))
    return round(bad / len(words), 3)


def quality_issues(text: str) -> dict[str, float | int]:
    """Cheap signals of extraction damage, recorded per document for the write-up."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    repeated = sum(1 for ln in set(lines) if len(ln) > 10 and lines.count(ln) >= 3)
    return {
        "chars": len(text),
        "garbled_word_ratio": garbled_ratio(text),
        "hyphen_breaks": len(re.findall(r"\w-\n\w", text)),
        "repeated_lines": repeated,
        "replacement_chars": text.count("�"),
        "non_ascii_ratio": round(sum(ord(c) > 127 for c in text) / len(text), 4) if text else 0.0,
    }


def extract_pdf(path: Path, ocr: OcrEngine) -> Extraction:
    """Use the PDF text layer where it is usable; OCR pages where it is missing or garbled.

    An OCR result only replaces an existing text layer if it scores as less garbled.
    """
    reader = PdfReader(path)
    texts = [(page.extract_text() or "").strip() for page in reader.pages]
    empty = [i + 1 for i, t in enumerate(texts) if len(t) < MIN_PAGE_CHARS]
    garbled = [
        i + 1
        for i, t in enumerate(texts)
        if len(t) >= MIN_PAGE_CHARS and garbled_ratio(t) > GARBLED_PAGE_RATIO
    ]
    ocr_pages: list[int] = []
    for page_no, text in ocr.ocr_pages(path, sorted(empty + garbled)).items():
        candidate = text.strip()
        current = texts[page_no - 1]
        if not candidate:
            continue
        if page_no in empty or garbled_ratio(candidate) < garbled_ratio(current):
            texts[page_no - 1] = candidate
            ocr_pages.append(page_no)
    if not ocr_pages:
        method = "pdf_text"
    elif len(ocr_pages) == len(texts):
        method = "ocr"
    else:
        method = "mixed"
    body = PAGE_BREAK.join(texts).strip()
    return Extraction(
        text=body,
        method=method,
        pages=len(texts),
        empty_pages=empty,
        garbled_pages=garbled,
        ocr_pages=sorted(ocr_pages),
        issues=quality_issues(body),
    )


@dataclass(frozen=True)
class PdfTarget:
    ev_id: str
    ntsb_no: str
    section: str  # report_pdf | docket_pdf
    source: str  # carol_report | docket
    url: str
    title: str
    page_url: str


def parse_docket_documents(html: str) -> list[tuple[str, str]]:
    """Return (title, absolute url) for each document linked from a docket page."""
    docs: list[tuple[str, str]] = []
    for m in _DOCKET_LINK.finditer(html):
        href = unescape(m["href"])
        name_match = _FILE_NAME.search(href)
        title = unquote(name_match["name"]) if name_match else href
        docs.append((title, urljoin(DOCKET_BASE, href)))
    return docs


def choose_docket_document(docs: list[tuple[str, str]]) -> tuple[str, str] | None:
    for pattern in DOCKET_PREFERENCES:
        for title, url in docs:
            if pattern.search(title):
                return title, url
    return None


_SAMPLE_REPORTS = """
SELECT e.ev_id, e.ntsb_no, e.report_url
FROM events e JOIN narratives n USING (ev_id)
WHERE e.report_url IS NOT NULL AND n.probable_cause IS NOT NULL AND n.aircraft_key = 1
ORDER BY md5(e.ev_id) LIMIT %s
"""

_SAMPLE_DOCKETS = """
SELECT e.ev_id, e.ntsb_no, e.docket_url
FROM events e
WHERE e.report_url IS NULL AND e.highest_injury = 'Fatal' AND e.ntsb_no IS NOT NULL
  AND e.ev_date < '2020-01-01' AND e.ntsb_no !~ 'WA[0-9]+$'
ORDER BY md5(e.ev_id) LIMIT %s
"""


class Fetcher:
    """Polite, cached HTTP fetches. Every response body lands in its own directory."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.client = httpx.Client(
            headers={"User-Agent": BROWSER_UA},
            timeout=httpx.Timeout(30, read=120),
            follow_redirects=True,
        )
        self._last = 0.0

    def _wait(self) -> None:
        delay = REQUEST_INTERVAL_S - (time.monotonic() - self._last)
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()

    def get_text(self, url: str) -> str:
        self._wait()
        response = self.client.get(url)
        response.raise_for_status()
        return response.text

    def get_pdf(self, url: str, dest: Path) -> Path:
        if dest.exists() and dest.read_bytes()[:5] == b"%PDF-":
            return dest
        self._wait()
        response = self.client.get(url)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF-"):
            raise ValueError(f"not a PDF: {url} ({response.headers.get('content-type')})")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        return dest

    def close(self) -> None:
        self.client.close()


def sample_targets(
    conn: Conn, fetcher: Fetcher, n_reports: int, n_dockets: int
) -> tuple[list[PdfTarget], list[str]]:
    targets: list[PdfTarget] = []
    problems: list[str] = []
    for ev_id, ntsb_no, url in conn.execute(_SAMPLE_REPORTS, (n_reports,)).fetchall():
        targets.append(
            PdfTarget(
                str(ev_id),
                str(ntsb_no),
                "report_pdf",
                "carol_report",
                str(url),
                f"{ntsb_no} - CAROL final report",
                str(url),
            )
        )
    # Over-sample dockets: many are unreleased or have no matching document.
    for ev_id, ntsb_no, docket_url in conn.execute(_SAMPLE_DOCKETS, (n_dockets * 4,)).fetchall():
        if sum(t.source == "docket" for t in targets) >= n_dockets:
            break
        try:
            docs = parse_docket_documents(fetcher.get_text(str(docket_url)))
        except httpx.HTTPError as exc:
            problems.append(f"{ntsb_no}: docket page failed ({exc})")
            continue
        chosen = choose_docket_document(docs)
        if chosen is None:
            reason = "no pilot or factual report among" if docs else "docket not released,"
            problems.append(f"{ntsb_no}: {reason} {len(docs)} documents")
            continue
        title, doc_url = chosen
        targets.append(
            PdfTarget(
                str(ev_id),
                str(ntsb_no),
                "docket_pdf",
                "docket",
                doc_url,
                f"{ntsb_no} - {title}",
                str(docket_url),
            )
        )
    return targets, problems


def store_extraction(
    conn: Conn, target: PdfTarget, extraction: Extraction, ocr_engine: str
) -> None:
    metadata = {
        "ntsb_no": target.ntsb_no,
        "page_url": target.page_url,
        "pages": extraction.pages,
        "empty_pages": len(extraction.empty_pages),
        "garbled_pages": extraction.garbled_pages,
        "ocr_pages": extraction.ocr_pages,
        "ocr_engine": ocr_engine,
        "quality": extraction.issues,
    }
    conn.execute(
        "INSERT INTO documents (ev_id, aircraft_key, section, source, source_url, title, body, "
        "extraction_method, metadata) VALUES (%s, NULL, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (ev_id, aircraft_key, section, source_url) DO UPDATE SET "
        "title = EXCLUDED.title, body = EXCLUDED.body, "
        "extraction_method = EXCLUDED.extraction_method, metadata = EXCLUDED.metadata",
        (
            target.ev_id,
            target.section,
            target.source,
            target.url,
            target.title,
            extraction.text,
            extraction.method,
            Jsonb(metadata),
        ),
    )


@dataclass
class PdfRunReport:
    stored: list[tuple[PdfTarget, Extraction]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def run_sample(
    conn: Conn, pdf_dir: Path, *, n_reports: int, n_dockets: int, ocr: OcrEngine
) -> PdfRunReport:
    report = PdfRunReport()
    fetcher = Fetcher(pdf_dir)
    try:
        targets, report.problems = sample_targets(conn, fetcher, n_reports, n_dockets)
        for target in targets:
            # One directory per URL: the cache key changes with the document, and the
            # OCR container mount exposes exactly one untrusted file.
            url_key = hashlib.sha256(target.url.encode()).hexdigest()[:16]
            dest = pdf_dir / target.source / target.ev_id / url_key / "document.pdf"
            try:
                path = fetcher.get_pdf(target.url, dest)
            except (httpx.HTTPError, ValueError) as exc:
                report.problems.append(f"{target.ntsb_no}: download failed ({exc})")
                shutil.rmtree(dest.parent, ignore_errors=True)
                continue
            try:
                extraction = extract_pdf(path, ocr)
            except (PdfError, subprocess.CalledProcessError) as exc:
                report.problems.append(f"{target.ntsb_no}: extraction failed ({exc!r})")
                continue
            with conn.transaction():
                store_extraction(conn, target, extraction, ocr.name)
            report.stored.append((target, extraction))
    finally:
        fetcher.close()
    return report
