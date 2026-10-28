"""Three chunking strategies behind one interface.

* ``fixed``: the naive baseline. All of an event's text is concatenated in reading order
  and cut into overlapping word windows, ignoring structure.
* ``section``: never crosses a section boundary. Database narratives are already split
  into factual / analysis / probable cause; factual narratives and PDFs are further split on
  their own headings ("HISTORY OF FLIGHT", "Probable Cause and Findings"). Each chunk is
  prefixed with its section title.
* ``parent_child``: small child windows are what gets indexed and matched; the larger
  section-aware parent they belong to is what the generator reads.

Sizes are counted in words, a stable proxy for tokens (about 1.3 tokens per word here).
"""

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

type Strategy = Literal["fixed", "section", "parent_child"]
STRATEGIES: tuple[Strategy, ...] = ("fixed", "section", "parent_child")

# Reading order of an event's documents.
SECTION_ORDER = ("factual", "analysis", "probable_cause", "report_pdf", "docket_pdf")
SECTION_TITLES = {
    "factual": "Factual narrative",
    "analysis": "Analysis",
    "probable_cause": "Probable cause",
    "report_pdf": "Final report",
    "docket_pdf": "Docket document",
}


@dataclass(frozen=True)
class SourceDocument:
    document_id: int
    ev_id: str
    section: str  # documents.section; this is the citation key's section part
    body: str


@dataclass
class Chunk:
    ev_id: str
    document_id: int
    section: str
    heading: str | None
    ordinal: int
    text: str  # what is embedded and indexed
    char_start: int
    char_end: int
    children: list["Chunk"] = field(default_factory=list)  # parent_child only

    @property
    def citation_key(self) -> str:
        return f"{self.ev_id}:{self.section}"


class Chunker(Protocol):
    strategy: Strategy

    def chunk_event(self, documents: Sequence[SourceDocument]) -> list[Chunk]: ...


# ---------------------------------------------------------------------------
# Text cleanup and section splitting
# ---------------------------------------------------------------------------

_PAGE_HEADER = re.compile(r"(?m)^\s*Page \d+ of \d+\b.*$\n?")
_FORM_FEED = re.compile(r"\n?\f\n?")
# Upper-case headings ("HISTORY OF FLIGHT"), alone on a line or glued to the next word
# ("PERSONNEL INFORMATIONThe pilot ...", a common artifact in the 2008-2019 narratives).
_UPPER_HEADING = re.compile(
    r"(?m)^(?P<heading>[A-Z][A-Z&/,'-]*(?: [A-Z&/,'-]+){1,6})(?=[A-Z][a-z]|[ \t]*$)"
)
# Title-case headings used in CAROL final reports.
PDF_HEADINGS = (
    "Analysis",
    "Probable Cause and Findings",
    "Findings",
    "Factual Information",
    "History of Flight",
    "Pilot Information",
    "Student Pilot Information",
    "Flight Instructor Information",
    "Aircraft and Owner/Operator Information",
    "Meteorological Information and Flight Plan",
    "Airport Information",
    "Wreckage and Impact Information",
    "Medical and Pathological Information",
    "Tests and Research",
    "Additional Information",
    "Preventing Similar Accidents",
    "Administrative Information",
)
_PDF_HEADING = re.compile(
    r"(?m)^[ \t]*(?P<heading>" + "|".join(re.escape(h) for h in PDF_HEADINGS) + r")[ \t]*$"
)


def strip_pdf_noise(text: str) -> str:
    """Remove page headers and lines repeated on most pages (running headers/footers)."""
    pages = _FORM_FEED.split(text)
    if len(pages) > 2:
        counts = Counter(
            line.strip() for page in pages for line in set(page.splitlines()) if line.strip()
        )
        running = {line for line, n in counts.items() if n >= max(3, len(pages) // 2)}
        pages = [
            "\n".join(ln for ln in page.splitlines() if ln.strip() not in running) for page in pages
        ]
    text = "\n".join(pages)
    return _PAGE_HEADER.sub("", text)


def split_sections(text: str, *, pdf: bool) -> list[tuple[str | None, str, int]]:
    """Split text on headings into (heading, body, offset) parts. Text before the first
    heading gets heading None."""
    pattern = _PDF_HEADING if pdf else _UPPER_HEADING
    marks = [(m.start(), m.end(), m["heading"].strip()) for m in pattern.finditer(text)]
    parts: list[tuple[str | None, str, int]] = []
    cursor, heading = 0, None
    for start, end, title in marks:
        body = text[cursor:start]
        if body.strip():
            parts.append((heading, body, cursor))
        cursor, heading = end, title
    if text[cursor:].strip():
        parts.append((heading, text[cursor:], cursor))
    return parts


def _title(heading: str) -> str:
    return heading.title() if heading.isupper() else heading


# ---------------------------------------------------------------------------
# Word windows
# ---------------------------------------------------------------------------

_WORD = re.compile(r"\S+")


def word_windows(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Character spans of overlapping windows of `size` words."""
    spans = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    if not spans:
        return []
    step = max(1, size - overlap)
    windows = []
    for i in range(0, len(spans), step):
        window = spans[i : i + size]
        windows.append((window[0][0], window[-1][1]))
        if i + size >= len(spans):
            break
    return windows


def pack_paragraphs(text: str, max_words: int) -> list[tuple[int, int]]:
    """Greedy-pack paragraphs into spans of at most `max_words`; split oversized ones."""
    paragraphs = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text)]
    spans: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    current_words = 0
    for start, end in paragraphs:
        words = len(_WORD.findall(text[start:end]))
        if words > max_words:
            if current:
                spans.append(current)
                current, current_words = None, 0
            spans += [
                (start + s, start + e) for s, e in word_windows(text[start:end], max_words, 0)
            ]
            continue
        if current and current_words + words > max_words:
            spans.append(current)
            current, current_words = None, 0
        current = (current[0], end) if current else (start, end)
        current_words += words
    if current:
        spans.append(current)
    return spans


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def ordered(documents: Iterable[SourceDocument]) -> list[SourceDocument]:
    rank = {s: i for i, s in enumerate(SECTION_ORDER)}
    return sorted(documents, key=lambda d: (rank.get(d.section, 99), d.document_id))


def clean_body(document: SourceDocument) -> str:
    if document.section in ("report_pdf", "docket_pdf"):
        return strip_pdf_noise(document.body)
    return document.body


class FixedSizeChunker:
    strategy: Strategy = "fixed"

    def __init__(self, size: int = 200, overlap: int = 40) -> None:
        self.size, self.overlap = size, overlap

    def chunk_event(self, documents: Sequence[SourceDocument]) -> list[Chunk]:
        docs = ordered(documents)
        if not docs:
            return []
        # Concatenate, remembering where each document starts so a chunk can be attributed
        # to the section it starts in.
        bodies, starts, offset = [], [], 0
        for doc in docs:
            body = clean_body(doc)
            bodies.append(body)
            starts.append(offset)
            offset += len(body) + 2
        text = "\n\n".join(bodies)
        chunks = []
        for ordinal, (start, end) in enumerate(word_windows(text, self.size, self.overlap)):
            index = max(i for i, s in enumerate(starts) if s <= start)
            doc = docs[index]
            chunks.append(
                Chunk(doc.ev_id, doc.document_id, doc.section, None, ordinal, text[start:end],
                      start - starts[index], end - starts[index])
            )  # fmt: skip
        return chunks


class SectionAwareChunker:
    strategy: Strategy = "section"

    def __init__(self, max_words: int = 220) -> None:
        self.max_words = max_words

    def chunk_event(self, documents: Sequence[SourceDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in ordered(documents):
            body = clean_body(doc)
            is_pdf = doc.section in ("report_pdf", "docket_pdf")
            parts = [(None, body, 0)] if doc.section == "probable_cause" else split_sections(
                body, pdf=is_pdf
            )  # fmt: skip
            for heading, part, offset in parts:
                title = SECTION_TITLES.get(doc.section, doc.section)
                label = f"{title} - {_title(heading)}" if heading else title
                for start, end in pack_paragraphs(part, self.max_words):
                    segment = part[start:end]
                    lead = len(segment) - len(segment.lstrip())
                    segment = segment.strip()
                    begin = offset + start + lead
                    chunks.append(
                        Chunk(doc.ev_id, doc.document_id, doc.section, heading, len(chunks),
                              f"{label}: {segment}", begin, begin + len(segment))
                    )  # fmt: skip
        return chunks


class ParentChildChunker:
    strategy: Strategy = "parent_child"

    def __init__(self, parent_words: int = 600, child_words: int = 110, overlap: int = 20) -> None:
        self.parents = SectionAwareChunker(max_words=parent_words)
        self.child_words, self.overlap = child_words, overlap

    def chunk_event(self, documents: Sequence[SourceDocument]) -> list[Chunk]:
        parents = self.parents.chunk_event(documents)
        for parent in parents:
            label, _, body = parent.text.partition(": ")
            body_offset = len(label) + 2
            for ordinal, (start, end) in enumerate(
                word_windows(body, self.child_words, self.overlap)
            ):
                parent.children.append(
                    Chunk(parent.ev_id, parent.document_id, parent.section, parent.heading,
                          ordinal, f"{label}: {body[start:end]}",
                          parent.char_start + body_offset + start,
                          parent.char_start + body_offset + end)
                )  # fmt: skip
        return parents


def make_chunker(strategy: Strategy) -> Chunker:
    chunkers: dict[Strategy, Chunker] = {
        "fixed": FixedSizeChunker(),
        "section": SectionAwareChunker(),
        "parent_child": ParentChildChunker(),
    }
    return chunkers[strategy]
