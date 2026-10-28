from probable_cause.rag.chunking import (
    FixedSizeChunker,
    ParentChildChunker,
    SectionAwareChunker,
    SourceDocument,
    pack_paragraphs,
    split_sections,
    strip_pdf_noise,
    word_windows,
)

FACTUAL = (
    "HISTORY OF FLIGHT\nOn June 1, the pilot departed for a local flight. " + "word " * 300 + "\n\n"
    "PERSONNEL INFORMATIONThe pilot held a private pilot certificate.\n\n"
    "WRECKAGE AND IMPACT INFORMATION\nThe airplane came to rest inverted."
)
DOCS = [
    SourceDocument(1, "20210102102480", "factual", FACTUAL),
    SourceDocument(
        2, "20210102102480", "probable_cause", "The pilot's failure to maintain airspeed."
    ),
    SourceDocument(3, "20210102102480", "analysis", "The airplane stalled during the turn."),
]


def test_word_windows_overlap_and_cover_everything() -> None:
    text = " ".join(f"w{i}" for i in range(25))
    spans = word_windows(text, size=10, overlap=2)
    pieces = [text[s:e].split() for s, e in spans]
    assert pieces[0][0] == "w0"
    assert pieces[-1][-1] == "w24"
    assert pieces[1][0] == "w8"  # 10 - 2 overlap
    assert all(len(p) <= 10 for p in pieces)


def test_split_sections_handles_glued_uppercase_headings() -> None:
    parts = split_sections(FACTUAL, pdf=False)
    assert [h for h, _, _ in parts] == [
        "HISTORY OF FLIGHT",
        "PERSONNEL INFORMATION",
        "WRECKAGE AND IMPACT INFORMATION",
    ]
    assert parts[1][1].startswith("The pilot held")


def test_strip_pdf_noise_removes_page_headers_and_running_lines() -> None:
    pages = [f"Page {i} of 4 ERA22LA394\nRunning footer\nBody text {i}" for i in range(1, 5)]
    cleaned = strip_pdf_noise("\n\f\n".join(pages))
    assert "Page 1 of 4" not in cleaned
    assert "Running footer" not in cleaned
    assert all(f"Body text {i}" in cleaned for i in range(1, 5))


def test_pack_paragraphs_respects_limit() -> None:
    text = "\n\n".join(" ".join(["x"] * 40) for _ in range(5))
    spans = pack_paragraphs(text, max_words=90)
    assert [len(text[s:e].split()) for s, e in spans] == [80, 80, 40]


def test_fixed_chunker_concatenates_in_reading_order() -> None:
    chunks = FixedSizeChunker(size=50, overlap=10).chunk_event(DOCS)
    assert chunks[0].section == "factual"
    assert chunks[-1].text.endswith("The pilot's failure to maintain airspeed.")
    assert all(len(c.text.split()) <= 50 for c in chunks)


def test_section_chunker_never_crosses_sections_and_labels_chunks() -> None:
    chunks = SectionAwareChunker(max_words=120).chunk_event(DOCS)
    assert [c.section for c in chunks][-2:] == ["analysis", "probable_cause"]
    cause = chunks[-1]
    assert cause.text == "Probable cause: The pilot's failure to maintain airspeed."
    assert cause.citation_key == "20210102102480:probable_cause"
    personnel = [c for c in chunks if c.heading == "PERSONNEL INFORMATION"]
    assert personnel[0].text.startswith("Factual narrative - Personnel Information: The pilot")


def test_parent_child_children_belong_to_their_parent() -> None:
    parents = ParentChildChunker(parent_words=400, child_words=60, overlap=10).chunk_event(DOCS)
    big = max(parents, key=lambda p: len(p.children))
    assert len(big.children) > 1
    label = big.text.split(": ", 1)[0]
    assert all(child.text.startswith(label + ": ") for child in big.children)
    assert all(child.section == big.section for child in big.children)
