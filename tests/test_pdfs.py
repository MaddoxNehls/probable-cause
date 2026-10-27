from collections.abc import Sequence
from pathlib import Path

from pypdf import PdfWriter

from probable_cause.ingest import pdfs

DOCKET_HTML = """
<a href="/Docket/Document/docBLOB?ID=1&amp;FileExtension=.PDF&amp;FileName=Release%20of%20Aircraft%20Wreckage,%20NTSB%20Form%206120.15-Master.PDF">View</a>
<a href="/Docket/Document/docBLOB?ID=2&amp;FileExtension=.PDF&amp;FileName=Figures%20from%20Factual%20Report-Rel.pdf">View</a>
<a href="/Docket/Document/docBLOB?ID=3&amp;FileExtension=.PDF&amp;FileName=Operational%20Factors%20-%20Factual%20Report-Rel.pdf">View</a>
<a href="/Docket/Document/docBLOB?ID=4&amp;FileExtension=.PDF&amp;FileName=Pilot%2FOperator%20Aircraft%20Accident%20Report,%20NTSB%20Form%206120.1-Redacted.PDF">View</a>
"""  # noqa: E501


class FakeOcr:
    name = "fake"

    def __init__(self) -> None:
        self.requested: list[int] = []

    def ocr_pages(self, pdf: Path, pages: Sequence[int]) -> dict[int, str]:
        self.requested = list(pages)
        return {p: f"Recovered text for page {p} of the pilot report." for p in pages}


def blank_pdf(path: Path, pages: int) -> Path:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as fh:
        writer.write(fh)
    return path


def test_garbled_ratio_separates_clean_prose_from_bad_ocr() -> None:
    clean = "The pilot's attained/maintained airspeed was low. Rating(s): McDonald iPad 55,Male."
    damaged = "Tne Nauonal TransPOrtation Safety Boara nas comptetea 1ts onvestogatiOn T1m"
    assert pdfs.garbled_ratio(clean) == 0.0
    assert pdfs.garbled_ratio(damaged) > pdfs.GARBLED_PAGE_RATIO


def test_parse_docket_documents_decodes_titles() -> None:
    docs = pdfs.parse_docket_documents(DOCKET_HTML)
    assert len(docs) == 4
    title, url = docs[3]
    assert title == "Pilot/Operator Aircraft Accident Report, NTSB Form 6120.1-Redacted.PDF"
    assert url.startswith("https://data.ntsb.gov/Docket/Document/docBLOB?ID=4&")


def test_choose_docket_document_prefers_pilot_report_not_wreckage_release() -> None:
    docs = pdfs.parse_docket_documents(DOCKET_HTML)
    chosen = pdfs.choose_docket_document(docs)
    assert chosen is not None
    assert "6120.1-" in chosen[0]
    chosen = pdfs.choose_docket_document(docs[:3])
    assert chosen is not None
    assert chosen[0].startswith("Operational Factors")
    assert pdfs.choose_docket_document(docs[:1]) is None


def test_extract_pdf_ocrs_pages_without_text_layer(tmp_path: Path) -> None:
    path = blank_pdf(tmp_path / "scan.pdf", pages=2)
    ocr = FakeOcr()
    extraction = pdfs.extract_pdf(path, ocr)
    assert ocr.requested == [1, 2]
    assert extraction.method == "ocr"
    assert extraction.ocr_pages == [1, 2]
    assert extraction.text.count("\f") == 1


def test_extract_pdf_without_ocr_flags_empty_pages(tmp_path: Path) -> None:
    extraction = pdfs.extract_pdf(blank_pdf(tmp_path / "scan.pdf", pages=2), pdfs.NoOcr())
    assert extraction.method == "pdf_text"
    assert extraction.empty_pages == [1, 2]
    assert extraction.text == ""
