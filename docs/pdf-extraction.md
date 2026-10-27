# PDF extraction: sample results and quality notes

`pc ingest pdfs` draws a deterministic sample (`ORDER BY md5(ev_id)`), fetches each PDF once
(1 request/s, cached under `data/raw/pdf/<source>/<ev_id>/<url-hash>/`), extracts text with
pypdf, and OCRs the pages that need it. Results go into `documents` as `report_pdf` or
`docket_pdf` sections, with per-document quality signals in `metadata`.

## Pipeline

1. Extract each page's text layer with pypdf. The output keeps page boundaries as `\f`.
2. Flag a page for OCR if its text layer has **fewer than 25 characters** (a scan), or if it is
   **garbled**: more than 10% of words have shapes real words almost never have, such as
   mid-word case flips (`TransPOrtation`), digits inside words (`T1m`, `1ts`), dot runs, or
   stray symbols. Clean prose scores below 0.01.
3. OCR the flagged pages with Tesseract. An OCR result replaces an existing text layer only if it
   scores as **less garbled** than that layer.

OCR sits behind a small `OcrEngine` protocol. `DockerTesseract` renders pages with `pdftoppm`
(300 dpi, grayscale) and runs `tesseract` 5.3.0, both inside the same sandboxed tools image
used for mdbtools (`--network none`, read-only input mount). This needs no local install and no
admin rights, which matters on Windows. `--no-ocr` swaps in `NoOcr`, which leaves scanned pages
empty but still records them as `empty_pages`.

## Sample run (25 documents)

| source | documents | pages | method | mean garbled-word ratio |
|---|---:|---:|---|---:|
| CAROL final reports (2021-2026) | 15 | 103 | text layer only | 0.001 |
| Docket, typed factual reports | 6 | 65 | text layer only | 0.005 |
| Docket, pilot/operator form 6120.1 | 4 | 35 | 1 fully OCRed, 2 partly OCRed, 1 text layer | 0.040 |

To fill 10 docket slots, 21 older fatal events were tried. Two dockets were unreleased. Nine
contained only wreckage-release forms, photos or figures, with no pilot report or factual report.

## Issues seen (write-up material)

**CAROL final reports are born-digital and clean, but they're laid out as forms.**
- Every page repeats a `Page N of M <accident number>` header. These need stripping before
  chunking, or every chunk will match the accident number.
- Two-column field/value grids are flattened into single lines:
  `Location: Hilliard, Florida Accident Number: ERA22LA394`. Fine for BM25, but noisy for
  embeddings and wrong for any field parser.
- The report repeats the analysis and probable cause that are already in avall. For post-2020
  events, though, the PDF is the **only** source of the factual sections (history of flight,
  pilot, aircraft, weather), because avall no longer carries them (see `data-quality.md`). Section
  splitting on the fixed headings (`Analysis`, `Probable Cause and Findings`, `Factual
  Information`) is the obvious milestone-2 chunking strategy.
- The PDFs are AES-encrypted with an empty user password, so pypdf needs the `cryptography`
  extra. Some use CFF Type1 fonts, which make pypdf warn about `fontTools`; extraction still
  works.

**Typed docket reports (group chairman, specialist factual reports) are good.** They have line-end
hyphenation (`hyphen_breaks` up to 7 per document) and repeated running headers, but nothing
needs OCR.

**Pilot/operator forms (NTSB 6120.1) are the hard case, and they show three distinct failures:**
1. *No text layer* (SEA08LA192, 9 pages): Tesseract recovers the printed form labels well, but
   checkboxes become `(J`, `[]`, `C1`, `LJ`, and handwritten answers become noise
   (`Nearest City/Place: S35 AWIBAAF 2A`).
2. *A text layer from NTSB's own scanner OCR, which is worse than none* (ERA10FA283, and a
   CEN13FA259 form seen during development):
   `Tne Nauonal TransPOrtation Safety Boara nas comptetea 1ts onvestogatiOn`. "The PDF has text"
   doesn't mean the text is usable. The garble check caught one ERA10FA283 page, and Tesseract's
   re-OCR scored better and replaced it.
3. *Below-threshold degradation* (WPR15FA021, garbled ratio 0.052): words are split by
   letter-spacing (`Ser ia l N umbe r`, `Amate ur-Bui lt`) and checkboxes become `0`/`O`. Each
   piece passes the shape test, so the page is never re-OCRed. A dictionary-based check would
   catch this; the shape heuristic deliberately doesn't use a dictionary.

**Structure loss is the bigger problem than character errors.** Even perfect OCR of a 6120.1
yields label/value soup with no association between a checkbox and its label. For retrieval,
these forms are better treated as structured data, which avall already provides, than as text.

**Personal data.** Docket documents marked "Redacted" still contain owner names, addresses and
investigator names. They are stored locally only.
