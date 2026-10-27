# Probable Cause

Ask questions about almost two decades of NTSB aviation accident investigations. Every answer
cites its sources, and the retrieval pipeline is chosen by benchmark, not by vibes.

> **Status: milestone 1 of 5.** Data ingest, structured database, text extraction, and a lexical
> baseline search are done. Chunking strategies, embeddings, the chat UI and the benchmark are
> next. See [`docs/spec.md`](docs/spec.md).

## The data story

The NTSB publishes every US civil aviation accident investigation as a 561 MB Microsoft Access
file that is refreshed monthly. The current snapshot has **31,303 investigations from
2008-01-01 to 2026-09-30**, with 24,998 probable-cause statements, 24,400 factual narratives and
25,015 analyses. It looks like a clean RAG corpus. It isn't:

- **Reading it at all is the first trap.** The convenient pure-Python Access reader silently
  corrupts 17.5% of narratives: curly quotes come out as control characters, and nothing raises
  an error. The loader uses mdbtools in a sandboxed container instead
  ([why](docs/data-source.md#reading-mdb-without-microsoft-access)).
- **The schema changed in 2020 and the data didn't say so.** When the NTSB moved to its CAROL
  system, the factual narrative mostly left the database: coverage fell from 98% to about 30%.
  It now exists only in report PDFs. The cause/factor flag on findings stopped being filled at
  the same time.
- **The documentation is wrong in places.** The data dictionary describes the *analysis* column
  as a "narrative summary", omits codes that the data actually uses, and doesn't list the injury
  categories the injury table actually contains (`Flig`, `Pass`, `LapC`).
- **"Missing" usually has a reason.** 21% of events have no probable cause, but nearly all of
  those are foreign investigations the NTSB only assisted with, or cases still open. For
  NTSB-led investigations, coverage is 94.9%.
- **The PDFs range from pristine to hostile.** CAROL final reports are born-digital and clean.
  Docket pilot reports are scanned, handwritten forms, sometimes carrying a text layer from
  NTSB's own OCR that is worse than none (`Tne Nauonal TransPOrtation Safety Boara`).
  See [`docs/pdf-extraction.md`](docs/pdf-extraction.md).

Full numbers: [`docs/data-quality.md`](docs/data-quality.md). Source decision and caveats:
[`docs/data-source.md`](docs/data-source.md).

## Current numbers (avall.mdb published 2026-10-01)

| | |
|---|---:|
| Events (investigations) | 31,303 |
| Aircraft | 31,854 |
| Findings / occurrences | 73,002 / 66,683 |
| Narrative rows | 28,516 |
| Text documents (factual, analysis, probable cause, plus 25 PDFs) | 67,692 |
| Years covered | 2008-2026 |
| Events with a probable cause (all / NTSB-led) | 78.6% / 94.9% |
| Postgres database size | 419 MB |

## Reproduce

Requirements: Python 3.13, [uv](https://docs.astral.sh/uv/), Docker. No Microsoft Access, no admin
rights.

```bash
uv sync
docker compose up -d --wait        # Postgres 17 + pgvector on localhost:5436 (PC_DB_PORT)
uv run pc ingest download          # ~97 MB; cached by listing date + size, verified by SHA-256
uv run pc ingest load              # unzip, mdbtools export, normalize, load (about 2.5 min the first time)
uv run pc ingest pdfs              # optional: 25-document PDF sample with OCR fallback (about 2 min)
uv run pc ingest stats --write docs/data-quality.md
uv run pc search "carburetor icing" --year-from 2010
```

Every step is idempotent. `download` re-fetches only when the listing changes or a checksum
fails. `load` is skipped when the same snapshot (by SHA-256) is already loaded; `--force`
reloads it in one transaction and keeps PDF-derived documents. Raw files live in `data/`, which
is gitignored.

```text
$ uv run pc search '"spatial disorientation" night' --limit 2
 1. 20210423102968  ERA21LA191  2021-04-23  Fatal    [probable_cause] score=0.182
    weather planning, which resulted in an inadvertent encounter with instrument meteorological
    conditions at **night**, **spatial** **disorientation**, and collision with terrain
    https://data.ntsb.gov/carol-repgen/api/Aviation/ReportMain/GenerateNewestReport/102968/pdf
 2. 20100815X75058  ERA10LA421  2010-08-14  Serious  [probable_cause] score=0.095
    instrument-rated pilot's **spatial** **disorientation** during **night**, over-water,
    low-ambient-light conditions, which resulted in the airplane impacting the ocean
    https://data.ntsb.gov/Docket/?NTSBNumber=ERA10LA421
```

The baseline also shows why later milestones exist. `pc search "icing fatal part 91"` treats
"fatal" and "part 91" as words, so it ranks a crash on *sea ice* near Nome first. Those terms
belong in structured filters (`events.highest_injury`, `aircraft.far_part`), which is exactly
what the LLM query-rewriting experiment in milestone 4 is meant to measure.

## Architecture

```text
data.ntsb.gov/avdata ──download──> avall.zip ──unzip──> avall.mdb
                                                          │  mdbtools (Docker, --network none)
                                                          ▼
                                                   CSV per table
                                                          │  normalize + decode codes
                                                          ▼
Postgres 17 + pgvector:  events ─┬─ aircraft ─┬─ injuries / occurrences / findings
                                 ├─ narratives
                                 └─ documents (section, source URL, tsvector) ── chunks (vector, M2)
CAROL / docket PDFs ──pypdf──> garbled or empty page? ──> Tesseract (Docker) ──> documents
```

| Module | Responsibility |
|---|---|
| `ingest/download.py` | avdata listing parse, checksummed download, safe single-member unzip |
| `ingest/mdb.py` | sandboxed mdbtools export |
| `ingest/codes.py` | data-dictionary code book plus hand-verified supplemental codes |
| `ingest/normalize.py` | typed records: dates, HHMM times, decoded codes, provenance URLs |
| `ingest/load.py` | `COPY` into staging, `MERGE` events, replace child tables, build `documents` |
| `ingest/pdfs.py` | sample selection, polite fetch, text-layer extraction, OCR fallback, quality signals |
| `ingest/stats.py` | the data-quality report |
| `search.py` | ranked Postgres full-text search, one hit per event |
| `migrations/*.sql` | plain SQL, applied in order and recorded in `schema_migrations` |

Design choices worth knowing:

- **Full snapshot, transactional replace.** avall is a full snapshot, so the loader replaces the
  structured tables inside one transaction. Events are `MERGE`d (Postgres 17's
  `WHEN NOT MATCHED BY SOURCE THEN DELETE`) rather than truncated, so PDF documents attached to
  surviving events are kept.
- **Codes are decoded at load time.** Columns hold `Fatal`, `Part 91: General Aviation`,
  `Approach-VFR Pattern Final`, not `FATL`/`091`/`508`. FAR part and occurrence codes are also
  kept raw because they are natural filter keys.
- **`documents` is the unit of provenance.** Each row is one section of one report, with a
  source URL (the CAROL report PDF when the key is derivable, otherwise the docket page) and a
  generated `tsvector`. `chunks.embedding` is a dimensionless `vector` column until the
  embedding model is chosen in milestone 2.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy                         # strict, over src/, tests/, scripts/
uv run pytest                       # unit tests + Postgres testcontainer tests (needs Docker)
```

Test fixtures in `tests/fixtures/avall/` are five real events cut from the export by
`scripts/make_fixtures.py`. Owner, operator and registration fields are blanked or replaced, and
narratives are truncated. The events were picked to cover a two-aircraft midair, a post-2020 ID,
an undocumented code, a duplicated narrative, and a foreign-assist event.
