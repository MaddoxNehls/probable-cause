# Data source

## Decision

**Structured data and narratives:** the NTSB's monthly full snapshot `avall.zip` from
<https://data.ntsb.gov/avdata>, read with **mdbtools inside a network-less Docker container**.

**Report PDFs (sample only):** CAROL final-report PDFs for post-2020 events, and documents from the
public docket system (`data.ntsb.gov/Docket`) for older events.

| | |
|---|---|
| File | `avall.zip` (one member, `avall.mdb`, Microsoft Access / Jet 4) |
| Coverage | investigations with event dates from 2008-01-01 to the present |
| Snapshot used | published 2026-10-01 07:36:33, 96,583,215 bytes zipped, 561 MB unzipped |
| SHA-256 | `bc74fefb1b39e1b8...` (full value in `data/raw/downloads/manifest.json`) |
| Refresh cadence | full snapshot on the 1st of each month; incremental `upDDMON.zip` files on the 1st, 8th, 15th and 22nd |
| Licence | US federal government work, public domain |

## What avall.mdb contains

21 tables. The loader uses these:

| mdb table | rows | becomes | notes |
|---|---:|---|---|
| `events` | 31,303 | `events` | one row per investigation (`ev_id`) |
| `aircraft` | 31,854 | `aircraft` | one row per aircraft; midairs have two or more |
| `injury` | 181,575 | `injuries` | person category x injury level x count |
| `Events_Sequence` | 66,683 | `occurrences` | phase of flight + occurrence, with a "defining event" flag |
| `Findings` | 73,002 | `findings` | hierarchical finding codes, cause/factor, "in probable cause" flag |
| `narratives` | 28,516 | `narratives`, `documents` | factual, analysis, probable cause text |
| `eADMSPUB_DataDictionary` | 4,574 | (code book) | `(table, column, code) -> meaning` |

Not loaded in milestone 1: `engines`, `Flight_Crew`, `flight_time`, the `dt_*` multi-value tables
(weather, airspace, certificates), and the legacy `Occurrences` / `seq_of_events` tables, which
are empty for post-2008 events.

## Alternatives considered

| Option | Verdict |
|---|---|
| **avall.zip (chosen)** | Official, one file, full refresh monthly, has every table including narratives. Reproducible: one URL plus a checksum. |
| CAROL query tool / API | Moved to `my.ntsb.gov` in 2026. Good for looking up single cases. Bulk API access needs a developer.ntsb.gov subscription key, and results are paginated, so it is not a reproducible bulk source. |
| `Pre2008.zip` | Investigations from 1982 to 2007 use the legacy coding scheme (`Occurrences`, `seq_of_events`) instead of findings, so it would need a second normalizer. Out of scope for the 2008+ corpus. |
| Weekly `up*.zip` incrementals | Could cut refresh bandwidth, but merging partial snapshots adds failure modes for a 97 MB saving. A full monthly reload takes about a minute. |
| Kaggle / GitHub CSV mirrors | Unofficial, stale, and they drop the narrative columns or the code tables. |
| FAA ASIAS mirror | Same NTSB data re-hosted with a different schema and slower updates. |

## Reading `.mdb` without Microsoft Access

| Reader | Result |
|---|---|
| **mdbtools 1.0.0 in Docker (chosen)** | Exact text, including Unicode punctuation. Exports all 7 tables in about 90 s. Runs anywhere Docker runs (Windows, Linux CI). |
| `access-parser` 0.0.6 (pure Python) | Opens the file and reads every row, but **silently corrupts 4,978 of 28,516 narratives (17.5%)**. Jet 4 "compressed Unicode" is decoded byte-wise, so `’` (U+2019) becomes `"\x19 "` and `“ ”` become `"\x1c "` / `"\x1d "`. Nothing raises an error; the text is just wrong. |
| `pyodbc` + Microsoft Access Database Engine | Windows only, needs a 64-bit ACE redistributable installed with admin rights, and cannot run in Linux CI. |

The comparison was run on the full narratives table, matching rows by `(ev_id, Aircraft_Key)`.
The quiet corruption is why mdbtools was chosen over the more convenient pure-Python option.

mdbtools runs in `probable-cause-tools:1` (`docker/tools/Dockerfile`: Debian bookworm, mdbtools,
poppler-utils, tesseract). Downloaded files are untrusted, so the container gets `--network none`,
the `.mdb` directory is mounted read-only, and the container runs as an unprivileged user.
`LC_ALL=C.UTF-8` is set so mdbtools converts text to UTF-8.

## Caveats found while building the loader

1. **No ETag, no useful Last-Modified.** The avdata server sends neither (Last-Modified is the
   request time). The cache key is the `(published date, size)` pair from the directory listing,
   and every local file is checked against the SHA-256 recorded in `manifest.json`.
2. **The narrative column descriptions in the dictionary are misleading.** By reading final
   reports side by side: `narr_accp` is the **factual** narrative, `narr_accf` is the
   **analysis** (described as "Narrative summary released at completion"), and `narr_cause` is
   the **probable cause**. `narr_inc` is always empty.
3. **Since 2020 the database has mostly lost the factual narrative.** Factual coverage falls from
   about 98% (2008-2019) to about 30-35% (2021+). For CAROL-era reports, the factual sections
   (history of flight, pilot, aircraft, weather, wreckage) are in the PDF only. This is the
   strongest reason the PDF path matters.
4. **About 2,600 short-form reports (mostly 2008-2016) copy the factual text into the analysis
   column.** The loader keeps the narrative row as-is but skips the duplicate `analysis` document,
   so retrieval doesn't index the same text twice.
5. **Foreign-assist events have no text.** `ntsb_no` values ending in `WA<n>` are foreign
   investigations that the NTSB only assisted with: 1,741 events from 2021 to 2024 and none of
   them have a narrative. Without them, probable-cause coverage is 94.9% instead of 78.6%.
6. **The code book has gaps.** Values in the data that are missing from the dictionary:
   aircraft categories `WSFT`/`PPAR`/`RCKT`, FAR parts `107`/`437`, operation types
   `PUBF`/`PUBL`/`PUBS`, phases `553`/`601`, and occurrence events `281`-`284`/`850`. The
   injury table also uses truncated categories (`Flig`, `Pass`, `Cabi`, `LapC`) that appear
   nowhere in the dictionary. These live in `SUPPLEMENTAL_CODES` in
   `src/probable_cause/ingest/codes.py`, each checked against the `Occurrence_Description` text.
7. **Schema drift in findings.** `Cause_Factor` (C/F) is empty for every finding since 2020. The
   newer `cm_inPc` flag ("in probable cause") is the only consistent signal, so both are kept.
8. **Report PDF keys are only available for new IDs.** For every event after September 2020
   (and a few back to late 2016), `ev_id` is `YYYYMMDD` + the 6-digit CAROL key
   (`20210102102480` gives key `102480`), and
   `.../GenerateNewestReport/{key}/pdf` returns the final report. 9,879 events have a derivable
   key. Older IDs (`20080107X00026`) carry no key, so their provenance link is the docket page.
   Some keys return HTTP 403 (for example `106534`, a major DCA investigation).
9. **Docket documents contain personal data.** Even documents marked "Redacted" include owner
   names, addresses and witness names. Raw PDFs and their text stay in the local database and
   `data/` (gitignored). A hosted demo should exclude docket text or redact it.
