# Data quality report

Generated 2026-10-27 22:01 UTC by `pc ingest stats --write docs/data-quality.md` from avall.mdb published 2026-10-01 07:36:33 (sha256 `bc74fefb1b39e1b8...`).

## Headline

| metric | value |
|---|---|
| events | 31,303 |
| aircraft | 31,854 |
| narrative rows | 28,516 |
| findings | 73,002 |
| occurrences (sequence of events) | 66,683 |
| text documents | 67,692 |
| date range | 2008-01-01 to 2026-09-30 |
| database size | 419 MB |
| events with any narrative | 89.7% |
| events with a probable cause | 78.6% |
| NTSB-led events with a probable cause | 94.9% |
| narratives whose analysis duplicates the factual text | 2,636 |

## Coverage by year

Percentages are of all events in the year, except the last column, which excludes foreign investigations the NTSB only assisted with (`ntsb_no` ending in `WA<n>`). Recent years are low because investigations stay open 12-24 months.

| year | events | fatal | foreign-assist | any narrative | factual | analysis | probable cause | probable cause (NTSB-led) |
|---|---|---|---|---|---|---|---|---|
| 2008 | 1,892 | 382 | 151 | 98.8% | 98.8% | 92.0% | 91.9% | 98.6% |
| 2009 | 1,783 | 359 | 151 | 99.3% | 99.3% | 92.9% | 92.7% | 99.1% |
| 2010 | 1,786 | 363 | 192 | 98.9% | 98.9% | 88.7% | 88.2% | 98.0% |
| 2011 | 1,848 | 398 | 216 | 99.0% | 98.9% | 88.8% | 88.6% | 99.0% |
| 2012 | 1,834 | 389 | 237 | 99.1% | 99.1% | 87.1% | 87.0% | 98.9% |
| 2013 | 1,561 | 344 | 231 | 98.0% | 98.0% | 87.3% | 87.2% | 99.2% |
| 2014 | 1,534 | 360 | 195 | 97.5% | 97.5% | 90.9% | 90.9% | 99.0% |
| 2015 | 1,581 | 365 | 246 | 98.0% | 97.9% | 83.3% | 83.3% | 98.7% |
| 2016 | 1,662 | 341 | 294 | 98.7% | 98.7% | 80.9% | 80.9% | 98.2% |
| 2017 | 1,635 | 338 | 281 | 99.0% | 98.7% | 81.6% | 81.6% | 98.5% |
| 2018 | 1,686 | 360 | 314 | 97.9% | 96.8% | 80.3% | 80.3% | 98.7% |
| 2019 | 1,624 | 380 | 308 | 97.5% | 95.4% | 80.5% | 80.6% | 99.5% |
| 2020 | 1,395 | 291 | 251 | 91.0% | 63.9% | 81.7% | 81.8% | 99.7% |
| 2021 | 1,643 | 321 | 428 | 74.0% | 34.5% | 74.0% | 74.0% | 100.0% |
| 2022 | 1,687 | 329 | 414 | 75.4% | 35.6% | 75.3% | 75.4% | 99.9% |
| 2023 | 1,653 | 295 | 434 | 73.7% | 34.8% | 73.5% | 73.5% | 99.7% |
| 2024 | 1,669 | 292 | 465 | 72.0% | 31.6% | 67.6% | 67.6% | 93.8% |
| 2025 | 1,632 | 275 | 451 | 72.1% | 29.2% | 46.6% | 46.6% | 64.4% |
| 2026 | 1,198 | 217 | 329 | 49.6% | 27.1% | 22.5% | 22.5% | 31.1% |

## Narrative length (characters)

| section | documents | p5 | p25 | median | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| analysis | 18,269 | 437 | 724 | 1,069 | 1,616 | 3,110 | 36,548 | 1,347 |
| factual | 24,400 | 410 | 793 | 1,740 | 4,659 | 15,118 | 78,667 | 4,020 |
| probable_cause | 24,998 | 57 | 97 | 136 | 192 | 327 | 2,445 | 157 |

Probable-cause statements under 60 characters: 1,354 of 24,998. Statements with an undetermined cause: 2,796.

## Missing or undecodable fields

| field | missing | rows | share |
|---|---|---|---|
| aircraft FAR part | 699 | 31,854 | 2.2% |
| aircraft operation type | 6,427 | 31,854 | 20.2% |
| event coordinates | 3,436 | 31,303 | 11.0% |
| event weather condition | 5,351 | 31,303 | 17.1% |
| finding cause/factor flag | 29,482 | 73,002 | 40.4% |
| occurrence phase (undecodable) | 0 | 66,683 | 0.0% |
| event highest injury | 1,449 | 31,303 | 4.6% |

## PDF sample

| source | method | documents | mean chars | pages | pages w/o text layer | pages OCRed |
|---|---|---|---|---|---|---|
| carol_report | pdf_text | 15 | 10,711 | 103 | 0 | 0 |
| docket | mixed | 2 | 21,652 | 17 | 2 | 8 |
| docket | ocr | 1 | 20,133 | 9 | 9 | 9 |
| docket | pdf_text | 7 | 18,095 | 74 | 0 | 0 |
