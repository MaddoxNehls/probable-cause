"""Data-quality report over the loaded database, rendered as Markdown."""

from dataclasses import dataclass
from datetime import UTC, date, datetime

import psycopg
from psycopg.rows import class_row

type Conn = psycopg.Connection[tuple[object, ...]]

# NTSB numbers ending in WA<digits> are foreign investigations the NTSB only assisted
# with; the US database carries no narrative or probable cause for them.
FOREIGN_ASSIST = r"e.ntsb_no ~ 'WA[0-9]+$'"


@dataclass(frozen=True)
class Headline:
    events: int
    aircraft: int
    narratives: int
    findings: int
    occurrences: int
    documents: int
    first_date: date
    last_date: date
    db_size: str
    sha256: str | None
    published: datetime | None
    duplicated_analysis: int


@dataclass(frozen=True)
class YearRow:
    year: int
    events: int
    fatal: int
    foreign_assist: int
    with_narrative: int
    with_factual: int
    with_analysis: int
    with_cause: int
    ntsb_led: int
    ntsb_led_with_cause: int


@dataclass(frozen=True)
class LengthRow:
    section: str
    documents: int
    p5: int
    p25: int
    p50: int
    p75: int
    p95: int
    max: int
    mean: int


@dataclass(frozen=True)
class GapRow:
    field: str
    missing: int
    total: int


@dataclass(frozen=True)
class CauseShape:
    short: int
    undetermined: int
    total: int


@dataclass(frozen=True)
class PdfRow:
    source: str
    method: str
    documents: int
    mean_chars: int
    pages: int
    empty_pages: int
    ocr_pages: int


_HEADLINE = """
SELECT (SELECT count(*) FROM events) AS events,
       (SELECT count(*) FROM aircraft) AS aircraft,
       (SELECT count(*) FROM narratives) AS narratives,
       (SELECT count(*) FROM findings) AS findings,
       (SELECT count(*) FROM occurrences) AS occurrences,
       (SELECT count(*) FROM documents) AS documents,
       (SELECT min(ev_date) FROM events) AS first_date,
       (SELECT max(ev_date) FROM events) AS last_date,
       pg_size_pretty(pg_database_size(current_database())) AS db_size,
       s.sha256, s.listed_at AS published,
       (SELECT count(*) FROM narratives WHERE analysis = factual) AS duplicated_analysis
FROM (SELECT 1) one
LEFT JOIN LATERAL (SELECT sha256, listed_at FROM source_snapshots ORDER BY id DESC LIMIT 1) s
  ON true
"""

_BY_YEAR = f"""
SELECT extract(year FROM e.ev_date)::int AS year,
       count(*) AS events,
       count(*) FILTER (WHERE e.highest_injury = 'Fatal') AS fatal,
       count(*) FILTER (WHERE {FOREIGN_ASSIST}) AS foreign_assist,
       count(*) FILTER (WHERE n.has_any) AS with_narrative,
       count(*) FILTER (WHERE n.has_factual) AS with_factual,
       count(*) FILTER (WHERE n.has_analysis) AS with_analysis,
       count(*) FILTER (WHERE n.has_cause) AS with_cause,
       count(*) FILTER (WHERE NOT {FOREIGN_ASSIST}) AS ntsb_led,
       count(*) FILTER (WHERE NOT {FOREIGN_ASSIST} AND n.has_cause) AS ntsb_led_with_cause
FROM events e
LEFT JOIN (
    SELECT ev_id,
           bool_or(coalesce(factual, analysis, probable_cause) IS NOT NULL) AS has_any,
           bool_or(factual IS NOT NULL) AS has_factual,
           bool_or(analysis IS NOT NULL) AS has_analysis,
           bool_or(probable_cause IS NOT NULL) AS has_cause
    FROM narratives GROUP BY ev_id
) n USING (ev_id)
GROUP BY 1 ORDER BY 1
"""

_LENGTHS = """
SELECT section, count(*) AS documents,
       percentile_disc(0.05) WITHIN GROUP (ORDER BY length(body)) AS p5,
       percentile_disc(0.25) WITHIN GROUP (ORDER BY length(body)) AS p25,
       percentile_disc(0.50) WITHIN GROUP (ORDER BY length(body)) AS p50,
       percentile_disc(0.75) WITHIN GROUP (ORDER BY length(body)) AS p75,
       percentile_disc(0.95) WITHIN GROUP (ORDER BY length(body)) AS p95,
       max(length(body)) AS max, round(avg(length(body)))::int AS mean
FROM documents WHERE source = 'avall'
GROUP BY section ORDER BY section
"""

_CAUSE_SHAPE = """
SELECT count(*) FILTER (WHERE length(body) < 60) AS short,
       count(*) FILTER (WHERE body ~* 'undetermined|could not be determined') AS undetermined,
       count(*) AS total
FROM documents WHERE section = 'probable_cause' AND source = 'avall'
"""

_GAPS = """
SELECT 'aircraft FAR part' AS field, count(*) FILTER (WHERE far_part IS NULL) AS missing,
       count(*) AS total FROM aircraft
UNION ALL
SELECT 'aircraft operation type', count(*) FILTER (WHERE operation_type IS NULL), count(*)
FROM aircraft
UNION ALL
SELECT 'event highest injury', count(*) FILTER (WHERE highest_injury IS NULL), count(*)
FROM events
UNION ALL
SELECT 'event weather condition', count(*) FILTER (WHERE weather_condition IS NULL), count(*)
FROM events
UNION ALL
SELECT 'event coordinates', count(*) FILTER (WHERE latitude IS NULL), count(*) FROM events
UNION ALL
SELECT 'occurrence phase (undecodable)', count(*) FILTER (WHERE phase ~ '^[0-9]+$'), count(*)
FROM occurrences
UNION ALL
SELECT 'finding cause/factor flag', count(*) FILTER (WHERE cause_factor IS NULL), count(*)
FROM findings
"""

_PDF = """
SELECT source, extraction_method AS method, count(*) AS documents,
       round(avg(length(body)))::int AS mean_chars,
       sum((metadata->>'pages')::int)::int AS pages,
       sum((metadata->>'empty_pages')::int)::int AS empty_pages,
       sum(jsonb_array_length(metadata->'ocr_pages'))::int AS ocr_pages
FROM documents WHERE source <> 'avall'
GROUP BY 1, 2 ORDER BY 1, 2
"""


@dataclass(frozen=True)
class Stats:
    headline: Headline
    by_year: list[YearRow]
    lengths: list[LengthRow]
    cause_shape: CauseShape
    gaps: list[GapRow]
    pdf: list[PdfRow]


def _rows[T](conn: Conn, query: str, row_type: type[T]) -> list[T]:
    with conn.cursor(row_factory=class_row(row_type)) as cur:
        return cur.execute(query).fetchall()


def collect(conn: Conn) -> Stats:
    return Stats(
        headline=_rows(conn, _HEADLINE, Headline)[0],
        by_year=_rows(conn, _BY_YEAR, YearRow),
        lengths=_rows(conn, _LENGTHS, LengthRow),
        cause_shape=_rows(conn, _CAUSE_SHAPE, CauseShape)[0],
        gaps=_rows(conn, _GAPS, GapRow),
        pdf=_rows(conn, _PDF, PdfRow),
    )


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(stats: Stats) -> str:
    h = stats.headline
    years = stats.by_year
    total = sum(y.events for y in years)
    ntsb_led = sum(y.ntsb_led for y in years)
    sha = (h.sha256 or "")[:16]
    out = [
        "# Data quality report",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `pc ingest stats --write "
        f"docs/data-quality.md` from avall.mdb published {h.published} (sha256 `{sha}...`).",
        "",
        "## Headline",
        "",
        _table(
            ["metric", "value"],
            [
                ["events", f"{h.events:,}"],
                ["aircraft", f"{h.aircraft:,}"],
                ["narrative rows", f"{h.narratives:,}"],
                ["findings", f"{h.findings:,}"],
                ["occurrences (sequence of events)", f"{h.occurrences:,}"],
                ["text documents", f"{h.documents:,}"],
                ["date range", f"{h.first_date} to {h.last_date}"],
                ["database size", h.db_size],
                ["events with any narrative", _pct(sum(y.with_narrative for y in years), total)],
                ["events with a probable cause", _pct(sum(y.with_cause for y in years), total)],
                [
                    "NTSB-led events with a probable cause",
                    _pct(sum(y.ntsb_led_with_cause for y in years), ntsb_led),
                ],
                [
                    "narratives whose analysis duplicates the factual text",
                    f"{h.duplicated_analysis:,}",
                ],
            ],
        ),
        "",
        "## Coverage by year",
        "",
        "Percentages are of all events in the year, except the last column, which excludes "
        "foreign investigations the NTSB only assisted with (`ntsb_no` ending in `WA<n>`). "
        "Recent years are low because investigations stay open 12-24 months.",
        "",
        _table(
            [
                "year",
                "events",
                "fatal",
                "foreign-assist",
                "any narrative",
                "factual",
                "analysis",
                "probable cause",
                "probable cause (NTSB-led)",
            ],
            [
                [
                    str(y.year),
                    f"{y.events:,}",
                    f"{y.fatal:,}",
                    f"{y.foreign_assist:,}",
                    _pct(y.with_narrative, y.events),
                    _pct(y.with_factual, y.events),
                    _pct(y.with_analysis, y.events),
                    _pct(y.with_cause, y.events),
                    _pct(y.ntsb_led_with_cause, y.ntsb_led),
                ]
                for y in years
            ],
        ),
        "",
        "## Narrative length (characters)",
        "",
        _table(
            ["section", "documents", "p5", "p25", "median", "p75", "p95", "max", "mean"],
            [
                [
                    r.section,
                    *(
                        f"{v:,}"
                        for v in (r.documents, r.p5, r.p25, r.p50, r.p75, r.p95, r.max, r.mean)
                    ),
                ]
                for r in stats.lengths
            ],
        ),
        "",
        f"Probable-cause statements under 60 characters: {stats.cause_shape.short:,} of "
        f"{stats.cause_shape.total:,}. Statements with an undetermined cause: "
        f"{stats.cause_shape.undetermined:,}.",
        "",
        "## Missing or undecodable fields",
        "",
        _table(
            ["field", "missing", "rows", "share"],
            [
                [g.field, f"{g.missing:,}", f"{g.total:,}", _pct(g.missing, g.total)]
                for g in stats.gaps
            ],
        ),
    ]
    if stats.pdf:
        out += [
            "",
            "## PDF sample",
            "",
            _table(
                [
                    "source",
                    "method",
                    "documents",
                    "mean chars",
                    "pages",
                    "pages w/o text layer",
                    "pages OCRed",
                ],
                [
                    [
                        p.source,
                        p.method,
                        str(p.documents),
                        f"{p.mean_chars:,}",
                        str(p.pages),
                        str(p.empty_pages),
                        str(p.ocr_pages),
                    ]
                    for p in stats.pdf
                ],
            ),
        ]
    return "\n".join(out) + "\n"
