"""Baseline lexical retrieval: Postgres full-text search, ranked per event.

This is the BM25-ish floor that later retrieval strategies are benchmarked against.
"""

from dataclasses import dataclass
from datetime import date

import psycopg
from psycopg.rows import class_row

type Conn = psycopg.Connection[tuple[object, ...]]

DEFAULT_SECTIONS = ("probable_cause", "analysis", "factual")
# Matching the probable-cause statement is stronger evidence than a passing mention in a
# long factual narrative, which ts_rank_cd's length normalization only partly corrects.
SECTION_WEIGHTS = {"probable_cause": 2.0, "analysis": 1.25}

_SEARCH = """
WITH q AS (
    SELECT CASE WHEN %(match_any)s
                THEN replace(websearch_to_tsquery('english', %(query)s)::text, '&', '|')::tsquery
                ELSE websearch_to_tsquery('english', %(query)s) END AS query
),
hits AS (
    SELECT d.ev_id, d.section, d.body,
           ts_rank_cd(d.tsv, q.query, 32)
             * CASE d.section WHEN 'probable_cause' THEN %(w_cause)s
                              WHEN 'analysis' THEN %(w_analysis)s ELSE 1.0 END AS score
    FROM documents d
    JOIN events e USING (ev_id)
    CROSS JOIN q
    WHERE d.tsv @@ q.query
      AND d.section = ANY(%(sections)s)
      AND (%(date_from)s::date IS NULL OR e.ev_date >= %(date_from)s::date)
      AND (%(date_to)s::date IS NULL OR e.ev_date <= %(date_to)s::date)
),
best AS (
    SELECT DISTINCT ON (ev_id) * FROM hits ORDER BY ev_id, score DESC
),
top AS (
    SELECT * FROM best ORDER BY score DESC, ev_id LIMIT %(limit)s
)
SELECT t.ev_id, e.ntsb_no, e.ev_date, e.highest_injury, t.section, t.score::float8 AS score,
       regexp_replace(ts_headline('english', t.body, q.query,
                   'MaxFragments=2, MinWords=8, MaxWords=30, FragmentDelimiter=" ... ", '
                   'StartSel=**, StopSel=**'), '[[:space:]]+', ' ', 'g') AS snippet,
       coalesce(e.report_url, e.docket_url) AS source_url
FROM top t JOIN events e USING (ev_id) CROSS JOIN q
ORDER BY t.score DESC, t.ev_id
"""


@dataclass(frozen=True)
class Hit:
    ev_id: str
    ntsb_no: str | None
    ev_date: date
    highest_injury: str | None
    section: str
    score: float
    snippet: str
    source_url: str


def search(
    conn: Conn,
    query: str,
    *,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int = 10,
    sections: tuple[str, ...] = DEFAULT_SECTIONS,
    match_any: bool = False,
) -> list[Hit]:
    params = {
        "query": query,
        "match_any": match_any,
        "sections": list(sections),
        "date_from": date(year_from, 1, 1) if year_from else None,
        "date_to": date(year_to, 12, 31) if year_to else None,
        "limit": limit,
        "w_cause": SECTION_WEIGHTS["probable_cause"],
        "w_analysis": SECTION_WEIGHTS["analysis"],
    }
    with conn.cursor(row_factory=class_row(Hit)) as cur:
        return cur.execute(_SEARCH, params).fetchall()
