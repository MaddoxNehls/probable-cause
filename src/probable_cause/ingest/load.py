"""Load a normalized avall snapshot into Postgres.

avall.mdb is a full snapshot, so a load replaces the structured tables wholesale inside
one transaction. Events are MERGEd rather than truncated so that PDF-derived documents
attached to surviving events are kept; events that vanished from the snapshot are
deleted, cascading to everything that hangs off them.
"""

import csv
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import astuple, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Protocol

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from probable_cause.ingest import normalize as n
from probable_cause.ingest.codes import CodeBook

type Conn = psycopg.Connection[tuple[object, ...]]


class Record(Protocol):
    """Any of the normalized dataclasses in probable_cause.ingest.normalize."""

    __dataclass_fields__: ClassVar[dict[str, Any]]


csv.field_size_limit(1 << 30)  # narratives exceed the default 128 KiB field limit


@dataclass
class LoadReport:
    loaded: dict[str, int] = field(default_factory=dict)
    duplicates_dropped: dict[str, int] = field(default_factory=dict)
    orphans_dropped: dict[str, int] = field(default_factory=dict)
    skipped: bool = False


@dataclass(frozen=True)
class Snapshot:
    file_name: str
    listed_at: str | None
    size_bytes: int
    sha256: str


def read_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def column_names(record_type: type[Record]) -> list[str]:
    return [f.name for f in fields(record_type)]


def dedupe[T](records: Iterable[T], key: Callable[[T], object]) -> tuple[list[T], int]:
    """Keep the last record per key (later rows in the export are later edits)."""
    by_key: dict[object, T] = {}
    total = 0
    for record in records:
        by_key[key(record)] = record
        total += 1
    return list(by_key.values()), total - len(by_key)


def copy_records(
    conn: Conn, table: str, records: Sequence[Record], record_type: type[Record]
) -> None:
    statement = sql.SQL("COPY {} ({}) FROM STDIN").format(
        sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, column_names(record_type)))
    )
    with conn.cursor() as cur, cur.copy(statement) as copy:
        for record in records:
            copy.write_row(astuple(record))


def latest_snapshot_sha(conn: Conn) -> str | None:
    row = conn.execute("SELECT sha256 FROM source_snapshots ORDER BY id DESC LIMIT 1").fetchone()
    return str(row[0]) if row else None


def build_records(
    csv_dir: Path, report: LoadReport
) -> dict[str, tuple[type[Record], list[Record]]]:
    codes = CodeBook.from_csv(csv_dir / "eADMSPUB_DataDictionary.csv")

    events, dup = dedupe(
        (n.normalize_event(r, codes) for r in read_csv(csv_dir / "events.csv")), lambda e: e.ev_id
    )
    report.duplicates_dropped["events"] = dup
    event_ids = {e.ev_id for e in events}

    aircraft, dup = dedupe(
        (n.normalize_aircraft(r, codes) for r in read_csv(csv_dir / "aircraft.csv")),
        lambda a: (a.ev_id, a.aircraft_key),
    )
    report.duplicates_dropped["aircraft"] = dup
    aircraft = [a for a in aircraft if a.ev_id in event_ids]
    aircraft_keys = {(a.ev_id, a.aircraft_key) for a in aircraft}

    def children[T: (n.Injury, n.Occurrence, n.Finding)](
        name: str, rows: Iterable[T], key: Callable[[T], object]
    ) -> list[Record]:
        unique, dup = dedupe(rows, key)
        kept = [r for r in unique if (r.ev_id, r.aircraft_key) in aircraft_keys]
        report.duplicates_dropped[name] = dup
        report.orphans_dropped[name] = len(unique) - len(kept)
        return list(kept)

    injuries = children(
        "injuries",
        (n.normalize_injury(r, codes) for r in read_csv(csv_dir / "injury.csv")),
        lambda i: (i.ev_id, i.aircraft_key, i.person_category, i.injury_level),
    )
    occurrences = children(
        "occurrences",
        (n.normalize_occurrence(r, codes) for r in read_csv(csv_dir / "Events_Sequence.csv")),
        lambda o: (o.ev_id, o.aircraft_key, o.occurrence_no),
    )
    findings = children(
        "findings",
        (n.normalize_finding(r) for r in read_csv(csv_dir / "Findings.csv")),
        lambda f: (f.ev_id, f.aircraft_key, f.finding_no),
    )
    narratives, dup = dedupe(
        (n.normalize_narrative(r) for r in read_csv(csv_dir / "narratives.csv")),
        lambda x: (x.ev_id, x.aircraft_key),
    )
    report.duplicates_dropped["narratives"] = dup
    kept_narratives = [x for x in narratives if x.ev_id in event_ids]
    report.orphans_dropped["narratives"] = len(narratives) - len(kept_narratives)

    return {
        "events": (n.Event, list(events)),
        "aircraft": (n.Aircraft, list(aircraft)),
        "injuries": (n.Injury, injuries),
        "occurrences": (n.Occurrence, occurrences),
        "findings": (n.Finding, findings),
        "narratives": (n.Narrative, list(kept_narratives)),
    }


def _merge_events(conn: Conn, events: Sequence[Record]) -> None:
    cols = column_names(n.Event)
    conn.execute("CREATE TEMP TABLE stage_events (LIKE events) ON COMMIT DROP")
    copy_records(conn, "stage_events", events, n.Event)
    ident = sql.Identifier
    staged = sql.SQL(", ").join(sql.SQL("s.{}").format(ident(c)) for c in cols)
    updates = sql.SQL(", ").join(
        sql.SQL("{} = s.{}").format(ident(c), ident(c)) for c in cols if c != "ev_id"
    )
    conn.execute(
        sql.SQL(
            "MERGE INTO events e USING stage_events s ON e.ev_id = s.ev_id "
            "WHEN MATCHED THEN UPDATE SET {updates} "
            "WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ({staged}) "
            "WHEN NOT MATCHED BY SOURCE THEN DELETE"
        ).format(updates=updates, cols=sql.SQL(", ").join(map(ident, cols)), staged=staged)
    )


# Each narrative section becomes one document. Provenance points at the CAROL final
# report when the key is derivable, otherwise at the public docket page.
_DOCUMENTS_FROM_NARRATIVES = """
INSERT INTO documents (ev_id, aircraft_key, section, source, source_url, title, body,
                       extraction_method, metadata)
SELECT nr.ev_id, nr.aircraft_key, s.section, 'avall',
       coalesce(e.report_url, e.docket_url),
       concat_ws(' ', e.ntsb_no, '-', initcap(replace(s.section, '_', ' '))),
       s.body, 'database',
       jsonb_build_object('ntsb_no', e.ntsb_no, 'ev_date', e.ev_date)
FROM narratives nr
JOIN events e USING (ev_id)
CROSS JOIN LATERAL (VALUES ('factual', nr.factual),
                           ('analysis', nr.analysis),
                           ('probable_cause', nr.probable_cause)) AS s(section, body)
WHERE s.body IS NOT NULL
  -- ~2,600 short-form reports repeat the factual text verbatim as the analysis.
  AND NOT (s.section = 'analysis' AND nr.analysis = nr.factual)
"""


def load_snapshot(
    conn: Conn, csv_dir: Path, snapshot: Snapshot, *, force: bool = False
) -> LoadReport:
    report = LoadReport()
    if not force and latest_snapshot_sha(conn) == snapshot.sha256:
        report.skipped = True
        return report

    records = build_records(csv_dir, report)
    with conn.transaction():
        _merge_events(conn, records["events"][1])
        conn.execute("TRUNCATE aircraft, injuries, occurrences, findings, narratives")
        for table in ("aircraft", "injuries", "occurrences", "findings", "narratives"):
            record_type, rows = records[table]
            copy_records(conn, table, rows, record_type)
        conn.execute("DELETE FROM documents WHERE source = 'avall'")
        documents = conn.execute(_DOCUMENTS_FROM_NARRATIVES).rowcount
        report.loaded = {table: len(rows) for table, (_, rows) in records.items()}
        report.loaded["documents"] = documents
        conn.execute(
            "INSERT INTO source_snapshots (file_name, listed_at, size_bytes, sha256, row_counts) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                snapshot.file_name,
                snapshot.listed_at,
                snapshot.size_bytes,
                snapshot.sha256,
                Jsonb(report.loaded),
            ),
        )
    return report
