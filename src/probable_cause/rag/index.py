"""Build and maintain the retrieval indexes: chunks, embeddings, HNSW, BM25.

Every step is resumable and idempotent:
* ``build_chunks`` replaces one strategy's chunks inside a transaction.
* ``embed_pending`` embeds only rows whose embedding is NULL, committing per batch, so an
  interrupted run picks up where it stopped.
* ``build_hnsw`` and ``build_bm25`` are cheap to re-run.
"""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import psycopg
from psycopg import sql
from tqdm import tqdm

from probable_cause.rag import bm25
from probable_cause.rag.chunking import Chunk, SourceDocument, Strategy, make_chunker
from probable_cause.rag.embeddings import embed_passages

type Conn = psycopg.Connection[tuple[object, ...]]

# The retrieval corpus. Embedding every chunk of all three strategies for 2008-2026 is a
# ~4.5 hour CPU job (about 560k chunks at 35-70 chunks/s); events from 2018 on cover nine
# years and about 12.5k chunked events, and keep the full benchmark grid under an hour.
CORPUS_SINCE = date(2018, 1, 1)

_COLUMNS = (
    "id", "strategy", "ev_id", "document_id", "section", "heading", "ordinal", "body",
    "char_start", "char_end", "is_parent", "parent_id",
)  # fmt: skip


def iter_events(conn: Conn, since: date) -> Iterator[list[SourceDocument]]:
    """Yield each event's documents, streaming the table with a server-side cursor."""
    with conn.cursor(name="documents_by_event") as cur:
        cur.itersize = 2000
        cur.execute(
            "SELECT d.id, d.ev_id, d.section, d.body FROM documents d JOIN events e USING (ev_id) "
            "WHERE e.ev_date >= %s ORDER BY d.ev_id, d.id",
            (since,),
        )
        batch: list[SourceDocument] = []
        for doc_id, ev_id, section, body in cur:
            doc = SourceDocument(int(doc_id), str(ev_id), str(section), str(body))  # type: ignore[call-overload]
            if batch and batch[0].ev_id != doc.ev_id:
                yield batch
                batch = []
            batch.append(doc)
        if batch:
            yield batch


def _reserve_ids(conn: Conn, n: int) -> int:
    """Reserve n consecutive ids from the chunks sequence; return the first."""
    row = conn.execute(
        "SELECT setval('chunks_id_seq', nextval('chunks_id_seq') + %s - 1) - %s + 1", (n, n)
    ).fetchone()
    assert row is not None and isinstance(row[0], int)
    return row[0]


def _rows(strategy: Strategy, chunks: list[Chunk], first_id: int) -> Iterator[tuple[object, ...]]:
    next_id = first_id
    for chunk in chunks:
        parent_id = next_id
        is_parent = bool(chunk.children)
        yield (next_id, strategy, chunk.ev_id, chunk.document_id, chunk.section, chunk.heading,
               chunk.ordinal, chunk.text, chunk.char_start, chunk.char_end, is_parent,
               None)  # fmt: skip
        next_id += 1
        for child in chunk.children:
            yield (next_id, strategy, child.ev_id, child.document_id, child.section,
                   child.heading, child.ordinal, child.text, child.char_start, child.char_end,
                   False, parent_id)  # fmt: skip
            next_id += 1


def build_chunks(conn: Conn, strategy: Strategy, since: date = CORPUS_SINCE) -> int:
    chunker = make_chunker(strategy)
    events = conn.execute(
        "SELECT count(DISTINCT ev_id) FROM documents JOIN events USING (ev_id) WHERE ev_date >= %s",
        (since,),
    ).fetchone()
    total = int(events[0]) if events else 0  # type: ignore[call-overload]
    copy_sql = sql.SQL("COPY chunks ({}) FROM STDIN").format(
        sql.SQL(", ").join(map(sql.Identifier, _COLUMNS))
    )
    written = 0
    with psycopg.connect(conn.info.dsn, password=conn.info.password) as reader:
        conn.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))
        with conn.cursor() as cur, cur.copy(copy_sql) as copy:
            for documents in tqdm(
                iter_events(reader, since), total=total, desc=f"chunk {strategy}"
            ):
                chunks = chunker.chunk_event(documents)
                n = len(chunks) + sum(len(c.children) for c in chunks)
                if n == 0:
                    continue
                for row in _rows(strategy, chunks, _reserve_ids(reader, n)):
                    copy.write_row(row)
                written += n
    conn.commit()
    return written


@dataclass(frozen=True)
class EmbedReport:
    embedded: int
    seconds: float


def embed_pending(
    conn: Conn,
    strategy: Strategy,
    *,
    batch_size: int = 512,
    embed: Callable[[list[str]], list[np.ndarray]] | None = None,
) -> EmbedReport:
    embed = embed or (lambda texts: list(embed_passages(texts)))
    row = conn.execute(
        "SELECT count(*) FROM chunks WHERE strategy = %s AND embedding IS NULL AND NOT is_parent",
        (strategy,),
    ).fetchone()
    pending = int(row[0]) if row else 0  # type: ignore[call-overload]
    started, done = time.perf_counter(), 0
    with tqdm(total=pending, desc=f"embed {strategy}", unit="chunk") as bar:
        while True:
            batch = conn.execute(
                "SELECT id, body FROM chunks WHERE strategy = %s AND embedding IS NULL "
                "AND NOT is_parent ORDER BY id LIMIT %s",
                (strategy, batch_size),
            ).fetchall()
            if not batch:
                break
            vectors = embed([str(b) for _, b in batch])
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE chunks SET embedding = %s WHERE id = %s",
                    [(_vector_literal(v), i) for (i, _), v in zip(batch, vectors, strict=True)],
                )
            conn.commit()
            done += len(batch)
            bar.update(len(batch))
    return EmbedReport(done, time.perf_counter() - started)


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vector.tolist()) + "]"


def build_hnsw(conn: Conn, strategy: Strategy) -> float:
    started = time.perf_counter()
    # Parallel HNSW builds allocate dynamic shared memory, and Docker's default /dev/shm is
    # only 64 MB; a single-process build with a local work_mem budget avoids that.
    conn.execute("SET maintenance_work_mem = '512MB'")
    conn.execute("SET max_parallel_maintenance_workers = 0")
    conn.execute(
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {} ON chunks USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64) WHERE strategy = {} AND NOT is_parent"
        ).format(sql.Identifier(f"chunks_hnsw_{strategy}"), sql.Literal(strategy))
    )
    conn.commit()
    return time.perf_counter() - started


def build_bm25(conn: Conn, strategy: Strategy, index_dir: Path) -> int:
    rows = conn.execute(
        "SELECT id, ev_id, body FROM chunks WHERE strategy = %s AND NOT is_parent ORDER BY id",
        (strategy,),
    ).fetchall()
    ids = np.array([int(r[0]) for r in rows], dtype=np.int64)  # type: ignore[call-overload]
    ev_ids = [str(r[1]) for r in rows]
    bm25.build(index_dir / f"bm25-{strategy}", ids, ev_ids, [str(r[2]) for r in rows])
    return len(rows)
