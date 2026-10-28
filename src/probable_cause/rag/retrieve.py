"""Retrieval: BM25, dense (pgvector HNSW), hybrid (reciprocal rank fusion), optional
cross-encoder reranking, structured filters from a query plan, and event-level ranking.

Everything a caller might want to inspect (per-retriever ranks and scores, rerank
movements, the filter set size, timings) is kept on the result for the debug view.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import psycopg
from psycopg import sql

from probable_cause import tracing
from probable_cause.rag import bm25
from probable_cause.rag.chunking import Strategy
from probable_cause.rag.embeddings import embed_query, rerank_scores
from probable_cause.rag.filters import QueryPlan, filter_sql

type Conn = psycopg.Connection[tuple[object, ...]]
type Method = Literal["bm25", "dense", "hybrid"]
METHODS: tuple[Method, ...] = ("bm25", "dense", "hybrid")

RRF_K = 60
type Ranked = list[tuple[int, float]]  # (chunk id, score), best first


@dataclass(frozen=True)
class RetrievalConfig:
    strategy: Strategy = "section"
    method: Method = "hybrid"
    rerank: bool = False
    rewrite: bool = False
    candidates: int = 100  # per retriever, before fusion
    rerank_top: int = 20
    context_chunks: int = 8  # what the generator sees

    @property
    def name(self) -> str:
        parts: list[str] = [self.strategy, self.method]
        parts += ["rerank"] if self.rerank else []
        parts += ["rewrite"] if self.rewrite else []
        return "+".join(parts)


@dataclass
class RetrievedChunk:
    chunk_id: int
    ev_id: str
    section: str
    heading: str | None
    match_text: str  # the indexed text that matched (a child, for parent_child)
    context_text: str  # what the generator reads (the parent, for parent_child)
    context_id: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # bm25 / dense / rrf / rerank
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def citation_key(self) -> str:
        return f"{self.ev_id}:{self.section}"


@dataclass
class RetrievalResult:
    question: str
    text_query: str
    plan: QueryPlan | None
    filtered_events: int | None  # size of the filter set, when filters applied
    chunks: list[RetrievedChunk]
    events: list[str]  # unique ev_ids in rank order
    timings_ms: dict[str, float] = field(default_factory=dict)
    top_similarity: float = 0.0  # max cosine(query, context chunk): the refusal signal

    def context(self, n: int) -> list[RetrievedChunk]:
        """Top chunks for the generator, deduplicated by context unit (parents)."""
        seen: set[int] = set()
        out: list[RetrievedChunk] = []
        for chunk in self.chunks:
            if chunk.context_id not in seen:
                seen.add(chunk.context_id)
                out.append(chunk)
            if len(out) == n:
                break
        return out


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank), rank from 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def unique_in_order(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vector.tolist()) + "]"


class Retriever:
    def __init__(
        self,
        conn: Conn,
        index_dir: Path,
        *,
        embed: Callable[[str], np.ndarray] = embed_query,
        rerank: Callable[[str, Sequence[str]], list[float]] = rerank_scores,
    ) -> None:
        self.conn = conn
        self.index_dir = index_dir
        self._embed = embed
        self._rerank = rerank
        self._query_vectors: dict[str, np.ndarray] = {}
        self.rerank_cache: dict[tuple[str, int], float] = {}

    # -- components -------------------------------------------------------------------

    def query_vector(self, text: str) -> np.ndarray:
        if text not in self._query_vectors:
            self._query_vectors[text] = self._embed(text)
        return self._query_vectors[text]

    def allowed_events(self, plan: QueryPlan | None) -> set[str] | None:
        if plan is None or plan.filters.is_empty():
            return None
        query, params = filter_sql(plan.filters)
        return {str(r[0]) for r in self.conn.execute(query, params)}

    def bm25_search(
        self, strategy: Strategy, text: str, k: int, allowed: set[str] | None
    ) -> list[tuple[int, float]]:
        index = bm25.load(self.index_dir / f"bm25-{strategy}")
        return index.search(text, k, allowed)

    def dense_search(
        self, strategy: Strategy, text: str, k: int, allowed: set[str] | None
    ) -> list[tuple[int, float]]:
        vector = _vector_literal(self.query_vector(text))
        if allowed is not None and not allowed:
            return []
        event_filter = (
            sql.SQL("AND ev_id = ANY(%(allowed)s)") if allowed is not None else sql.SQL("")
        )
        query = sql.SQL(
            "SELECT id, 1 - (embedding <=> %(v)s::vector) FROM chunks "
            "WHERE strategy = {strategy} AND NOT is_parent {event_filter} "
            "ORDER BY embedding <=> %(v)s::vector LIMIT %(k)s"
        ).format(strategy=sql.Literal(strategy), event_filter=event_filter)
        with self.conn.transaction():
            # Filtered HNSW scans keep searching until k rows pass the filter (pgvector 0.8).
            self.conn.execute("SET LOCAL hnsw.ef_search = 200")
            self.conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = self.conn.execute(
                query, {"v": vector, "k": k, "allowed": list(allowed or [])}
            ).fetchall()
        hits = [(int(r[0]), float(r[1])) for r in rows]  # type: ignore[call-overload, arg-type]
        return sorted(hits, key=lambda h: -h[1])

    # -- the pipeline -----------------------------------------------------------------

    def retrieve(
        self, question: str, config: RetrievalConfig, plan: QueryPlan | None = None
    ) -> RetrievalResult:
        timings: dict[str, float] = {}
        # Without rewriting, the retriever sees the question verbatim.
        rewritten = plan.text_query.strip() if plan and config.rewrite else ""
        text = rewritten or question
        started = time.perf_counter()
        allowed = self.allowed_events(plan) if config.rewrite else None
        timings["filter"] = (time.perf_counter() - started) * 1000

        with tracing.span("retrieve", kind="tool", input={"text": text, "config": config.name}):
            ranked, ranks, scores = self._candidates(config, text, allowed, timings)
        chunks = self._load_chunks(config.strategy, [cid for cid, _ in ranked])
        for (cid, score), chunk in zip(ranked, chunks, strict=True):
            chunk.score = score
            chunk.ranks = {name: r[cid] for name, r in ranks.items() if cid in r}
            chunk.scores = {name: s[cid] for name, s in scores.items() if cid in s}

        if config.rerank and chunks:
            started = time.perf_counter()
            with tracing.span("rerank", kind="tool", input={"candidates": config.rerank_top}):
                chunks = self._apply_rerank(text, chunks, config.rerank_top)
            timings["rerank"] = (time.perf_counter() - started) * 1000

        result = RetrievalResult(
            question=question,
            text_query=text,
            plan=plan if config.rewrite else None,
            filtered_events=len(allowed) if allowed is not None else None,
            chunks=chunks,
            events=unique_in_order([c.ev_id for c in chunks]),
            timings_ms=timings,
        )
        return result

    def _candidates(
        self, config: RetrievalConfig, text: str, allowed: set[str] | None,
        timings: dict[str, float],
    ) -> tuple[Ranked, dict[str, dict[int, int]], dict[str, dict[int, float]]]:  # fmt: skip
        lists: dict[str, list[tuple[int, float]]] = {}
        if config.method in ("bm25", "hybrid"):
            started = time.perf_counter()
            lists["bm25"] = self.bm25_search(config.strategy, text, config.candidates, allowed)
            timings["bm25"] = (time.perf_counter() - started) * 1000
        if config.method in ("dense", "hybrid"):
            started = time.perf_counter()
            lists["dense"] = self.dense_search(config.strategy, text, config.candidates, allowed)
            timings["dense"] = (time.perf_counter() - started) * 1000
        ranks = {n: {cid: i for i, (cid, _) in enumerate(hits, 1)} for n, hits in lists.items()}
        scores = {n: dict(hits) for n, hits in lists.items()}
        if config.method == "hybrid":
            fused = rrf([[cid for cid, _ in hits] for hits in lists.values()])
            ranks["rrf"] = {cid: i for i, (cid, _) in enumerate(fused, 1)}
            scores["rrf"] = dict(fused)
            return fused, ranks, scores
        return lists[config.method], ranks, scores

    def _load_chunks(self, strategy: Strategy, ids: list[int]) -> list[RetrievedChunk]:
        if not ids:
            return []
        rows = self.conn.execute(
            "SELECT c.id, c.ev_id, c.section, c.heading, c.body, "
            "coalesce(p.body, c.body), coalesce(p.id, c.id) "
            "FROM chunks c LEFT JOIN chunks p ON p.id = c.parent_id WHERE c.id = ANY(%s)",
            (ids,),
        ).fetchall()
        by_id = {
            int(r[0]): RetrievedChunk(  # type: ignore[call-overload]
                chunk_id=int(r[0]),  # type: ignore[call-overload]
                ev_id=str(r[1]),
                section=str(r[2]),
                heading=None if r[3] is None else str(r[3]),
                match_text=str(r[4]),
                context_text=str(r[5]),
                context_id=int(r[6]),  # type: ignore[call-overload]
                score=0.0,
            )
            for r in rows
        }
        return [by_id[i] for i in ids if i in by_id]

    def _apply_rerank(
        self, text: str, chunks: list[RetrievedChunk], top: int
    ) -> list[RetrievedChunk]:
        head, tail = chunks[:top], chunks[top:]
        todo = [c for c in head if (text, c.chunk_id) not in self.rerank_cache]
        if todo:
            for chunk, score in zip(
                todo, self._rerank(text, [c.match_text for c in todo]), strict=True
            ):
                self.rerank_cache[(text, chunk.chunk_id)] = score
        for chunk in head:
            chunk.scores["rerank"] = self.rerank_cache[(text, chunk.chunk_id)]
        head.sort(key=lambda c: -c.scores["rerank"])
        for rank, chunk in enumerate(head, 1):
            chunk.ranks["rerank"] = rank
            chunk.score = chunk.scores["rerank"]
        return head + tail

    def score_similarity(self, text: str, chunk_ids: list[int]) -> float:
        """Max cosine similarity between the query and the given chunks."""
        if not chunk_ids:
            return 0.0
        row = self.conn.execute(
            "SELECT max(1 - (embedding <=> %s::vector)) FROM chunks "
            "WHERE id = ANY(%s) AND embedding IS NOT NULL",
            (_vector_literal(self.query_vector(text)), chunk_ids),
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0  # type: ignore[arg-type]
