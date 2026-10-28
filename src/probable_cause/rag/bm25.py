"""Okapi BM25 over chunks with bm25s (sparse matrices, in memory, persisted to disk).

Postgres full-text search was the milestone-1 baseline, but ``ts_rank_cd`` is not BM25: it
has no inverse document frequency and no term-frequency saturation, and natural-language
questions need OR semantics, which makes GIN-backed ranking scan most of the table.
bm25s gives real BM25 (k1=1.5, b=0.75, Lucene IDF) with English stemming and stopwords,
answers a query in milliseconds, and supports a per-query document mask for filters.
"""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import bm25s
import numpy as np
import Stemmer
from numpy.typing import NDArray

_STEMMER = Stemmer.Stemmer("english")


def tokenize(texts: list[str], *, show_progress: bool = False) -> bm25s.tokenization.Tokenized:
    return bm25s.tokenize(texts, stopwords="en", stemmer=_STEMMER, show_progress=show_progress)


@dataclass
class Bm25Index:
    model: bm25s.BM25
    chunk_ids: NDArray[np.int64]
    ev_ids: list[str]
    _positions: dict[str, NDArray[np.int64]] | None = None

    def positions_for_events(self, events: set[str]) -> NDArray[np.int64]:
        if self._positions is None:
            by_event: dict[str, list[int]] = {}
            for i, ev in enumerate(self.ev_ids):
                by_event.setdefault(ev, []).append(i)
            self._positions = {k: np.array(v, dtype=np.int64) for k, v in by_event.items()}
        parts = [self._positions[e] for e in events if e in self._positions]
        return np.concatenate(parts) if parts else np.array([], dtype=np.int64)

    def search(
        self, query: str, k: int, allowed_events: set[str] | None = None
    ) -> list[tuple[int, float]]:
        """Return (chunk id, score) pairs, best first. Only positive scores are returned."""
        tokens = tokenize([query])
        if not tokens.vocab:
            return []
        mask = None
        if allowed_events is not None:
            mask = np.zeros(len(self.chunk_ids), dtype=np.float32)
            mask[self.positions_for_events(allowed_events)] = 1.0
            if not mask.any():
                return []
        k = min(k, len(self.chunk_ids))
        docs, scores = self.model.retrieve(
            tokens, k=k, show_progress=False, weight_mask=mask, n_threads=1
        )
        return [
            (int(self.chunk_ids[d]), float(s))
            for d, s in zip(docs[0], scores[0], strict=True)
            if s > 0
        ]


def build(path: Path, chunk_ids: NDArray[np.int64], ev_ids: list[str], texts: list[str]) -> None:
    model = bm25s.BM25()
    model.index(tokenize(texts), show_progress=False)
    path.mkdir(parents=True, exist_ok=True)
    model.save(str(path), show_progress=False)
    np.save(path / "chunk_ids.npy", chunk_ids)
    (path / "ev_ids.json").write_text(json.dumps(ev_ids), encoding="utf-8")


@cache
def load(path: Path) -> Bm25Index:
    model = bm25s.BM25.load(str(path))
    chunk_ids = np.load(path / "chunk_ids.npy")
    ev_ids = json.loads((path / "ev_ids.json").read_text(encoding="utf-8"))
    return Bm25Index(model, chunk_ids, ev_ids)
