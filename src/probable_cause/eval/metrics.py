"""Retrieval and agreement metrics. Relevance is binary at the event level.

Recall is *capped*: |gold ∩ top-k| / min(|gold|, k). For lookup and structured items
(1-10 gold events) this is ordinary recall at k=10; for analytical items with dozens of gold
events it asks "how much of the first page is relevant" instead of being bounded by k/|gold|.
"""

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = len(set(ranked[:k]) & set(gold))
    return hits / min(len(set(gold)), k)


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str]) -> float:
    golden = set(gold)
    for rank, item in enumerate(ranked, start=1):
        if item in golden:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    golden = set(gold)
    if not golden:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in golden)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(golden), k)))
    return dcg / ideal


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Agreement between two binary raters, corrected for chance."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty label lists")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
