"""Local embedding and reranking models (fastembed / ONNX Runtime, CPU only).

Model choice. Throughput measured on this machine (Ryzen 7 7800X3D, CPU only, shared with
other workloads, 200-word chunks); quality is the MTEB English retrieval average (nDCG@10)
reported on each model card:

| model                               | dim | MTEB retrieval | chunks/s |
|-------------------------------------|-----|----------------|----------|
| snowflake/snowflake-arctic-embed-xs | 384 | 50.15          | 42.9     |
| BAAI/bge-small-en-v1.5              | 384 | 51.68          | 13-17    |
| snowflake/snowflake-arctic-embed-s  | 384 | 51.98          | 11.0     |
| nomic-embed-text-v1.5 (int8)        | 768 | ~53            | 9.1      |
| BAAI/bge-base-en-v1.5               | 768 | 53.25          | 4.9      |

About 400k chunks across three strategies have to be embedded on a CPU. arctic-embed-xs
does that in hours rather than most of a day, for 1.5 points of retrieval quality against
bge-small. If the benchmark shows dense retrieval is the bottleneck, swapping the model is
one constant and a re-embed.
"""

from collections.abc import Iterator, Sequence
from functools import cache

import numpy as np
from numpy.typing import NDArray

EMBED_MODEL = "snowflake/snowflake-arctic-embed-xs"
EMBED_DIM = 384
# Arctic-embed (like bge v1.5) prefixes queries with an instruction; passages get none.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

type Vector = NDArray[np.float32]


@cache
def _embedder() -> object:
    from fastembed import TextEmbedding

    return TextEmbedding(EMBED_MODEL)


@cache
def _reranker() -> object:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(RERANK_MODEL)


def embed_passages(texts: Sequence[str], batch_size: int = 32) -> Iterator[Vector]:
    model = _embedder()
    yield from model.embed(list(texts), batch_size=batch_size)  # type: ignore[attr-defined]


def embed_query(text: str) -> Vector:
    model = _embedder()
    vector: Vector = next(iter(model.embed([QUERY_PREFIX + text])))  # type: ignore[attr-defined]
    return vector


def rerank_scores(query: str, passages: Sequence[str]) -> list[float]:
    """Cross-encoder relevance logits, one per passage (higher is more relevant)."""
    if not passages:
        return []
    model = _reranker()
    return [float(s) for s in model.rerank(query, list(passages), batch_size=32)]  # type: ignore[attr-defined]
