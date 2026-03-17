"""Cross-encoder reranking using sentence-transformers.

Loads ``cross-encoder/ms-marco-MiniLM-L-6-v2`` once at module level (80 MB,
CPU-fast).  Falls back gracefully to the original ordering when the model is
unavailable (e.g., ``sentence-transformers`` not installed).

Usage::

    from app.services.reranker import rerank

    indices = rerank(query, [text1, text2, ...])
    # indices is a list of positions sorted by descending relevance score
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _load_model():
    """Load the cross-encoder model once and cache it."""
    from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore[import]

    model = CrossEncoder(_MODEL_NAME)
    logger.info("Cross-encoder reranker loaded: %s", _MODEL_NAME)
    return model


def rerank(query: str, candidate_texts: list[str]) -> list[int]:
    """Return indices of ``candidate_texts`` sorted by descending relevance.

    If fewer than 2 candidates are supplied, returns the original order.
    On any error (missing package, model unavailable), returns original order
    so callers can safely fall back.
    """
    if len(candidate_texts) < 2:
        return list(range(len(candidate_texts)))
    try:
        model = _load_model()
        pairs = [(query, text) for text in candidate_texts]
        scores = model.predict(pairs)
        # Sort indices by descending score.
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [i for i, _ in indexed]
    except Exception as exc:
        logger.warning("Cross-encoder reranking failed, keeping original order: %s", exc)
        return list(range(len(candidate_texts)))
