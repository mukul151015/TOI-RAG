from __future__ import annotations

from app.core.config import get_settings
from app.schemas import EvidenceItem


_settings = get_settings()


def rerank_items(query: str, items: list[EvidenceItem]) -> tuple[list[EvidenceItem], bool]:
    if not _settings.reranking_enabled or len(items) < 2:
        return items, False
    try:
        from app.services.reranker import rerank  # noqa: PLC0415

        candidate_texts = [
            " ".join(
                part for part in [
                    item.headline or "",
                    item.excerpt or "",
                    item.section or "",
                    item.edition or "",
                ]
                if part
            )
            for item in items
        ]
        reranked_indices = rerank(query, candidate_texts)
        return [items[index] for index in reranked_indices], True
    except Exception:
        return items, False
