from __future__ import annotations

import re

from app.schemas import AnswerDraft, DistilledEvidence, EvidenceBundle, UserIntent
from app.services.openai_client import chat_completion
def generate_answer(intent: UserIntent, bundle: EvidenceBundle, distilled: DistilledEvidence) -> AnswerDraft:
    if bundle.plan.task_type == "compare":
        answer = _compare_answer(intent, distilled)
        return AnswerDraft(answer=answer, mode=bundle.mode, citations=[])
    if bundle.plan.task_type == "ranking":
        section_counts = bundle.raw_filters.get("section_counts") or []
        publication_counts = bundle.raw_filters.get("publication_counts") or []
        author_counts = bundle.raw_filters.get("author_counts") or []
        year_counts = bundle.raw_filters.get("year_counts") or []
        ranking_kind = bundle.raw_filters.get("ranking_kind") or "section"
        if ranking_kind == "edition":
            ranking_rows = publication_counts
        elif ranking_kind == "author":
            ranking_rows = author_counts
        elif ranking_kind == "year":
            ranking_rows = year_counts
        else:
            ranking_rows = section_counts
        if ranking_rows:
            direction = _ranking_direction(bundle.question)
            if ranking_kind == "edition":
                label_key = "publication_name"
                subject_name = "edition"
            elif ranking_kind == "author":
                label_key = "author"
                subject_name = "author"
            elif ranking_kind == "year":
                label_key = "year"
                subject_name = "year"
            else:
                label_key = "section"
                subject_name = "section"
            chosen = _select_ranked_row(ranking_rows, direction, label_key)
            chosen_section = str(chosen.get(label_key) or f"Unknown {subject_name}")
            chosen_count = int(chosen.get("article_count") or 0)
            answer = f"The {subject_name} with the {direction} articles was {chosen_section} with {chosen_count} articles."
            if len(ranking_rows) > 1 and direction == "most":
                ranking_preview = ", ".join(
                    f"{str(row.get(label_key) or f'Unknown {subject_name}')} ({int(row.get('article_count') or 0)})"
                    for row in ranking_rows[:3]
                )
                answer += f" Top {subject_name}s: {ranking_preview}."
            elif len(ranking_rows) > 1 and direction == "least":
                least_preview = ", ".join(
                    f"{str(row.get(label_key) or f'Unknown {subject_name}')} ({int(row.get('article_count') or 0)})"
                    for row in list(ranking_rows)[-3:]
                )
                answer += f" Lowest {subject_name}s: {least_preview}."
            return AnswerDraft(answer=answer, mode=bundle.mode, citations=[])
    if intent.needs_article_text and bundle.items:
        item = bundle.items[0]
        return AnswerDraft(
            answer=(
                f"{item.headline or 'Untitled'}\n"
                f"{item.section or 'Unknown section'} | {item.issue_date or 'Unknown date'}\n\n"
                f"{item.excerpt or 'No excerpt available.'}"
            ),
            mode=bundle.mode,
            citations=_citations(bundle, limit=1),
        )
    if intent.needs_listing:
        lines = [f"I found {len(bundle.items)} matching articles."]
        for index, item in enumerate(bundle.items[:10], start=1):
            lines.append(
                f"{index}. {item.headline or 'Untitled'} | {item.edition or 'Unknown edition'} | "
                f"{item.section or 'Unknown section'} | {item.issue_date or 'Unknown date'}"
            )
        return AnswerDraft(answer="\n".join(lines), mode=bundle.mode, citations=_citations(bundle))
    if intent.needs_count:
        count = _deterministic_count(bundle)
        if intent.needs_summary:
            summary = _count_summary_text(distilled)
            answer = f"I found {count} relevant article{'s' if count != 1 else ''}."
            if summary:
                answer += f" The main coverage points were: {summary}"
            return AnswerDraft(answer=answer, mode=bundle.mode, citations=[])
        return AnswerDraft(
            answer=f"I found {count} relevant article{'s' if count != 1 else ''}.",
            mode=bundle.mode,
            citations=[],
        )
    if not bundle.items:
        return AnswerDraft(
            answer="I couldn't find enough grounded evidence in the current dataset to answer that confidently.",
            mode=bundle.mode,
            citations=[],
            grounded=False,
        )
    answer = distilled.summary or "; ".join(distilled.key_points[:3])
    if not answer or _is_abstention_like_answer(answer):
        answer = "I couldn't find enough grounded evidence in the current dataset to answer that confidently."
    return AnswerDraft(answer=answer, mode=bundle.mode, citations=[])


def _deterministic_count(bundle: EvidenceBundle) -> int:
    value = bundle.raw_filters.get("exact_article_count")
    if value is not None:
        return int(value)
    for item in bundle.items:
        metadata = item.metadata or {}
        if "author_article_count" in metadata:
            return int(metadata["author_article_count"])
    return len(bundle.items)


def _citations(bundle: EvidenceBundle, limit: int = 6) -> list[dict]:
    return [
        {
            "article_id": item.article_id,
            "headline": item.headline,
            "edition": item.edition,
            "section": item.section,
            "issue_date": item.issue_date,
            "reference_text": item.excerpt or "",
        }
        for item in bundle.items[:limit]
    ]


def _count_summary_text(distilled: DistilledEvidence) -> str:
    if distilled.coverage == "structured_contexts" and distilled.key_points:
        cleaned_points = [_headline_like_summary(point) for point in distilled.key_points[:3]]
        cleaned_points = [point for point in cleaned_points if point]
        if cleaned_points:
            if len(cleaned_points) == 1:
                return cleaned_points[0]
            if len(cleaned_points) == 2:
                return f"{cleaned_points[0]}, and {cleaned_points[1]}"
            return f"{cleaned_points[0]}, {cleaned_points[1]}, and {cleaned_points[2]}"
    return distilled.summary or "; ".join(distilled.key_points[:3])


def _headline_like_summary(point: str) -> str:
    text = " ".join((point or "").split()).strip()
    if not text:
        return ""
    count_match = re.search(r"\(\d+\s+articles?\)", text)
    if count_match:
        return text[:count_match.end()].strip()
    match = re.match(r"^(.*?\(\d+\s+articles?\))\b", text)
    if match:
        return match.group(1).strip()
    text = re.split(r"\s+in\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b", text, maxsplit=1)[0].strip()
    return text


def _is_abstention_like_answer(answer: str) -> bool:
    lowered = answer.lower()
    return any(
        phrase in lowered
        for phrase in [
            "provided evidence does not contain",
            "does not contain any information",
            "there is no news about",
            "no news about",
            "not present in the provided evidence",
        ]
    )


def _compare_answer(intent: UserIntent, distilled: DistilledEvidence) -> str:
    summary = distilled.summary.strip()
    points = [point.strip() for point in distilled.key_points[:6] if point.strip()]
    if summary or points:
        prompt = (
            f"Question: {intent.standalone_question}\n"
            f"Entity terms: {intent.entities.get('entity_terms', [])}\n"
            f"Summary evidence: {summary}\n"
            f"Key points: {points}\n"
            "Write a short grounded comparison in 2-4 sentences. "
            "Only use the provided evidence. Do not invent facts or counts."
        )
        try:
            result = chat_completion(
                "You write concise grounded comparisons for a news RAG system. Output only the answer.",
                prompt,
                timeout=18.0,
            ).strip()
            if result:
                return result
        except Exception:
            pass
    if summary:
        return summary
    if points:
        return "; ".join(points[:3])
    return "I couldn't find enough grounded evidence in the current dataset to answer that confidently."


def _ranking_direction(question: str) -> str:
    lowered = question.lower()
    if any(token in lowered for token in ["least", "lowest"]):
        return "least"
    return "most"


def _select_ranked_row(section_counts, direction: str, label_key: str):
    rows = [row for row in section_counts if row.get(label_key) is not None]
    if not rows:
        return {label_key: None, "article_count": 0}
    return rows[-1] if direction == "least" else rows[0]
