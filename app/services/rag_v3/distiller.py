from __future__ import annotations

import re
from typing import Any

from app.schemas import DistilledEvidence, EvidenceBundle, UserIntent
from app.services.openai_client import chat_completion


DISTILL_SYSTEM_PROMPT = """You compress news evidence without changing facts.
Return strict JSON with keys:
- summary
- key_points
- coverage
- notes
Only use the provided evidence. Return JSON only."""


def distill_evidence(intent: UserIntent, bundle: EvidenceBundle) -> DistilledEvidence:
    if not bundle.items:
        return DistilledEvidence(summary="", key_points=[], supporting_article_ids=[], coverage="no_evidence", notes=["No evidence retrieved."])
    supporting_article_ids = [item.article_id for item in bundle.items[:8] if item.article_id]
    if _should_build_timeline_overview(intent, bundle):
        return _timeline_distillation(intent, bundle, supporting_article_ids)
    if _should_build_archive_overview(intent, bundle):
        return _archive_overview_distillation(intent, bundle, supporting_article_ids)
    exact_contexts = [_format_exact_context(value) for value in (bundle.raw_filters.get("exact_contexts") or [])]
    exact_contexts = [value for value in exact_contexts if value]
    if intent.needs_count and intent.needs_summary and exact_contexts:
        key_points = exact_contexts[:4]
        return DistilledEvidence(
            summary="; ".join(key_points[:2]),
            key_points=key_points,
            supporting_article_ids=supporting_article_ids,
            coverage="structured_contexts",
            notes=["Used deterministic entity contexts without LLM distillation."],
        )
    if bundle.mode == "sql" and bundle.plan.answer_shape in {"count_only", "list"}:
        key_points = [
            f"{item.headline}: {(item.excerpt or '').strip()[:140]}".strip()
            for item in bundle.items[:4]
            if item.headline
        ]
        return DistilledEvidence(
            summary=" ".join(point for point in key_points[:2]),
            key_points=key_points,
            supporting_article_ids=supporting_article_ids,
            coverage="structured",
            notes=["Used deterministic structured evidence without LLM distillation."],
        )
    prompt = (
        f"Question: {intent.standalone_question}\n"
        "Evidence:\n" +
        "\n".join(
            f"- {item.headline or 'Untitled'} | {item.edition or 'Unknown edition'} | "
            f"{item.section or 'Unknown section'} | {item.issue_date or 'Unknown date'} | "
            f"{item.excerpt or ''}"
            for item in bundle.items[:8]
        ) +
        "\nReturn JSON only."
    )
    try:
        import json
        from app.services.rag_v3.common import safe_load_json

        payload = safe_load_json(chat_completion(DISTILL_SYSTEM_PROMPT, prompt, timeout=25.0))
        return DistilledEvidence(
            summary=payload.get("summary") or "",
            key_points=_ensure_text_list(payload.get("key_points")),
            supporting_article_ids=supporting_article_ids,
            coverage=payload.get("coverage") or "semantic",
            notes=_ensure_text_list(payload.get("notes")),
        )
    except Exception:
        return _fallback_distillation(intent, bundle, supporting_article_ids)


def _format_exact_context(value: Any) -> str:
    if isinstance(value, dict):
        headline = str(value.get("headline") or "").strip()
        article_count = value.get("article_count")
        section = str(value.get("section") or "").strip()
        excerpt = str(value.get("excerpt") or "").strip()
        prefix = headline or "Coverage"
        if article_count is not None:
            prefix += f" ({article_count} article{'s' if int(article_count) != 1 else ''})"
        details: list[str] = []
        if section:
            details.append(f"in {section}")
        excerpt_clause = _leading_clause(excerpt)
        if excerpt_clause:
            details.append(excerpt_clause)
        if details:
            return f"{prefix} {' '.join(details)}".strip()
        return prefix
    text = str(value or "").strip()
    return text


def _ensure_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _leading_clause(text: str) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return ""
    cleaned = cleaned.split(": ", 1)[-1]
    clause = re.split(r"(?<=[.!?])\s+|\s+\b(?:However|Meanwhile|But|And)\b", cleaned, maxsplit=1)[0].strip()
    return clause[:140].rstrip(" ,;:")


def _fallback_distillation(intent: UserIntent, bundle: EvidenceBundle, supporting_article_ids: list[str]) -> DistilledEvidence:
    key_points: list[str] = []
    for item in bundle.items[:4]:
        headline = (item.headline or "Untitled").strip()
        clause = _leading_clause(item.excerpt or "")
        key_points.append(f"{headline}: {clause}".strip(": ") if clause else headline)
    key_points = [point for point in key_points if point]
    lowered_question = (intent.standalone_question or "").lower()
    if any(token in lowered_question for token in ["latest", "most recent", "recent update", "latest update"]):
        summary = "Latest coverage includes: " + "; ".join(key_points[:2]) if key_points else ""
    else:
        summary = "; ".join(key_points[:2]) if key_points else ""
    return DistilledEvidence(
        summary=summary.strip(),
        key_points=key_points,
        supporting_article_ids=supporting_article_ids,
        coverage="fallback",
        notes=["Used fallback distillation after model error."],
    )


def _should_build_archive_overview(intent: UserIntent, bundle: EvidenceBundle) -> bool:
    if bundle.plan.task_type in {"compare", "ranking", "article_text", "list", "count"}:
        return False
    if intent.needs_count or intent.needs_listing or intent.needs_article_text:
        return False
    return bundle.plan.time_scope in {"all_time", "date_range"}


def _should_build_timeline_overview(intent: UserIntent, bundle: EvidenceBundle) -> bool:
    if not _should_build_archive_overview(intent, bundle):
        return False
    lowered = (intent.standalone_question or "").lower()
    return any(
        token in lowered
        for token in [
            "latest",
            "most recent",
            "recent update",
            "latest update",
            "after that",
            "after this",
            "what happened next",
            "develop",
            "timeline",
            "how did",
        ]
    )


def _archive_overview_distillation(intent: UserIntent, bundle: EvidenceBundle, supporting_article_ids: list[str]) -> DistilledEvidence:
    items = [item for item in bundle.items if item.headline]
    clusters = _cluster_story_items(items)
    key_points: list[str] = []
    for cluster in clusters[:4]:
        representative = cluster[0]
        headline = (representative.headline or "").strip()
        clause = _leading_clause(representative.excerpt or "")
        point = f"{headline}: {clause}".strip(": ") if clause else headline
        if len(cluster) > 1:
            point += f" ({len(cluster)} related reports)"
        key_points.append(point)
    dated_items = [item for item in items if item.issue_date]
    latest_date = max((item.issue_date for item in dated_items), default=None)
    earliest_date = min((item.issue_date for item in dated_items), default=None)
    year_counts = _year_counts(dated_items)
    lowered_question = (intent.standalone_question or "").lower()
    if any(token in lowered_question for token in ["latest", "most recent", "recent update", "latest update"]):
        summary = "Latest retrieved coverage"
        if latest_date:
            summary += f" on {latest_date}"
        if key_points:
            summary += f" includes: {'; '.join(key_points[:2])}"
    else:
        if latest_date and earliest_date and latest_date != earliest_date:
            span_text = f" from {earliest_date} to {latest_date}"
        elif latest_date:
            span_text = f" up to {latest_date}"
        else:
            span_text = ""
        summary = f"Archive coverage{span_text} includes"
        if key_points:
            summary += f": {'; '.join(key_points[:2])}"
        if year_counts:
            top_years = ", ".join(f"{year} ({count})" for year, count in year_counts[:3])
            summary += f". Retrieved coverage is concentrated in: {top_years}."
    notes = ["Built archive-style overview from retrieved evidence."]
    if clusters:
        notes.append("Grouped similar retrieved headlines into story-like developments.")
    if year_counts:
        notes.append("Year grouping derived from retrieved evidence, not the full corpus.")
    return DistilledEvidence(
        summary=summary.strip(),
        key_points=key_points,
        supporting_article_ids=supporting_article_ids,
        coverage="archive_overview",
        notes=notes,
    )


def _timeline_distillation(intent: UserIntent, bundle: EvidenceBundle, supporting_article_ids: list[str]) -> DistilledEvidence:
    items = [item for item in bundle.items if item.headline]
    clusters = _cluster_story_items(items)
    dated_clusters = sorted(
        clusters,
        key=lambda cluster: _cluster_latest_date(cluster) or "",
        reverse=True,
    )
    key_points: list[str] = []
    for cluster in dated_clusters[:4]:
        representative = cluster[0]
        point = _cluster_timeline_point(cluster, representative)
        if point:
            key_points.append(point)
    latest_cluster = dated_clusters[0] if dated_clusters else []
    latest_date = _cluster_latest_date(latest_cluster) if latest_cluster else None
    lowered = (intent.standalone_question or "").lower()
    if any(token in lowered for token in ["latest", "most recent", "recent update", "latest update"]):
        summary = "Latest update"
        if latest_date:
            summary += f" on {latest_date}"
        if key_points:
            summary += f": {_strip_leading_date(key_points[0], latest_date)}"
        if len(key_points) > 1:
            earlier_points = [_strip_leading_date(point) for point in key_points[1:3]]
            summary += f" Earlier related developments include: {'; '.join(earlier_points)}."
    elif any(token in lowered for token in ["after that", "after this", "what happened next"]):
        summary = "The next retrieved developments"
        if latest_date:
            summary += f" up to {latest_date}"
        if key_points:
            summary += f" include: {'; '.join(_strip_leading_date(point) for point in key_points[:3])}"
    else:
        summary = "Key developments over time"
        if key_points:
            summary += f": {'; '.join(_strip_leading_date(point) for point in reversed(key_points[:3]))}"
        if latest_date:
            summary += f". Latest retrieved update: {latest_date}."
    return DistilledEvidence(
        summary=summary.strip(),
        key_points=key_points,
        supporting_article_ids=supporting_article_ids,
        coverage="timeline_overview",
        notes=[
            "Built timeline-style overview from retrieved evidence.",
            "Ordered grouped developments chronologically by retrieved issue dates.",
        ],
    )


def _year_counts(items) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for item in items:
        if not item.issue_date:
            continue
        year = str(item.issue_date)[:4]
        counts[year] = counts.get(year, 0) + 1
    return sorted(counts.items(), key=lambda value: (-value[1], value[0]))


def _cluster_story_items(items) -> list[list]:
    clusters: list[list] = []
    for item in items:
        signature = _headline_signature(item.headline or "")
        if not signature:
            clusters.append([item])
            continue
        placed = False
        for cluster in clusters:
            cluster_signature = _headline_signature(cluster[0].headline or "")
            if _signatures_match(signature, cluster_signature):
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
    clusters.sort(key=lambda cluster: (-len(cluster), max((member.issue_date or "") for member in cluster)))
    return clusters


def _headline_signature(headline: str) -> set[str]:
    stop_words = {
        "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "by", "with", "from",
        "is", "was", "are", "were", "be", "this", "that", "these", "those", "says", "said",
        "pm", "cm", "will", "had", "has",
    }
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", (headline or "").lower())
        if len(token) >= 4 and token not in stop_words and not token.isdigit()
    ]
    return set(tokens[:8])


def _signatures_match(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    overlap = len(left & right)
    threshold = 2 if min(len(left), len(right)) >= 3 else 1
    return overlap >= threshold


def _cluster_latest_date(cluster) -> str | None:
    dates = [item.issue_date for item in cluster if item.issue_date]
    return max(dates) if dates else None


def _cluster_timeline_point(cluster, representative) -> str:
    headline = (representative.headline or "").strip()
    clause = _leading_clause(representative.excerpt or "")
    latest_date = _cluster_latest_date(cluster)
    prefix = f"{latest_date}: " if latest_date else ""
    point = f"{prefix}{headline}"
    if clause:
        point += f": {clause}"
    if len(cluster) > 1:
        point += f" ({len(cluster)} related reports)"
    return point.strip()


def _strip_leading_date(point: str, expected_date: str | None = None) -> str:
    text = (point or "").strip()
    if not text:
        return text
    if expected_date and text.startswith(f"{expected_date}: "):
        return text[len(expected_date) + 2 :]
    return re.sub(r"^20\d{2}-\d{2}-\d{2}:\s+", "", text)
