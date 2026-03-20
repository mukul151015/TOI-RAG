from __future__ import annotations

from difflib import SequenceMatcher
import re

from app.schemas import AnswerDraft, EvidenceBundle, UserIntent, VerificationReport
from app.services.rag_v3.common import first_number
from app.services.openai_client import chat_completion
from app.services.rag_v3.common import safe_load_json


def verify_answer(intent: UserIntent, bundle: EvidenceBundle, draft: AnswerDraft) -> VerificationReport:
    answer_lower = draft.answer.lower()
    unsupported: list[str] = []
    supported: list[str] = []
    deterministic_count = bundle.raw_filters.get("exact_article_count")
    deterministic_count_only = intent.needs_count and not intent.needs_summary and deterministic_count is not None and bundle.plan.task_type != "ranking"
    abstained = (
        "couldn't verify a grounded answer" in answer_lower
        or "couldn't find enough grounded evidence" in answer_lower
        or "i can only answer questions grounded in the dataset" in answer_lower
    )

    if intent.needs_count and deterministic_count is not None and bundle.plan.task_type != "ranking":
        returned_number = first_number(draft.answer)
        if returned_number is None:
            unsupported.append("Answer did not state a count for a count request.")
        elif returned_number != int(deterministic_count):
            unsupported.append(f"Answer count {returned_number} did not match deterministic count {deterministic_count}.")
        else:
            supported.append(f"Deterministic count matched: {deterministic_count}.")

    if intent.entities.get("entity_terms") and "there were" in answer_lower and "about" not in answer_lower and "mentioning" not in answer_lower:
        unsupported.append("Answer fell back to a generic count form that ignored the requested topic/entity.")
    if not bundle.items and not deterministic_count_only and not abstained:
        unsupported.append("Answer was produced without evidence or abstention.")
    if intent.needs_summary and not draft.answer.strip():
        unsupported.append("Summary request returned an empty answer.")
    if bundle.items and not abstained and _should_use_semantic_verifier(intent, bundle):
        semantic_verdict = _semantic_verification(intent, bundle, draft)
        if semantic_verdict is False:
            if _looks_like_weak_match(intent, bundle):
                unsupported.append("Retrieved evidence did not appear to match the core query terms strongly enough.")
            else:
                unsupported.append("Semantic verification found the answer unsupported by the retrieved evidence.")
        elif semantic_verdict is True:
            supported.append("Semantic verification accepted the answer against retrieved evidence.")
        elif _looks_like_weak_match(intent, bundle):
            unsupported.append("Retrieved evidence did not appear to match the core query terms strongly enough.")
    elif bundle.items and not abstained and _looks_like_weak_match(intent, bundle):
        unsupported.append("Retrieved evidence did not appear to match the core query terms strongly enough.")
    if abstained and not intent.needs_count:
        supported.append("Answer abstained because grounded evidence was insufficient.")

    if not unsupported and bundle.items:
        supported.extend(_select_supported_headlines(intent, bundle))

    return VerificationReport(
        grounded=not unsupported,
        supported_claims=supported,
        unsupported_claims=unsupported,
        rationale="Answer accepted." if not unsupported else "Answer rejected due to unsupported or mismatched output.",
        answer_accepted=not unsupported,
    )


def _looks_like_weak_match(intent: UserIntent, bundle: EvidenceBundle) -> bool:
    if intent.needs_count:
        return False
    if bundle.plan.task_type == "ranking":
        return False
    if bundle.plan.author or intent.filters.get("author"):
        return False
    query_terms = _core_query_terms(intent)
    if not query_terms:
        return False
    evidence_text = " ".join(
        " ".join(
            filter(
                None,
                [
                    item.headline,
                    item.excerpt,
                    item.section,
                    item.edition,
                ],
            )
        ).lower()
        for item in bundle.items[:6]
    )
    matched = {term for term in query_terms if _contains_approximate_term(evidence_text, term)}
    required_matches = 1 if len(query_terms) <= 2 else 2
    return len(matched) < required_matches


def _should_use_semantic_verifier(intent: UserIntent, bundle: EvidenceBundle) -> bool:
    if intent.needs_count or bundle.plan.task_type == "ranking":
        return False
    if not bundle.items:
        return False
    if bundle.plan.author or intent.filters.get("author"):
        return False
    return bundle.plan.task_type in {"summary", "compare"} or intent.needs_summary


def _semantic_verification(intent: UserIntent, bundle: EvidenceBundle, draft: AnswerDraft) -> bool | None:
    if intent.needs_count or bundle.plan.task_type == "ranking":
        return None
    evidence_lines = []
    for item in bundle.items[:5]:
        evidence_lines.append(
            f"- {item.headline or 'Untitled'} | {item.section or 'Unknown section'} | "
            f"{item.issue_date or 'Unknown date'} | {(item.excerpt or '')[:220]}"
        )
    if not evidence_lines:
        return None
    prompt = (
        f"Question: {intent.standalone_question}\n"
        f"Answer: {draft.answer}\n"
        "Evidence:\n"
        + "\n".join(evidence_lines)
        + "\nReturn strict JSON with keys grounded (boolean) and rationale (string). "
          "Set grounded to true only if the answer is supported by the evidence and relevant to the question."
    )
    try:
        payload = safe_load_json(
            chat_completion(
                "You verify whether a RAG answer is grounded and relevant. Return JSON only.",
                prompt,
                timeout=10.0,
            )
        )
        grounded = payload.get("grounded")
        if isinstance(grounded, bool):
            return grounded
    except Exception:
        return None
    return None


def _core_query_terms(intent: UserIntent) -> list[str]:
    candidates = list(intent.entities.get("entity_terms") or [])
    if not candidates:
        candidates = [intent.standalone_question]
    stop_words = {
        "what", "did", "does", "is", "there", "any", "news", "about", "the", "a", "an",
        "in", "this", "archive", "context", "summary", "key", "points", "tell", "me",
        "show", "list", "how", "many", "were", "was", "on", "for", "and", "of",
    }
    terms: list[str] = []
    for candidate in candidates:
        for token in re.findall(r"[a-z0-9]+", candidate.lower()):
            if len(token) >= 4 and token not in stop_words:
                terms.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped[:5]


def _contains_approximate_term(haystack: str, term: str) -> bool:
    if not haystack or not term:
        return False
    if term in haystack:
        return True
    if len(term) < 5:
        return False
    hay_tokens = {token for token in re.findall(r"[a-z0-9]+", haystack.lower()) if len(token) >= 4}
    for token in hay_tokens:
        if token == term:
            return True
        if token[0] != term[0]:
            continue
        if abs(len(token) - len(term)) > 2:
            continue
        if SequenceMatcher(a=token, b=term).ratio() >= 0.83:
            return True
    return False


def _select_supported_headlines(intent: UserIntent, bundle: EvidenceBundle) -> list[str]:
    query_terms = _core_query_terms(intent)
    ranked: list[tuple[int, str]] = []
    for item in bundle.items[:6]:
        headline = item.headline or "supported evidence"
        haystack = " ".join(filter(None, [item.headline, item.excerpt, item.section, item.edition])).lower()
        score = sum(1 for term in query_terms if _contains_approximate_term(haystack, term))
        ranked.append((score, headline))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    minimum_score = 2 if len(query_terms) >= 2 else 1
    chosen = [headline for score, headline in ranked if score >= minimum_score][:3]
    if not chosen:
        chosen = [headline for score, headline in ranked if score > 0][:3]
    if chosen:
        return chosen
    return [item.headline or "supported evidence" for item in bundle.items[:3]]
