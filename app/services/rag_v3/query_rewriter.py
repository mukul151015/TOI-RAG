from __future__ import annotations

from app.services.openai_client import chat_completion
from app.services.rag_v3.common import safe_load_json


REWRITE_SYSTEM_PROMPT = """You rewrite user questions for a grounded news RAG system.
Return strict JSON with keys:
- standalone_question
- references_session_context (boolean)
- reasoning
Preserve the original meaning. Resolve follow-ups using the supplied history and session context.
If the session context contains a prior topic, entity terms, answer shape, or time scope, use them when the user asks a follow-up.
Return JSON only."""


def rewrite_question(
    question: str,
    history: list[dict[str, str]] | None = None,
    session_context: dict | None = None,
) -> dict:
    history = history or []
    session_context = session_context or {}
    if not history and not session_context:
        return {
            "standalone_question": question,
            "references_session_context": False,
            "reasoning": "No follow-up context was needed.",
        }
    prompt = (
        f"Question: {question}\n"
        f"History: {history[-6:]}\n"
        f"Session context: {session_context}\n"
        "Return JSON only."
    )
    try:
        payload = safe_load_json(chat_completion(REWRITE_SYSTEM_PROMPT, prompt, timeout=20.0))
    except Exception:
        payload = {}
    standalone_question = payload.get("standalone_question") or _fallback_rewrite(question, session_context)
    standalone_question = _preserve_time_scope(standalone_question, session_context)
    return {
        "standalone_question": standalone_question,
        "references_session_context": bool(payload.get("references_session_context")),
        "reasoning": payload.get("reasoning") or "Used available session context conservatively.",
    }


def _fallback_rewrite(question: str, session_context: dict) -> str:
    subject = (
        session_context.get("last_topic")
        or session_context.get("query_focus")
        or " ".join(session_context.get("last_entity_terms") or [])
        or session_context.get("last_question")
    )
    if not subject:
        return question
    lowered = question.lower()
    time_scope = session_context.get("last_time_scope")
    last_issue_date = session_context.get("last_issue_date")
    scope_phrase = ""
    if time_scope == "single_issue" and last_issue_date:
        scope_phrase = f" on {last_issue_date}"
    elif time_scope == "all_time":
        scope_phrase = " in the archive"
    if any(token in lowered for token in ["latest", "most recent", "recent update", "latest update"]):
        return f"What is the latest news about {subject}?"
    if "which year" in lowered:
        return f"Which year had the most coverage about {subject}?"
    if any(token in lowered for token in ["after that", "after this", "what happened next", "what happened after"]):
        return f"What happened after that regarding {subject}{scope_phrase}?"
    if any(token in lowered for token in ["what about", "and context", "key points", "those", "them", "these", "their", "there", "it", "that"]):
        return f"{question} about {subject}{scope_phrase}"
    if any(token in lowered for token in ["more", "summary", "context", "coverage", "why", "how"]):
        return f"{question} regarding {subject}{scope_phrase}"
    return question


def _preserve_time_scope(standalone_question: str, session_context: dict) -> str:
    text = _normalize_latest_phrasing((standalone_question or "").strip())
    if not text:
        return text
    lowered = text.lower()
    time_scope = session_context.get("last_time_scope")
    last_issue_date = session_context.get("last_issue_date")
    last_start_date = session_context.get("last_start_date")
    last_end_date = session_context.get("last_end_date")
    if time_scope == "single_issue" and last_issue_date and last_issue_date not in text:
        return f"{text} on {last_issue_date}"
    if (
        time_scope == "date_range"
        and last_start_date
        and last_end_date
        and last_start_date not in text
        and last_end_date not in text
    ):
        return f"{text} between {last_start_date} and {last_end_date}"
    if time_scope == "all_time" and "archive" not in lowered and "latest" not in lowered:
        return f"{text} in the archive"
    return text


def _normalize_latest_phrasing(text: str) -> str:
    normalized = " ".join(text.split())
    normalized = normalized.replace("latest news or update about", "latest news about")
    normalized = normalized.replace("Latest news or update about", "Latest news about")
    normalized = normalized.replace("latest update or news about", "latest news about")
    normalized = normalized.replace("Latest update or news about", "Latest news about")
    normalized = normalized.replace("latest news on", "latest news about")
    normalized = normalized.replace("Latest news on", "Latest news about")
    normalized = normalized.replace("latest information about", "latest news about")
    normalized = normalized.replace("Latest information about", "Latest news about")
    return normalized
