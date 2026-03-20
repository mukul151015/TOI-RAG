from __future__ import annotations

from datetime import date, datetime
import json
import re
import uuid


def make_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex[:12]}"


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def normalize_issue_date(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            output.append(cleaned)
    return output


def headline_key(value: str | None) -> str:
    return normalize_text(value)


def safe_load_json(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def first_number(text: str) -> int | None:
    match = re.search(r"\b(\d+)\b", text)
    if not match:
        return None
    return int(match.group(1))


def parse_calendar_date(text: str | None) -> str | None:
    value = (text or "").strip()
    if not value:
        return None
    iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", value)
    if iso:
        return iso.group(1)
    for fmt in ("%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None
