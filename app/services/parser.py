from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Any


def _first_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Iterable) and not isinstance(value, (dict, bytes)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


@dataclass
class ParsedDoc:
    external_article_id: str
    external_doc_id: str | None
    headline: str | None
    deck: str | None
    label: str | None
    location: str | None
    body_text: str | None
    cleaned_body_text: str | None
    bylines: list[str]
    publication_id: str
    publication_name: str
    zone: str | None
    pagegroup: str | None
    layoutdesk: str | None
    pageno: int | None
    issue_name: str
    issue_date: datetime | None
    updated_at: datetime | None
    article_filename: str | None
    status: str | None
    raw_json: dict[str, Any]

    @property
    def is_searchable(self) -> bool:
        return bool(self.headline and self.cleaned_body_text)

    @property
    def embedding_text(self) -> str:
        parts = [self.headline, self.cleaned_body_text]
        return "\n".join(part for part in parts if part)

    @property
    def embedding_source_hash(self) -> str | None:
        text = self.embedding_text
        if not text:
            return None
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_body_text(headline: str | None, body_text: str | None) -> str | None:
    if not body_text:
        return None

    text = body_text.replace("|", " ")
    text = " ".join(text.split())

    junk_phrases = [
        "Complimentary with the times of india",
        "Price2.00",
        "PAGES4",
        "in advanced stages of scripting",
    ]
    for phrase in junk_phrases:
        text = text.replace(phrase, " ")

    lowered = text.lower()
    if "thumbnail" in lowered or "complimentary with the times of india" in lowered:
        return None

    if headline:
        headline_lower = headline.strip().lower()
        if headline_lower and lowered == headline_lower:
            return None

    cleaned = " ".join(text.split()).strip()
    if len(cleaned) < 120:
        return None
    return cleaned


def parse_doc(doc: dict[str, Any]) -> ParsedDoc:
    headline = _first_str(doc.get("articleheadline"))
    body_text = _first_str(doc.get("CONTENT"))
    issue_date = doc.get("issue_date")
    updated_date = doc.get("updated_date")
    return ParsedDoc(
        external_article_id=doc["article_id"],
        external_doc_id=str(doc.get("id")) if doc.get("id") is not None else None,
        headline=headline,
        deck=_first_str(doc.get("articleheaddeck")),
        label=_first_str(doc.get("articleheadlabel")),
        location=_first_str(doc.get("articlelocation")),
        body_text=body_text,
        cleaned_body_text=clean_body_text(headline, body_text),
        bylines=[
            item.strip()
            for item in (doc.get("articlebyline") or [])
            if isinstance(item, str) and item.strip()
        ],
        publication_id=doc["publication_id"],
        publication_name=doc["publication_name"],
        zone=doc.get("zone"),
        pagegroup=doc.get("pagegroup"),
        layoutdesk=doc.get("layoutdesk"),
        pageno=int(doc["pageno"]) if str(doc.get("pageno", "")).isdigit() else None,
        issue_name=doc["issue_name"],
        issue_date=datetime.fromisoformat(issue_date.replace("Z", "+00:00"))
        if issue_date
        else None,
        updated_at=datetime.fromisoformat(updated_date.replace("Z", "+00:00"))
        if updated_date
        else None,
        article_filename=doc.get("articlefilename"),
        status=str(doc.get("status")) if doc.get("status") is not None else None,
        raw_json=doc,
    )
