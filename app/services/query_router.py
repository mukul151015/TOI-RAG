from datetime import datetime
from difflib import get_close_matches
import re
from typing import Any

from app.schemas import RoutedQuery
from app.services.repository import fetch_publication_catalog, fetch_section_catalog


SEMANTIC_CUES = [
    "about",
    "related to",
    "discuss",
    "covered",
    "cover",
    "war",
    "impact",
    "tensions",
    "conflict",
    "win",
    "winning",
    "victory",
    "champion",
    "champions",
    "world cup",
    "t20",
    "t-20",
    "budget",
]
STRUCTURED_CUES = [
    "show me",
    "list",
    "which sections",
    "published",
    "edition",
    "section",
    "articles from",
]
BROAD_LIST_CUES = [
    "show me all",
    "list all",
    "all articles",
    "which sections had the most articles",
]
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
SUFFIX_STRIPS = ("city", "upcountry", "main", "north", "south", "east", "west", "early")
TOKEN_CORRECTIONS = {
    "sporst": "sports",
    "sportss": "sports",
    "sporst": "sports",
    "vctory": "victory",
    "victoy": "victory",
    "worl": "world",
    "wrld": "world",
    "chapmions": "champions",
    "geopoltical": "geopolitical",
}


def route_query(query: str, issue_date: str | None = None) -> RoutedQuery:
    normalized_query = normalize_user_query(query)
    lowered = normalized_query.lower()
    edition = _extract_edition(lowered)
    section = _extract_section(lowered)
    has_semantic = any(phrase in lowered for phrase in SEMANTIC_CUES)
    has_structured = bool(edition or section) or any(
        phrase in lowered for phrase in STRUCTURED_CUES
    )

    if has_structured and has_semantic:
        mode = "hybrid"
    elif has_semantic:
        mode = "semantic"
    else:
        mode = "sql"

    if "which sections had the most articles" in lowered:
        mode = "sql"

    semantic_query = _build_semantic_query(normalized_query, edition, section) if mode in {"semantic", "hybrid"} else None
    return RoutedQuery(
        mode=mode,
        issue_date=issue_date or _extract_date(lowered),
        edition=edition,
        section=section,
        semantic_query=semantic_query,
    )


def is_section_count_query(query: str) -> bool:
    return "which sections had the most articles" in query.lower()


def is_broad_listing_query(query: str) -> bool:
    lowered = query.lower()
    return any(phrase in lowered for phrase in BROAD_LIST_CUES)


def normalize_user_query(query: str) -> str:
    lowered = query.lower()
    lowered = re.sub(r"\bt[\s-]?20\b", "t20", lowered)
    tokens = re.findall(r"[a-z0-9]+|[^a-z0-9\s]+", lowered)
    vocabulary = {
        "sports",
        "sport",
        "section",
        "edition",
        "world",
        "cup",
        "t20",
        "victory",
        "win",
        "winning",
        "champion",
        "champions",
        "budget",
        "middle",
        "class",
        "iran",
        "conflict",
        "geopolitical",
        "editorial",
        "opinion",
        "mumbai",
        "delhi",
    }
    corrected: list[str] = []
    for token in tokens:
        if not re.fullmatch(r"[a-z0-9]+", token):
            corrected.append(token)
            continue
        if token in TOKEN_CORRECTIONS:
            corrected.append(TOKEN_CORRECTIONS[token])
            continue
        if len(token) >= 5:
            match = get_close_matches(token, vocabulary, n=1, cutoff=0.85)
            corrected.append(match[0] if match else token)
        else:
            corrected.append(token)
    text = "".join(
        f" {token}" if index and re.fullmatch(r"[a-z0-9]+", token) and re.fullmatch(r"[a-z0-9]+", corrected[index - 1]) else token
        for index, token in enumerate(corrected)
    )
    return re.sub(r"\s+", " ", text).strip()


def expand_semantic_queries(base_query: str) -> list[str]:
    variants = [base_query]
    lowered = base_query.lower()
    if "india" in lowered and "world cup" in lowered:
        variants.append("india world cup win")
        variants.append("world champions india")
    if "t20" in lowered and "world cup" in lowered:
        variants.append("t20 world cup india win")
        variants.append("bcci reward world champions")
    if "victory" in lowered:
        variants.append(base_query.replace("victory", "win"))
    if "covered" in lowered:
        variants.append(base_query.replace("covered", "reported on"))
    deduped: list[str] = []
    seen: set[str] = set()
    for value in variants:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def _extract_edition(text: str) -> str | None:
    alias_map = _build_publication_alias_map()
    matches: list[tuple[int, str, set[str]]] = []
    for alias, publication_names in alias_map.items():
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, text):
            matches.append((len(alias), alias, publication_names))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    _, alias, publication_names = matches[0]
    if len(publication_names) == 1:
        return next(iter(publication_names))
    return alias.title()


def _extract_section(text: str) -> str | None:
    section_catalog = fetch_section_catalog()
    normalized_map = {
        _normalize(section): section
        for section in section_catalog
        if section
    }
    for alias in ["sports", "sport", "city", "nation", "editorial", "opinion", "business", "world", "entertainment", "frontpage"]:
        if re.search(rf"\b{re.escape(alias)}\b", text):
            if alias == "sport":
                return normalized_map.get("sports", "Sports")
            if alias == "opinion":
                return normalized_map.get("editorial", "Editorial")
            return normalized_map.get(alias, alias.title())
    for normalized, original in normalized_map.items():
        if re.search(rf"\b{re.escape(normalized)}\b", _normalize(text)):
            return original
    return None


def _extract_date(text: str) -> str | None:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
    if match:
        return match.group(1)
    textual = re.search(
        r"\b(" + "|".join(MONTHS.keys()) + r")\s+(\d{1,2})(?:,\s*(20\d{2}))?\b",
        text,
    )
    if textual:
        month = MONTHS[textual.group(1)]
        day = int(textual.group(2))
        year = int(textual.group(3)) if textual.group(3) else datetime.now().year
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _build_semantic_query(query: str, edition: str | None, section: str | None) -> str:
    semantic = query
    if edition:
        semantic = re.sub(rf"(?i)\bpublished in the {re.escape(_edition_core_name(edition))} edition\b", "", semantic)
        semantic = re.sub(rf"(?i)\bin the {re.escape(_edition_core_name(edition))} edition\b", "", semantic)
        semantic = re.sub(rf"(?i)\bfrom the {re.escape(_edition_core_name(edition))} edition\b", "", semantic)
    if section:
        semantic = re.sub(rf"(?i)\bfrom the {re.escape(section)} section\b", "", semantic)
        semantic = re.sub(rf"(?i)\b{re.escape(section)} section articles\b", "", semantic)
        semantic = re.sub(rf"(?i)\bfrom {re.escape(section)} section\b", "", semantic)
        semantic = re.sub(rf"(?i)\bwhich {re.escape(section)} section\b", "", semantic)
    semantic = re.sub(r"(?i)\bwhich sport section\b", "", semantic)
    semantic = re.sub(r"(?i)\bwhich sports section\b", "", semantic)
    semantic = re.sub(r"(?i)\b(show me|list all articles|which stories|find articles|articles)\b", "", semantic)
    semantic = re.sub(r"\s+", " ", semantic).strip(" ,.-")
    semantic = _expand_semantic_query(semantic or query)
    return semantic


def _expand_semantic_query(query: str) -> str:
    expanded = re.sub(r"(?i)\bt[\s-]?20\b", "t20", query)
    lowered = expanded.lower()
    additions: list[str] = []
    if "victory" in lowered:
        additions.append("win")
    if "champion" in lowered or "champions" in lowered:
        additions.append("world champions")
    if "world cup" in lowered and "india" in lowered:
        additions.append("india world cup win")
    if "t20" in lowered and "world cup" in lowered:
        additions.append("t20 world cup")
    if "covered" in lowered and "victory" in lowered:
        additions.append("coverage")
    if additions:
        expanded = f"{expanded} {' '.join(additions)}"
    return re.sub(r"\s+", " ", expanded).strip()


def _build_publication_alias_map() -> dict[str, set[str]]:
    alias_map: dict[str, set[str]] = {}
    for row in fetch_publication_catalog():
        publication_id = row["id"]
        publication_name = row["publication_name"]
        for alias in _publication_aliases(publication_id, publication_name):
            alias_map.setdefault(alias, set()).add(publication_name)
    return alias_map


def _publication_aliases(publication_id: str, publication_name: str) -> set[str]:
    aliases: set[str] = set()
    suffix = publication_id.split("_", 1)[1] if "_" in publication_id else publication_id
    aliases.add(_normalize(_strip_suffixes(suffix)))
    aliases.add(_normalize(suffix))
    family_alias = _publication_family_alias(publication_id, publication_name)
    if family_alias:
        aliases.add(family_alias)

    if " - " in publication_name:
        right = publication_name.split(" - ", 1)[1]
        right = right.replace("_Digital", "")
        normalized_right = _normalize(right)
        aliases.add(normalized_right)
        aliases.add(_normalize(_strip_suffixes(normalized_right)))

    cleaned = {alias for alias in aliases if alias}
    return cleaned


def _strip_suffixes(value: str) -> str:
    normalized = _normalize(value)
    for suffix in SUFFIX_STRIPS:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _edition_core_name(publication_name: str) -> str:
    if " - " in publication_name:
        right = publication_name.split(" - ", 1)[1].replace("_Digital", "")
        return re.sub(r"(?<!^)([A-Z])", r" \1", right).strip()
    return publication_name


def _publication_family_alias(publication_id: str, publication_name: str) -> str | None:
    left = publication_name.split(" - ", 1)[0] if " - " in publication_name else publication_id.split("_", 1)[0]
    normalized = _normalize(left)
    normalized = re.sub(r"^toi", "", normalized)
    normalized = re.sub(r"(h?bs)$", "", normalized)
    return normalized or None
