from datetime import datetime
from difflib import get_close_matches
from functools import lru_cache
import json
import logging
import re
from difflib import SequenceMatcher

from pydantic import BaseModel

from app.schemas import RoutedQuery
from app.services.repository import fetch_author_catalog, fetch_publication_catalog, fetch_section_catalog

logger = logging.getLogger(__name__)


SEMANTIC_CUES = [
    "about",
    "around",
    "related to",
    "discuss",
    "covered",
    "cover",
    "context",
    "appeared",
    "mentioned",
    "name appeared",
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
    "middle class",
    "inflation",
    "prices",
    "growth",
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
    "abput": "about",
    "aboput": "about",
    "vctory": "victory",
    "victoy": "victory",
    "worl": "world",
    "wrld": "world",
    "chapmions": "champions",
    "geopoltical": "geopolitical",
    "brijing": "beijing",
}
PERSON_ALIAS_MAP = {
    "modi": {
        "canonical": "Narendra Modi",
        "terms": ["narendra modi", "pm modi", "prime minister modi", "modi"],
    },
    "narendra modi": {
        "canonical": "Narendra Modi",
        "terms": ["narendra modi", "pm modi", "prime minister modi", "modi"],
    },
    "rahul gandhi": {
        "canonical": "Rahul Gandhi",
        "terms": ["rahul gandhi"],
    },
}


class QueryAnalysis(BaseModel):
    raw_query: str
    normalized_query: str
    lowered_query: str
    routed: RoutedQuery
    entities: dict[str, list[str]]
    ambiguous_edition: bool = False


KNOWN_ORGANIZATIONS = [
    "bcci",
    "congress",
    "bjp",
    "dmk",
    "ed",
    "supreme court",
    "high court",
    "toi",
    "times of india",
    "government",
    "govt",
]
KNOWN_PLACES = [
    "delhi",
    "mumbai",
    "kolkata",
    "chennai",
    "bangalore",
    "bengaluru",
    "hyderabad",
    "lucknow",
    "nagpur",
    "ludhiana",
    "agra",
    "bareilly",
    "dehradun",
    "iran",
    "israel",
    "beijing",
    "china",
    "saudi arabia",
    "bahrain",
]


def analyze_query(query: str, issue_date: str | None = None) -> QueryAnalysis:
    normalized_query = normalize_user_query(query)
    lowered = normalized_query.lower()
    edition = _extract_edition(lowered) if _should_extract_edition(lowered) else None
    ambiguous_edition = False
    if re.search(r"\bdelhi edition\b", lowered):
        edition = "Delhi"
        ambiguous_edition = True
    author = _extract_author(lowered)
    section = _extract_section(lowered)
    if not section and _looks_like_sports_intent(lowered):
        section = "Sports"
    if not section and _looks_like_china_editorial_intent(lowered):
        section = "Edit"
    if not section and _looks_like_business_intent(lowered):
        section = "Business"
    intent = _detect_intent(lowered, author)
    has_semantic = any(phrase in lowered for phrase in SEMANTIC_CUES) or intent in {"topic_count", "fact_lookup"}
    has_structured = bool(edition or section or author) or any(phrase in lowered for phrase in STRUCTURED_CUES)
    mode = _select_mode(has_structured, has_semantic, author, lowered)
    semantic_query = _build_semantic_query(normalized_query, edition, section) if mode in {"semantic", "hybrid"} else None
    routed = RoutedQuery(
        mode=mode,
        intent=intent,
        issue_date=issue_date or _extract_date(lowered),
        edition=edition,
        section=section,
        author=author,
        semantic_query=semantic_query,
    )
    places = _extract_places(lowered, edition)
    organizations = _extract_organizations(lowered)
    people = _extract_people(query, author)

    # LLM-based entity enrichment (runs only when flag is enabled).
    llm_meta: dict = {}
    try:
        llm_meta = llm_analyze_query(query)
    except Exception:
        pass

    if llm_meta:
        # Merge: rule-based wins for known entities (no regression).
        for llm_person in llm_meta.get("persons", []):
            if llm_person and llm_person not in people:
                people.append(llm_person)
        for llm_org in llm_meta.get("organizations", []):
            if llm_org and llm_org.lower() not in {o.lower() for o in organizations}:
                organizations.append(llm_org)
        for llm_place in llm_meta.get("places", []):
            if llm_place and llm_place.lower() not in {p.lower() for p in places}:
                places.append(llm_place)

    entity_roles = _resolve_entity_roles(
        lowered,
        edition=edition,
        section=section,
        author=author,
        people=people,
        places=places,
        organizations=organizations,
    )

    # Incorporate LLM-generated query paraphrases into topics for query expansion.
    llm_paraphrases: list[str] = llm_meta.get("query_paraphrases", []) if llm_meta else []

    entities = {
        "authors": [author] if author else [],
        "editions": [edition] if edition else [],
        "sections": [section] if section else [],
        "people": people,
        "places": places,
        "organizations": organizations,
        **entity_roles,
        "topics": [semantic_query] if semantic_query else [],
        "llm_paraphrases": llm_paraphrases,
    }
    return QueryAnalysis(
        raw_query=query,
        normalized_query=normalized_query,
        lowered_query=lowered,
        routed=routed,
        entities=entities,
        ambiguous_edition=ambiguous_edition,
    )


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
        "ludhiana",
        "author",
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
    """Expand base query with rule-based variants.

    When ``llm_query_analysis_enabled`` is True, this function is also called
    with paraphrases generated by :func:`llm_analyze_query` so we simply
    deduplicate across all variants.
    """
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


@lru_cache(maxsize=256)
def llm_analyze_query(query: str) -> dict:
    """Call the LLM to extract structured query metadata and paraphrases.

    Returns a dict with keys:
        persons, organizations, places, topics, intent, date_mentions,
        query_paraphrases

    Falls back to an empty dict on any failure so callers can safely ignore it.
    """
    try:
        from app.core.config import get_settings  # late import to avoid circular
        from app.services.openai_client import chat_completion  # late import

        settings = get_settings()
        if not settings.llm_query_analysis_enabled:
            return {}

        system = (
            "You are a query understanding assistant for a news search system. "
            "Given a user query, return JSON with these exact keys:\n"
            '  "persons": list of person names mentioned\n'
            '  "organizations": list of organization names mentioned\n'
            '  "places": list of place names mentioned\n'
            '  "topics": list of main topics (e.g. budget, cricket, geopolitics)\n'
            '  "intent": one of lookup|article_count|topic_count|fact_lookup|author_lookup|author_count\n'
            '  "date_mentions": list of date strings mentioned\n'
            '  "query_paraphrases": exactly 2 alternative phrasings of the query\n'
            "Return only valid JSON, no explanation."
        )
        raw = chat_completion(system, query, model=settings.openai_chat_model, timeout=15.0)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
        result = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        logger.debug("LLM query analysis failed (falling back to rule-based): %s", exc)
        return {}


def _select_mode(has_structured: bool, has_semantic: bool, author: str | None, lowered: str) -> str:
    if author:
        return "sql"
    if has_structured and has_semantic:
        return "hybrid"
    if has_semantic:
        return "semantic"
    if "which sections had the most articles" in lowered:
        return "sql"
    return "sql"


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
    if alias == "delhi":
        return "Delhi"
    canonical = _main_publication_for_family(alias, publication_names)
    if canonical:
        return canonical
    return alias.title()


def _should_extract_edition(text: str) -> bool:
    patterns = [
        r"\bedition\b",
        r"\bpublished in\b",
        r"\bfrom the\b.*\bedition\b",
        r"\bin the\b.*\bedition\b",
        r"\barticles? in\b.*\bedition\b",
        # city-name patterns that imply edition filtering even without "edition" keyword
        r"\b(mumbai|delhi|kolkata|chennai|bangalore|bengaluru|hyderabad|lucknow|nagpur|ludhiana|agra|bareilly|dehradun)\b.*(front page|frontpage|section|articles?|stories|news)",
        r"(front page|frontpage).*(mumbai|delhi|kolkata|chennai|bangalore|bengaluru|hyderabad|lucknow|nagpur|ludhiana)",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _extract_section(text: str) -> str | None:
    section_catalog = fetch_section_catalog()
    normalized_map = {_normalize(section): section for section in section_catalog if section}
    alias_map = {
        "sports": "sports",
        "sport": "sports",
        "city": "city",
        "nation": "nation",
        "editorial": "edit",
        "opinion": "edit",
        "edit": "edit",
        "oped": "oped",
        "op-ed": "oped",
        "business": "business",
        "world": "world",
        "entertainment": "entertainment",
        "frontpage": "frontpage",
        "front page": "frontpage",
    }
    for alias, normalized_alias in alias_map.items():
        if normalized_alias == "world" and re.search(r"\bworld cup\b", text):
            continue
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return normalized_map.get(normalized_alias, normalized_alias.title())
    for normalized, original in normalized_map.items():
        if re.search(rf"\b{re.escape(normalized)}\b", _normalize(text)):
            return original
    return None


def _extract_date(text: str) -> str | None:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
    if match:
        return match.group(1)
    textual = re.search(r"\b(" + "|".join(MONTHS.keys()) + r")\s+(\d{1,2})(?:,\s*(20\d{2}))?\b", text)
    if textual:
        month = MONTHS[textual.group(1)]
        day = int(textual.group(2))
        year = int(textual.group(3)) if textual.group(3) else datetime.now().year
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _build_semantic_query(query: str, edition: str | None, section: str | None) -> str:
    focused_topic = _extract_focus_topic(query)
    if focused_topic:
        return _expand_semantic_query(focused_topic)
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
    return _expand_semantic_query(semantic or query)


def _extract_focus_topic(query: str) -> str | None:
    lowered = query.lower().strip(" ?.")
    patterns = [
        r"\bhow many article around\s+(.+)",
        r"\bhow many article about\s+(.+)",
        r"\bhow many article regarding\s+(.+)",
        r"\bhow many articles around\s+(.+)",
        r"\bhow many articles about\s+(.+)",
        r"\bhow many articles regarding\s+(.+)",
        r"\bhow many times\s+(.+?)\s+(?:name\s+)?appeared\b",
        r"\bhow many article\s+(.+)",
        r"\bhow many articles\s+(.+)",
        r"\b(?:news|articles|stories)\s+about\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        topic = _clean_topic_phrase(match.group(1))
        if topic:
            return topic
    return None


def _clean_topic_phrase(value: str) -> str:
    cleaned = value.lower().strip(" ,.?")
    cleaned = re.sub(r"\band\s+and\b", " and", cleaned)
    cleaned = re.sub(
        r"\s+and\s+(?:in (?:what|which) context(?: they are)?|(?:what|which) context(?: they are)?|what they (?:were|are)? about.*)$",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\bin (?:what|which) context(?: they are)?\b.*$", "", cleaned)
    cleaned = re.sub(r"\b(?:what|which) context(?: they are)?\b.*$", "", cleaned)
    cleaned = re.sub(r"\bwhat they (?:were|are)? about\b.*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.?")
    return cleaned


def _extract_author(text: str) -> str | None:
    normalized_text = _normalize(text)
    matches: list[tuple[int, str]] = []
    for display_name in fetch_author_catalog():
        normalized_name = _normalize(display_name)
        if normalized_name and normalized_name in normalized_text:
            matches.append((len(normalized_name), display_name))
    if not matches:
        author_catalog = fetch_author_catalog()
        query_tokens = re.findall(r"[a-z]+", text.lower())
        windows: list[str] = []
        for size in (3, 2):
            for index in range(len(query_tokens) - size + 1):
                window = " ".join(query_tokens[index : index + size]).strip()
                if window:
                    windows.append(window)
        best_fuzzy: tuple[float, str] | None = None
        for display_name in author_catalog:
            lowered_name = display_name.lower()
            candidate_names = [lowered_name]
            candidate_names.extend(
                lowered_name.replace(".", "").split(" / ")
                if " / " in lowered_name
                else []
            )
            for candidate_name in candidate_names:
                close = get_close_matches(candidate_name, windows, n=1, cutoff=0.8)
                if not close:
                    continue
                ratio = SequenceMatcher(None, candidate_name, close[0]).ratio()
                if not best_fuzzy or ratio > best_fuzzy[0]:
                    best_fuzzy = (ratio, display_name)
        if best_fuzzy:
            return best_fuzzy[1]
    if not matches:
        by_match = re.search(r"\b(?:by|author)\s+([a-z][a-z\s'.-]{3,80})", text)
        if by_match:
            return by_match.group(1).strip(" ,.?").title()
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def _extract_people(query: str, author: str | None) -> list[str]:
    people: list[str] = []
    if author:
        people.append(author)
    capitalized = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", query)
    single_names = re.findall(r"\b([A-Z][a-z]{2,})\b", query)
    stop = {
        "Sports Section", "World Cup", "Editorial Section", "Middle Class",
        "Delhi Edition", "Mumbai Edition", "Iran Conflict",
        # Prevent "Find X", "Show X", "List X" from being treated as person names
        "Find World", "Find Articles", "Find Stories", "Find News",
        "Show Articles", "Show Stories", "Show News",
        "List Articles", "List Stories",
        # Section-related multi-word phrases
        "Edit Section", "Nation Section", "City Section", "World Section",
        "Business Section", "Front Page", "Feature Section",
    }
    for candidate in capitalized:
        if candidate in stop:
            continue
        candidate = _canonicalize_person(candidate)
        if candidate not in people:
            people.append(candidate)
    single_stop = {
        # Question / imperative verbs
        "How", "Show", "Which", "What", "Find", "List", "Give", "Tell",
        "Get", "Fetch", "Look", "Search", "Count", "Display",
        # Known section names (prevent section names from being people)
        "World", "Nation", "City", "Business", "Sports", "Edit", "Oped",
        "Feature", "Regional", "Advt", "FrontPage", "Editorial", "Section",
        "Articles", "Stories", "Article", "Story",
        # Known places already handled separately
        "Delhi", "Mumbai", "Ludhiana", "Iran", "India", "China", "Israel",
        "Bahrain", "Kolkata", "Chennai", "Bangalore", "Bengaluru",
        "Hyderabad", "Lucknow", "Nagpur", "Agra", "Bareilly", "Dehradun",
        # Other common non-person caps
        "March", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday", "January", "February", "April", "May",
        "June", "July", "August", "September", "October", "November", "December",
    }
    for candidate in single_names:
        if candidate in single_stop:
            continue
        candidate = _canonicalize_person(candidate)
        if candidate not in people:
            people.append(candidate)
    return people


def expand_person_alias_terms(name: str) -> list[str]:
    alias = PERSON_ALIAS_MAP.get(name.lower())
    if not alias:
        return [name]
    return list(alias["terms"])


def canonical_person_name(name: str) -> str:
    alias = PERSON_ALIAS_MAP.get(name.lower())
    if not alias:
        return name
    return str(alias["canonical"])


def _canonicalize_person(name: str) -> str:
    return canonical_person_name(name.strip())


def _extract_places(text: str, edition: str | None) -> list[str]:
    places: list[str] = []
    if edition:
        core_name = _edition_core_name(edition).lower()
        if core_name and core_name not in places:
            places.append(core_name)
    for place in KNOWN_PLACES:
        if re.search(rf"\b{re.escape(place)}\b", text) and place not in places:
            places.append(place)
    return places


def _extract_organizations(text: str) -> list[str]:
    organizations: list[str] = []
    for org in KNOWN_ORGANIZATIONS:
        if re.search(rf"\b{re.escape(org)}\b", text) and org not in organizations:
            organizations.append(org)
    return organizations


def _resolve_entity_roles(
    text: str,
    *,
    edition: str | None,
    section: str | None,
    author: str | None,
    people: list[str],
    places: list[str],
    organizations: list[str],
) -> dict[str, list[str]]:
    edition_filters: list[str] = []
    section_filters: list[str] = []
    author_filters: list[str] = []
    content_locations: list[str] = []
    content_people: list[str] = []
    content_organizations: list[str] = []

    if edition:
        edition_filters.append(edition)
    if section:
        section_filters.append(section)
    if author:
        author_filters.append(author)

    edition_context = _has_edition_context(text)
    for place in places:
        if edition_context and _place_used_as_filter(text, place):
            continue
        if place not in content_locations:
            content_locations.append(place)

    for person in people:
        if author and _normalize(person) == _normalize(author):
            continue
        if person not in content_people:
            content_people.append(person)

    for organization in organizations:
        if organization not in content_organizations:
            content_organizations.append(organization)

    return {
        "edition_filters": edition_filters,
        "section_filters": section_filters,
        "author_filters": author_filters,
        "content_locations": content_locations,
        "content_people": content_people,
        "content_organizations": content_organizations,
    }


def _has_edition_context(text: str) -> bool:
    return any(
        re.search(pattern, text)
        for pattern in [
            r"\bedition\b",
            r"\bpublished in\b",
            r"\bfrom the\b.*\bedition\b",
            r"\bin the\b.*\bedition\b",
            r"\barticles? in\b.*\bedition\b",
        ]
    )


def _place_used_as_filter(text: str, place: str) -> bool:
    patterns = [
        rf"\b{re.escape(place)}\s+edition\b",
        rf"\bin\s+{re.escape(place)}\s+edition\b",
        rf"\bpublished in\s+the\s+{re.escape(place)}\s+edition\b",
        rf"\bfrom\s+the\s+{re.escape(place)}\s+edition\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


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
    if "china" in lowered and "growth" in lowered:
        additions.extend(["world newest beijing problem gdp target ambitions"])
    if "lpg" in lowered or "png" in lowered or "cng" in lowered:
        additions.extend(["gas allocation priority domestic supply"])
    if "budget" in lowered and "middle class" in lowered:
        additions.extend(["inflation prices households tax growth economists"])
    if "budget" in lowered:
        additions.extend(["economy prices inflation business"])
    if additions:
        expanded = f"{expanded} {' '.join(additions)}"
    return re.sub(r"\s+", " ", expanded).strip()


def _looks_like_sports_intent(text: str) -> bool:
    return any(cue in text for cue in ["world cup", "t20", "bcci", "champions", "cricket", "football", "ipl", "match", "trophy"])


def _looks_like_business_intent(text: str) -> bool:
    return any(cue in text for cue in ["budget", "middle class", "inflation", "price rise", "prices", "economists", "growth", "tax", "fuel inflation", "oil prices"])


def _looks_like_china_editorial_intent(text: str) -> bool:
    return "china" in text and ("growth ambition" in text or "growth ambitions" in text or "gdp target" in text)


def _detect_intent(text: str, author: str | None) -> str:
    if author:
        if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", text):
            return "author_count"
        return "author_lookup"
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", text):
        if re.search(r"\bhow many times\b|\bname appeared\b|\barticle(?:s)?\s+(?:around|about|regarding|on)\b", text):
            return "topic_count"
        if re.search(r"\barticle(?:s)?\b", text):
            return "article_count"
        return "fact_lookup"
    return "lookup"


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
        right = publication_name.split(" - ", 1)[1].replace("_Digital", "")
        normalized_right = _normalize(right)
        aliases.add(normalized_right)
        aliases.add(_normalize(_strip_suffixes(normalized_right)))
        aliases.add(_normalize(re.sub(r"^(timesof|toi)", "", normalized_right)))
    return {alias for alias in aliases if alias}


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


def _main_publication_for_family(alias: str, publication_names: set[str]) -> str | None:
    ordered = sorted(publication_names)
    alias_norm = _normalize(alias)
    special_targets = {
        "mumbai": "mumbaicity",
        "kolkata": "kolkatacity",
        "chennai": "chennai",
        "bangalore": "bangalorecity",
        "hyderabad": "hyderabad",
        "lucknow": "lucknowcity",
        "nagpur": "nagpurcity",
    }
    target = special_targets.get(alias_norm)
    if target:
        for publication_name in ordered:
            right = publication_name.split(" - ", 1)[1] if " - " in publication_name else publication_name
            right_norm = _normalize(right.replace("_Digital", ""))
            if target in right_norm:
                return publication_name
    def score(publication_name: str) -> tuple[int, str]:
        right = publication_name.split(" - ", 1)[1] if " - " in publication_name else publication_name
        right_norm = _normalize(right.replace("_Digital", ""))
        if alias_norm and alias_norm in right_norm and "city" in right_norm:
            return (1, publication_name)
        if alias_norm and alias_norm in right_norm:
            return (2, publication_name)
        if "city" in right_norm:
            return (3, publication_name)
        return (4, publication_name)
    candidate = sorted(ordered, key=score)[0]
    candidate_right = candidate.split(" - ", 1)[1] if " - " in candidate else candidate
    candidate_norm = _normalize(candidate_right.replace("_Digital", ""))
    if alias_norm in candidate_norm:
        return candidate
    return None
