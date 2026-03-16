import re

from app.schemas import RoutedQuery
from app.services.query_analyzer import analyze_query, expand_semantic_queries, normalize_user_query


BROAD_LIST_CUES = [
    "show me all",
    "show me articles",
    "show me stories",
    "show me pieces",
    "show me opinion pieces",
    "which stories",
    "what stories",
    "which opinion pieces",
    "what opinion pieces",
    "list all",
    "all articles",
    "which sections had the most articles",
]


def route_query(query: str, issue_date: str | None = None) -> RoutedQuery:
    return analyze_query(query, issue_date).routed


def is_section_count_query(query: str) -> bool:
    lowered = query.lower()
    return (
        "which sections had the most articles" in lowered
        or "which section had the least articles" in lowered
        or "show me the full ranking" in lowered
    )


def is_broad_listing_query(query: str) -> bool:
    lowered = query.lower()
    if any(phrase in lowered for phrase in BROAD_LIST_CUES):
        return True
    patterns = [
        r"\bshow me\b.*\barticles?\b",
        r"\bshow me\b.*\bstories\b",
        r"\bshow me\b.*\bpieces\b",
        r"\bwhich stories\b",
        r"\bwhat stories\b",
        r"\bwhich opinion pieces\b",
        r"\bwhat opinion pieces\b",
        r"\blist\b.*\barticles?\b",
        r"\blist\b.*\bstories\b",
        r"\bgive me\b.*\barticles?\b",
        r"\bgive me\b.*\bstories\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)
