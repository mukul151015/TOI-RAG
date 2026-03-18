import { query } from "./db.ts";
import type { QueryAnalysis, QueryIntent, QueryMode, RoutedQuery } from "./types.ts";

const SEMANTIC_CUES = [
  "about",
  "around",
  "related to",
  "discuss",
  "covered",
  "cover",
  "context",
  "mentioned",
  "war",
  "impact",
  "conflict",
  "world cup",
  "t20",
  "champion",
  "budget",
  "inflation",
  "prices",
  "growth",
];

const STRUCTURED_CUES = [
  "show me",
  "list",
  "edition",
  "section",
  "articles from",
  "front page",
  "frontpage",
];

const MONTHS: Record<string, number> = {
  january: 1,
  february: 2,
  march: 3,
  april: 4,
  may: 5,
  june: 6,
  july: 7,
  august: 8,
  september: 9,
  october: 10,
  november: 11,
  december: 12,
};

const TOKEN_CORRECTIONS: Record<string, string> = {
  sporst: "sports",
  sportss: "sports",
  abput: "about",
  aboput: "about",
  vctory: "victory",
  wrld: "world",
  chapmions: "champions",
  geopoltical: "geopolitical",
};

const PERSON_ALIAS_MAP: Record<string, { canonical: string; terms: string[] }> = {
  modi: {
    canonical: "Narendra Modi",
    terms: ["narendra modi", "pm modi", "prime minister modi", "modi"],
  },
  "narendra modi": {
    canonical: "Narendra Modi",
    terms: ["narendra modi", "pm modi", "prime minister modi", "modi"],
  },
  "rahul gandhi": {
    canonical: "Rahul Gandhi",
    terms: ["rahul gandhi"],
  },
};

let publicationCache: Array<{ id: string; publication_name: string }> | null = null;
let sectionCache: string[] | null = null;

function normalizeKey(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function stripSuffixes(value: string): string {
  let out = normalizeKey(value);
  for (const suffix of ["city", "upcountry", "main", "north", "south", "east", "west", "early"]) {
    if (out.endsWith(suffix) && out.length > suffix.length) {
      out = out.slice(0, -suffix.length);
      break;
    }
  }
  return out;
}

function similarity(a: string, b: string): number {
  if (a === b) {
    return 1;
  }
  const longer = a.length >= b.length ? a : b;
  const shorter = a.length >= b.length ? b : a;
  if (!longer.length) {
    return 1;
  }
  let same = 0;
  for (const char of shorter) {
    if (longer.includes(char)) {
      same += 1;
    }
  }
  return same / longer.length;
}

function closestWord(token: string, vocabulary: string[]): string {
  let best = token;
  let bestScore = 0;
  for (const candidate of vocabulary) {
    const score = similarity(token, candidate);
    if (score > bestScore) {
      bestScore = score;
      best = candidate;
    }
  }
  return bestScore >= 0.85 ? best : token;
}

async function getPublicationCatalog(): Promise<Array<{ id: string; publication_name: string }>> {
  if (publicationCache) {
    return publicationCache;
  }
  publicationCache = await query("publications", "select=id,publication_name&order=publication_name");
  return publicationCache;
}

async function getSectionCatalog(): Promise<string[]> {
  if (sectionCache) {
    return sectionCache;
  }
  const rows = await query<{ normalized_section: string | null }>(
    "sections",
    "select=normalized_section&normalized_section=not.is.null&order=normalized_section",
  );
  sectionCache = Array.from(
    new Set(rows.map((row) => row.normalized_section).filter((value): value is string => Boolean(value))),
  );
  return sectionCache;
}

function detectIntent(text: string, author: string | null): QueryIntent {
  if (author) {
    return /\bhow many\b|\bcount\b|\bnumber of\b/.test(text) ? "author_count" : "author_lookup";
  }
  if (/\bhow many\b|\bcount\b|\bnumber of\b/.test(text)) {
    if (/\babout\b|\baround\b|\bregarding\b|\bname appeared\b/.test(text)) {
      return "topic_count";
    }
    if (/\barticle/.test(text)) {
      return "article_count";
    }
    return "fact_lookup";
  }
  return "lookup";
}

function extractDate(text: string, fallbackIssueDate: string | null): string | null {
  const iso = text.match(/\b(20\d{2}-\d{2}-\d{2})\b/);
  if (iso) {
    return iso[1];
  }
  const textual = text.match(
    new RegExp(`\\b(${Object.keys(MONTHS).join("|")})\\s+(\\d{1,2})(?:,\\s*(20\\d{2}))?\\b`, "i"),
  );
  if (!textual) {
    return fallbackIssueDate;
  }
  const month = MONTHS[textual[1].toLowerCase()];
  const day = Number(textual[2]);
  const year = textual[3] ? Number(textual[3]) : new Date().getUTCFullYear();
  return `${year.toString().padStart(4, "0")}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function cleanTopicPhrase(value: string): string {
  let cleaned = value.toLowerCase().trim().replace(/[,.?]+$/g, "");
  cleaned = cleaned.replace(/\s+and\s+(?:in (?:what|which) context(?: they are)?|(?:what|which) context(?: they are)?|what they (?:were|are)? about.*)$/g, "");
  cleaned = cleaned.replace(/\bin (?:what|which) context(?: they are)?\b.*$/g, "");
  cleaned = cleaned.replace(/\b(?:what|which) context(?: they are)?\b.*$/g, "");
  cleaned = cleaned.replace(/\bwhat they (?:were|are)? about\b.*$/g, "");
  return cleaned.trim();
}

function extractTopic(lowered: string): string | null {
  const patterns = [
    /\bhow many articles? about\s+(.+)/,
    /\bhow many articles? around\s+(.+)/,
    /\bhow many articles? regarding\s+(.+)/,
    /\b(?:news|articles|stories)\s+about\s+(.+)/,
  ];
  for (const pattern of patterns) {
    const match = lowered.match(pattern);
    if (match) {
      const topic = cleanTopicPhrase(match[1]);
      if (topic) {
        return topic;
      }
    }
  }
  return null;
}

function expandSemanticBase(query: string): string {
  const lowered = query.toLowerCase();
  const additions: string[] = [];
  if (lowered.includes("victory")) {
    additions.push("win");
  }
  if (lowered.includes("world cup") && lowered.includes("india")) {
    additions.push("india world cup win");
  }
  if (lowered.includes("budget")) {
    additions.push("economy prices inflation business");
  }
  if (lowered.includes("lpg") || lowered.includes("png") || lowered.includes("cng")) {
    additions.push("gas allocation priority domestic supply");
  }
  return [query, ...additions].join(" ").replace(/\s+/g, " ").trim();
}

async function extractEdition(text: string): Promise<{ edition: string | null; ambiguousEdition: boolean }> {
  const catalog = await getPublicationCatalog();
  const aliasMap = new Map<string, Set<string>>();

  for (const row of catalog) {
    const right = row.publication_name.includes(" - ")
      ? row.publication_name.split(" - ", 2)[1].replace("_Digital", "")
      : row.publication_name;
    const aliases = new Set<string>([
      normalizeKey(right),
      stripSuffixes(right),
      normalizeKey(row.id.includes("_") ? row.id.split("_", 2)[1] : row.id),
      stripSuffixes(row.id.includes("_") ? row.id.split("_", 2)[1] : row.id),
    ]);
    for (const alias of aliases) {
      if (!alias) {
        continue;
      }
      if (!aliasMap.has(alias)) {
        aliasMap.set(alias, new Set());
      }
      aliasMap.get(alias)?.add(row.publication_name);
    }
  }

  const normalizedText = normalizeKey(text);
  let winner: { alias: string; names: string[] } | null = null;
  for (const [alias, names] of aliasMap.entries()) {
    if (!normalizedText.includes(alias)) {
      continue;
    }
    if (!winner || alias.length > winner.alias.length) {
      winner = { alias, names: Array.from(names).sort() };
    }
  }

  if (!winner) {
    return { edition: null, ambiguousEdition: false };
  }
  if (winner.names.length === 1) {
    return { edition: winner.names[0], ambiguousEdition: false };
  }
  if (winner.alias === "delhi") {
    return { edition: "Delhi", ambiguousEdition: true };
  }
  return { edition: winner.names[0], ambiguousEdition: true };
}

async function extractSection(text: string): Promise<string | null> {
  const catalog = await getSectionCatalog();
  const normalizedMap = new Map(catalog.map((value) => [normalizeKey(value), value]));
  const aliases: Record<string, string> = {
    sports: "Sports",
    sport: "Sports",
    opinion: "Edit",
    editorial: "Edit",
    edit: "Edit",
    "frontpage": "FrontPage",
    "front page": "FrontPage",
    business: "Business",
    world: "World",
    city: "City",
  };

  for (const [alias, mapped] of Object.entries(aliases)) {
    if (new RegExp(`\\b${alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(text)) {
      return normalizedMap.get(normalizeKey(mapped)) ?? mapped;
    }
  }

  const normalizedText = normalizeKey(text);
  for (const [normalized, original] of normalizedMap.entries()) {
    if (normalized && normalizedText.includes(normalized)) {
      return original;
    }
  }
  return null;
}

function extractAuthor(text: string): string | null {
  const byMatch = text.match(/\b(?:by|author)\s+([a-z][a-z\s'.-]{3,80})/i);
  return byMatch ? byMatch[1].trim().replace(/\b\w/g, (value) => value.toUpperCase()) : null;
}

function extractPeople(query: string, author: string | null): string[] {
  const people = new Set<string>();
  if (author) {
    people.add(author);
  }

  for (const [alias, meta] of Object.entries(PERSON_ALIAS_MAP)) {
    if (new RegExp(`\\b${alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(query)) {
      people.add(meta.canonical);
    }
  }

  return Array.from(people);
}

function expandSemanticQueries(baseQuery: string): string[] {
  const variants = [baseQuery];
  const lowered = baseQuery.toLowerCase();
  if (lowered.includes("india") && lowered.includes("world cup")) {
    variants.push("india world cup win");
    variants.push("world champions india");
  }
  if (lowered.includes("victory")) {
    variants.push(baseQuery.replace(/victory/gi, "win"));
  }
  if (lowered.includes("covered")) {
    variants.push(baseQuery.replace(/covered/gi, "reported on"));
  }
  return Array.from(new Set(variants.map((value) => value.replace(/\s+/g, " ").trim()).filter(Boolean)));
}

export function canonicalPersonName(name: string): string {
  return PERSON_ALIAS_MAP[name.toLowerCase()]?.canonical ?? name;
}

export function expandPersonAliasTerms(name: string): string[] {
  return PERSON_ALIAS_MAP[name.toLowerCase()]?.terms ?? [name];
}

export function isSectionCountQuery(query: string): boolean {
  const lowered = query.toLowerCase();
  return (
    lowered.includes("which sections had the most articles") ||
    lowered.includes("which section had the least articles") ||
    lowered.includes("show me the full ranking")
  );
}

export function isBroadListingQuery(query: string): boolean {
  const lowered = query.toLowerCase();
  return [
    "show me all",
    "show me articles",
    "show me stories",
    "which stories",
    "what stories",
    "list all",
    "give me all articles",
  ].some((phrase) => lowered.includes(phrase));
}

export function normalizeUserQuery(query: string): string {
  const vocabulary = [
    "sports",
    "section",
    "edition",
    "world",
    "cup",
    "t20",
    "victory",
    "budget",
    "middle",
    "class",
    "iran",
    "conflict",
    "editorial",
    "opinion",
  ];
  const tokens = query.toLowerCase().match(/[a-z0-9]+|[^a-z0-9\s]+/g) ?? [];
  const corrected = tokens.map((token) => {
    if (!/^[a-z0-9]+$/.test(token)) {
      return token;
    }
    if (TOKEN_CORRECTIONS[token]) {
      return TOKEN_CORRECTIONS[token];
    }
    return token.length >= 5 ? closestWord(token, vocabulary) : token;
  });
  return corrected.join(" ").replace(/\s+([,.!?])/g, "$1").replace(/\s+/g, " ").trim();
}

export async function analyzeQuery(query: string, issueDate: string | null = null): Promise<QueryAnalysis> {
  const normalizedQuery = normalizeUserQuery(query);
  const lowered = normalizedQuery.toLowerCase();
  const shouldExtractEdition =
    /\bedition\b|\bpublished in\b|\bfront page\b|\bfrontpage\b|\barticles from\b/.test(lowered);
  const editionResult = shouldExtractEdition
    ? await extractEdition(lowered)
    : { edition: null, ambiguousEdition: false };
  let section = await extractSection(lowered);
  const author = extractAuthor(normalizedQuery);

  if (!section && /\bworld cup\b|\bt20\b|\bbcci\b|\bcricket\b|\bmatch\b/.test(lowered)) {
    section = "Sports";
  }
  if (!section && /\bbudget\b|\binflation\b|\bgrowth\b|\btax\b/.test(lowered)) {
    section = "Business";
  }

  const intent = detectIntent(lowered, author);
  const hasSemantic = SEMANTIC_CUES.some((cue) => lowered.includes(cue)) || intent === "topic_count" || intent === "fact_lookup";
  const hasStructured = Boolean(editionResult.edition || section || author || STRUCTURED_CUES.some((cue) => lowered.includes(cue)));

  let mode: QueryMode = "sql";
  if (hasSemantic && hasStructured) {
    mode = "hybrid";
  } else if (hasSemantic) {
    mode = "semantic";
  }

  const topic = extractTopic(lowered);
  const semanticQuery = mode === "semantic" || mode === "hybrid"
    ? expandSemanticBase(topic || normalizedQuery)
    : null;

  const routed: RoutedQuery = {
    mode,
    intent,
    issue_date: extractDate(lowered, issueDate),
    edition: editionResult.edition,
    section,
    author,
    semantic_query: semanticQuery,
  };

  const people = extractPeople(query, author);
  const organizations = ["bcci", "congress", "bjp", "dmk", "supreme court"].filter((term) =>
    new RegExp(`\\b${term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(lowered)
  );

  return {
    raw_query: query,
    normalized_query: normalizedQuery,
    lowered_query: lowered,
    routed,
    ambiguous_edition: editionResult.ambiguousEdition,
    entities: {
      authors: author ? [author] : [],
      editions: routed.edition ? [routed.edition] : [],
      sections: routed.section ? [routed.section] : [],
      people,
      content_people: people,
      organizations,
      content_organizations: organizations,
      topics: semanticQuery ? [semanticQuery] : [],
      llm_paraphrases: [],
    },
  };
}

export { expandSemanticQueries };
