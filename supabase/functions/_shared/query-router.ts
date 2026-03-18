import { query } from "./db.ts";
import type { RoutedQuery } from "./types.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Cue lists
// ---------------------------------------------------------------------------
const SEMANTIC_CUES = [
  "about", "related to", "discuss", "covered", "cover", "war", "impact",
  "tensions", "conflict", "win", "winning", "victory", "champion", "champions",
  "world cup", "t20", "t-20", "budget", "middle class", "inflation", "prices", "growth",
];

const STRUCTURED_CUES = [
  "show me", "list", "which sections", "published", "edition", "section", "articles from",
];

const BROAD_LIST_CUES = [
  "show me all", "show me articles", "show me stories", "show me pieces",
  "show me opinion pieces", "which stories", "what stories",
  "which opinion pieces", "what opinion pieces", "list all", "all articles",
  "which sections had the most articles",
];

const MONTHS: Record<string, number> = {
  january: 1, february: 2, march: 3, april: 4, may: 5, june: 6,
  july: 7, august: 8, september: 9, october: 10, november: 11, december: 12,
};

const SUFFIX_STRIPS = ["city", "upcountry", "main", "north", "south", "east", "west", "early"];

const TOKEN_CORRECTIONS: Record<string, string> = {
  sporst: "sports", sportss: "sports", vctory: "victory", victoy: "victory",
  worl: "world", wrld: "world", chapmions: "champions", geopoltical: "geopolitical",
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

const KNOWN_ORGANIZATIONS = [
  "bcci", "congress", "bjp", "dmk", "ed", "supreme court", "high court",
  "toi", "times of india", "government", "govt",
];

const KNOWN_PLACES = [
  "delhi", "mumbai", "kolkata", "chennai", "bangalore", "bengaluru",
  "hyderabad", "lucknow", "nagpur", "ludhiana", "agra", "bareilly",
  "dehradun", "iran", "israel", "beijing", "china", "saudi arabia", "bahrain",
];

const VOCABULARY = new Set([
  "sports", "sport", "section", "edition", "world", "cup", "t20", "victory",
  "win", "winning", "champion", "champions", "budget", "middle", "class",
  "iran", "conflict", "geopolitical", "editorial", "opinion", "mumbai", "delhi",
]);

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
export async function routeQuery(
  query: string,
  issueDate: string | null = null,
): Promise<RoutedQuery> {
  const normalized = normalizeUserQuery(query);
  const lowered = normalized.toLowerCase();
  let edition = shouldExtractEdition(lowered) ? await extractEdition(lowered) : null;
  if (/\bdelhi edition\b/.test(lowered)) edition = "Delhi";
  const author = extractAuthor(lowered);
  let section = await extractSection(lowered);
  if (!section && looksLikeSportsIntent(lowered)) section = "Sports";
  if (!section && looksLikeChinaEditorialIntent(lowered)) section = "Edit";
  if (!section && looksLikeBusinessIntent(lowered)) section = "Business";

  const intent = detectIntent(lowered, author);
  const hasSemantic = SEMANTIC_CUES.some((c) => lowered.includes(c)) || intent === "topic_count" || intent === "fact_lookup";
  const hasStructured = !!(edition || section || author) || STRUCTURED_CUES.some((c) => lowered.includes(c));

  let mode: RoutedQuery["mode"];
  if (hasStructured && hasSemantic) mode = "hybrid";
  else if (hasSemantic) mode = "semantic";
  else mode = "sql";

  if (lowered.includes("which sections had the most articles")) mode = "sql";

  const people = extractPeople(lowered);
  const organizations = extractOrganizations(lowered);
  const places = extractPlaces(lowered, edition);
  const entityTerms = expandEntityTerms(people);
  const entityLabel = people.length ? people[0] : null;

  const semanticQuery =
    mode === "semantic" || mode === "hybrid"
      ? buildSemanticQuery(normalized, edition, section)
      : null;

  return {
    mode,
    intent,
    issue_date: issueDate || extractDate(lowered),
    edition,
    section,
    author,
    semantic_query: semanticQuery,
    entity_terms: entityTerms,
    entity_label: entityLabel,
    subject_strict: people.length > 0 && intent === "topic_count",
    content_people: people,
    content_organizations: organizations,
    content_locations: places,
  };
}

export function isSectionCountQuery(query: string): boolean {
  return query.toLowerCase().includes("which sections had the most articles");
}

export function isBroadListingQuery(query: string): boolean {
  const lowered = query.toLowerCase();
  if (BROAD_LIST_CUES.some((c) => lowered.includes(c))) return true;
  const patterns = [
    /\bshow me\b.*\barticles?\b/, /\bshow me\b.*\bstories\b/,
    /\bshow me\b.*\bpieces\b/, /\bwhich stories\b/, /\bwhat stories\b/,
    /\bwhich opinion pieces\b/, /\bwhat opinion pieces\b/,
    /\blist\b.*\barticles?\b/, /\blist\b.*\bstories\b/,
    /\bgive me\b.*\barticles?\b/, /\bgive me\b.*\bstories\b/,
  ];
  return patterns.some((p) => p.test(lowered));
}

export function normalizeUserQuery(query: string): string {
  let lowered = query.toLowerCase();
  lowered = lowered.replace(/\bt[\s-]?20\b/g, "t20");
  const tokens = lowered.match(/[a-z0-9]+|[^a-z0-9\s]+/g) || [];
  const corrected: string[] = [];
  for (const token of tokens) {
    if (!/^[a-z0-9]+$/.test(token)) {
      corrected.push(token);
      continue;
    }
    if (TOKEN_CORRECTIONS[token]) {
      corrected.push(TOKEN_CORRECTIONS[token]);
      continue;
    }
    if (token.length >= 5) {
      const match = getCloseMatch(token, VOCABULARY, 0.85);
      corrected.push(match || token);
    } else {
      corrected.push(token);
    }
  }
  const text = corrected
    .map((tok, i) => {
      if (i > 0 && /^[a-z0-9]+$/.test(tok) && /^[a-z0-9]+$/.test(corrected[i - 1])) {
        return " " + tok;
      }
      return tok;
    })
    .join("");
  return text.replace(/\s+/g, " ").trim();
}

export function expandSemanticQueries(baseQuery: string): string[] {
  const variants = [baseQuery];
  const lowered = baseQuery.toLowerCase();
  if (lowered.includes("india") && lowered.includes("world cup")) {
    variants.push("india world cup win", "world champions india");
  }
  if (lowered.includes("t20") && lowered.includes("world cup")) {
    variants.push("t20 world cup india win", "bcci reward world champions");
  }
  if (lowered.includes("victory")) {
    variants.push(baseQuery.replace(/victory/gi, "win"));
  }
  if (lowered.includes("covered")) {
    variants.push(baseQuery.replace(/covered/gi, "reported on"));
  }
  const seen = new Set<string>();
  const deduped: string[] = [];
  for (const v of variants) {
    const cleaned = v.replace(/\s+/g, " ").trim();
    if (cleaned && !seen.has(cleaned)) {
      seen.add(cleaned);
      deduped.push(cleaned);
    }
  }
  return deduped;
}

// ---------------------------------------------------------------------------
// Edition extraction
// ---------------------------------------------------------------------------
async function extractEdition(text: string): Promise<string | null> {
  const aliasMap = await buildPublicationAliasMap();
  const matches: { len: number; alias: string; names: Set<string> }[] = [];
  for (const [alias, names] of aliasMap) {
    const pattern = new RegExp(`\\b${escapeRegex(alias)}\\b`);
    if (pattern.test(text)) {
      matches.push({ len: alias.length, alias, names });
    }
  }
  if (!matches.length) return null;
  matches.sort((a, b) => b.len - a.len);
  const { alias, names } = matches[0];
  if (names.size === 1) return [...names][0];
  if (alias === "delhi") return "Delhi";
  const canonical = mainPublicationForFamily(alias, names);
  return canonical || alias.charAt(0).toUpperCase() + alias.slice(1);
}

function shouldExtractEdition(text: string): boolean {
  const patterns = [
    /\bedition\b/,
    /\bpublished in\b/,
    /\bfrom the\b.*\bedition\b/,
    /\bin the\b.*\bedition\b/,
    /\barticles? in\b.*\bedition\b/,
    /\b(mumbai|delhi|kolkata|chennai|bangalore|bengaluru|hyderabad|lucknow|nagpur|ludhiana|agra|bareilly|dehradun)\b.*(front page|frontpage|section|articles?|stories|news)/,
    /(front page|frontpage).*(mumbai|delhi|kolkata|chennai|bangalore|bengaluru|hyderabad|lucknow|nagpur|ludhiana)/,
  ];
  return patterns.some((pattern) => pattern.test(text));
}

// ---------------------------------------------------------------------------
// Section extraction
// ---------------------------------------------------------------------------
async function extractSection(text: string): Promise<string | null> {
  const sectionCatalog = await fetchSectionCatalog();
  const normalizedMap = new Map<string, string>();
  for (const s of sectionCatalog) {
    if (s) normalizedMap.set(normalize(s), s);
  }
  const aliasMap: Record<string, string> = {
    sports: "sports", sport: "sports", city: "city", nation: "nation",
    editorial: "edit", opinion: "edit", edit: "edit", oped: "oped",
    "op-ed": "oped", business: "business", world: "world",
    entertainment: "entertainment", frontpage: "frontpage", "front page": "frontpage",
  };
  for (const [alias, normalizedAlias] of Object.entries(aliasMap)) {
    if (normalizedAlias === "world" && /\bworld cup\b/.test(text)) continue;
    if (new RegExp(`\\b${escapeRegex(alias)}\\b`).test(text)) {
      return normalizedMap.get(normalizedAlias) || normalizedAlias.charAt(0).toUpperCase() + normalizedAlias.slice(1);
    }
  }
  const normalizedText = normalize(text);
  for (const [norm, original] of normalizedMap) {
    if (new RegExp(`\\b${escapeRegex(norm)}\\b`).test(normalizedText)) {
      return original;
    }
  }
  return null;
}

function extractDate(text: string): string | null {
  const isoMatch = text.match(/(20\d{2}-\d{2}-\d{2})/);
  if (isoMatch) return isoMatch[1];
  const monthPattern = Object.keys(MONTHS).join("|");
  const textual = text.match(new RegExp(`\\b(${monthPattern})\\s+(\\d{1,2})(?:,\\s*(20\\d{2}))?\\b`));
  if (textual) {
    const month = MONTHS[textual[1]];
    const day = parseInt(textual[2]);
    const year = textual[3] ? parseInt(textual[3]) : new Date().getFullYear();
    return `${String(year).padStart(4, "0")}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Semantic query building
// ---------------------------------------------------------------------------
function buildSemanticQuery(
  query: string,
  edition: string | null,
  section: string | null,
): string {
  const focusedTopic = extractFocusTopic(query);
  if (focusedTopic) {
    return expandSemanticQuery(focusedTopic);
  }
  let semantic = query;
  if (edition) {
    const core = editionCoreName(edition);
    semantic = semantic.replace(new RegExp(`(?:published in|in|from) the ${escapeRegex(core)} edition`, "gi"), "");
  }
  if (section) {
    semantic = semantic.replace(new RegExp(`from the ${escapeRegex(section)} section`, "gi"), "");
    semantic = semantic.replace(new RegExp(`${escapeRegex(section)} section articles`, "gi"), "");
    semantic = semantic.replace(new RegExp(`from ${escapeRegex(section)} section`, "gi"), "");
    semantic = semantic.replace(new RegExp(`which ${escapeRegex(section)} section`, "gi"), "");
  }
  semantic = semantic.replace(/\bwhich sport(?:s)? section\b/gi, "");
  semantic = semantic.replace(/\b(?:show me|list all articles|which stories|find articles|articles)\b/gi, "");
  semantic = semantic.replace(/\s+/g, " ").replace(/^[\s,.\-]+|[\s,.\-]+$/g, "");
  return expandSemanticQuery(semantic || query);
}

function extractFocusTopic(query: string): string | null {
  const lowered = query.toLowerCase().trim().replace(/[?.]+$/g, "");
  const patterns = [
    /\bhow many article around\s+(.+)/,
    /\bhow many article about\s+(.+)/,
    /\bhow many article regarding\s+(.+)/,
    /\bhow many articles around\s+(.+)/,
    /\bhow many articles about\s+(.+)/,
    /\bhow many articles regarding\s+(.+)/,
    /\bhow many times\s+(.+?)\s+(?:name\s+)?appeared\b/,
    /\bhow many article\s+(.+)/,
    /\bhow many articles\s+(.+)/,
    /\b(?:news|articles|stories)\s+about\s+(.+)/,
  ];
  for (const pattern of patterns) {
    const match = lowered.match(pattern);
    if (!match) continue;
    const topic = cleanTopicPhrase(match[1]);
    if (topic) return topic;
  }
  return null;
}

function cleanTopicPhrase(value: string): string {
  let cleaned = value.toLowerCase().trim().replace(/^[,.\s]+|[,.\s]+$/g, "");
  cleaned = cleaned.replace(/\band\s+and\b/g, " and");
  cleaned = cleaned.replace(/\s+and\s+(?:in (?:what|which) context(?: they are)?|(?:what|which) context(?: they are)?|what they (?:were|are)? about.*)$/g, "");
  cleaned = cleaned.replace(/\bin (?:what|which) context(?: they are)?\b.*$/g, "");
  cleaned = cleaned.replace(/\b(?:what|which) context(?: they are)?\b.*$/g, "");
  cleaned = cleaned.replace(/\bwhat they (?:were|are)? about\b.*$/g, "");
  return cleaned.replace(/\s+/g, " ").trim().replace(/^[,.\s]+|[,.\s]+$/g, "");
}

function expandSemanticQuery(query: string): string {
  let expanded = query.replace(/\bt[\s-]?20\b/gi, "t20");
  const lowered = expanded.toLowerCase();
  const additions: string[] = [];
  if (lowered.includes("victory")) additions.push("win");
  if (lowered.includes("champion") || lowered.includes("champions")) additions.push("world champions");
  if (lowered.includes("world cup") && lowered.includes("india")) additions.push("india world cup win");
  if (lowered.includes("t20") && lowered.includes("world cup")) additions.push("t20 world cup");
  if (lowered.includes("covered") && lowered.includes("victory")) additions.push("coverage");
  if (lowered.includes("budget") && lowered.includes("middle class")) {
    additions.push("inflation prices households tax growth economists");
  }
  if (lowered.includes("budget")) additions.push("economy prices inflation business");
  if (additions.length) expanded = `${expanded} ${additions.join(" ")}`;
  return expanded.replace(/\s+/g, " ").trim();
}

// ---------------------------------------------------------------------------
// Intent detection
// ---------------------------------------------------------------------------
function looksLikeSportsIntent(text: string): boolean {
  return ["world cup", "t20", "bcci", "champions", "cricket", "football", "ipl", "match", "trophy"]
    .some((c) => text.includes(c));
}

function looksLikeBusinessIntent(text: string): boolean {
  return ["budget", "middle class", "inflation", "price rise", "prices", "economists", "growth", "tax", "fuel inflation", "oil prices"]
    .some((c) => text.includes(c));
}

function looksLikeChinaEditorialIntent(text: string): boolean {
  return text.includes("china") && (text.includes("growth ambition") || text.includes("growth ambitions") || text.includes("gdp target"));
}

function detectIntent(text: string, author: string | null): RoutedQuery["intent"] {
  if (author) {
    return /\bhow many\b|\bcount\b|\bnumber of\b/.test(text) ? "author_count" : "author_lookup";
  }
  if (/\bhow many\b|\bcount\b|\bnumber of\b/.test(text)) {
    if (/\bhow many times\b|\bname appeared\b|\barticle(?:s)?\s+(?:around|about|regarding|on)\b/.test(text)) {
      return "topic_count";
    }
    if (/\barticle(?:s)?\b/.test(text)) {
      return "article_count";
    }
    return "fact_lookup";
  }
  return "lookup";
}

function extractPeople(text: string): string[] {
  const out: string[] = [];
  for (const [alias, meta] of Object.entries(PERSON_ALIAS_MAP)) {
    if (new RegExp(`\\b${escapeRegex(alias)}\\b`).test(text) && !out.includes(meta.canonical)) {
      out.push(meta.canonical);
    }
  }
  return out;
}

function extractOrganizations(text: string): string[] {
  const out: string[] = [];
  for (const org of KNOWN_ORGANIZATIONS) {
    if (new RegExp(`\\b${escapeRegex(org)}\\b`).test(text) && !out.includes(org)) {
      out.push(org);
    }
  }
  return out;
}

function extractPlaces(text: string, edition: string | null): string[] {
  const out: string[] = [];
  if (edition) {
    const core = editionCoreName(edition).toLowerCase();
    if (core) {
      out.push(core);
    }
  }
  for (const place of KNOWN_PLACES) {
    if (new RegExp(`\\b${escapeRegex(place)}\\b`).test(text) && !out.includes(place)) {
      out.push(place);
    }
  }
  return out;
}

function extractAuthor(text: string): string | null {
  const match = text.match(/\b(?:by|author)\s+([a-z][a-z\s'.-]{3,80})/i);
  if (!match) return null;
  return match[1]
    .trim()
    .replace(/\b\w/g, (value) => value.toUpperCase());
}

function expandEntityTerms(people: string[]): string[] {
  const out: string[] = [];
  for (const person of people) {
    const alias = PERSON_ALIAS_MAP[person.toLowerCase()] || PERSON_ALIAS_MAP[person.toLowerCase().replace(/\s+/g, " ")];
    const terms = alias ? alias.terms : [person];
    for (const term of terms) {
      if (!out.some((existing) => existing.toLowerCase() === term.toLowerCase())) {
        out.push(term);
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Publication helpers
// ---------------------------------------------------------------------------
async function buildPublicationAliasMap(): Promise<Map<string, Set<string>>> {
  const catalog = await fetchPublicationCatalog();
  const aliasMap = new Map<string, Set<string>>();
  for (const row of catalog) {
    const pubId = row.id as string;
    const pubName = row.publication_name as string;
    for (const alias of publicationAliases(pubId, pubName)) {
      if (!aliasMap.has(alias)) aliasMap.set(alias, new Set());
      aliasMap.get(alias)!.add(pubName);
    }
  }
  return aliasMap;
}

function publicationAliases(pubId: string, pubName: string): Set<string> {
  const aliases = new Set<string>();
  const suffix = pubId.includes("_") ? pubId.split("_").slice(1).join("_") : pubId;
  aliases.add(normalize(stripSuffixes(suffix)));
  aliases.add(normalize(suffix));
  const familyAlias = publicationFamilyAlias(pubId, pubName);
  if (familyAlias) aliases.add(familyAlias);
  if (pubName.includes(" - ")) {
    let right = pubName.split(" - ").slice(1).join(" - ");
    right = right.replace(/_Digital/g, "");
    const normalizedRight = normalize(right);
    aliases.add(normalizedRight);
    aliases.add(normalize(stripSuffixes(normalizedRight)));
  }
  aliases.delete("");
  return aliases;
}

function stripSuffixes(value: string): string {
  const normalized = normalize(value);
  for (const suffix of SUFFIX_STRIPS) {
    if (normalized.endsWith(suffix) && normalized.length > suffix.length) {
      return normalized.slice(0, -suffix.length);
    }
  }
  return normalized;
}

function editionCoreName(pubName: string): string {
  if (pubName.includes(" - ")) {
    const right = pubName.split(" - ").slice(1).join(" - ").replace(/_Digital/g, "");
    return right.replace(/(?<!^)([A-Z])/g, " $1").trim();
  }
  return pubName;
}

function publicationFamilyAlias(pubId: string, pubName: string): string | null {
  const left = pubName.includes(" - ") ? pubName.split(" - ")[0] : pubId.split("_")[0];
  let normalized = normalize(left);
  normalized = normalized.replace(/^toi/, "");
  normalized = normalized.replace(/h?bs$/, "");
  return normalized || null;
}

function mainPublicationForFamily(alias: string, names: Set<string>): string | null {
  const ordered = [...names].sort();
  const aliasNorm = normalize(alias);
  const specialTargets: Record<string, string> = {
    mumbai: "mumbaicity", kolkata: "kolkatacity", chennai: "chennai",
    bangalore: "bangalorecity", hyderabad: "hyderabad",
    lucknow: "lucknowcity", nagpur: "nagpurcity",
  };
  const target = specialTargets[aliasNorm];
  if (target) {
    for (const name of ordered) {
      const right = name.includes(" - ") ? name.split(" - ").slice(1).join(" - ") : name;
      const rightNorm = normalize(right.replace(/_Digital/g, ""));
      if (rightNorm.includes(target)) return name;
    }
  }
  const scored = ordered.map((name): [number, string] => {
    const right = name.includes(" - ") ? name.split(" - ").slice(1).join(" - ") : name;
    const rightNorm = normalize(right.replace(/_Digital/g, ""));
    if (aliasNorm && rightNorm.includes(aliasNorm) && rightNorm.includes("city")) return [1, name];
    if (aliasNorm && rightNorm.includes(aliasNorm)) return [2, name];
    if (rightNorm.includes("city")) return [3, name];
    return [4, name];
  });
  scored.sort((a, b) => a[0] - b[0] || a[1].localeCompare(b[1]));
  const [, candidate] = scored[0];
  const candidateRight = candidate.includes(" - ") ? candidate.split(" - ").slice(1).join(" - ") : candidate;
  const candidateNorm = normalize(candidateRight.replace(/_Digital/g, ""));
  if (aliasNorm && candidateNorm.includes(aliasNorm)) return candidate;
  return null;
}

// ---------------------------------------------------------------------------
// DB fetchers (uncached — edge functions are short-lived)
// ---------------------------------------------------------------------------
export async function fetchPublicationCatalog(): Promise<
  { id: string; publication_name: string }[]
> {
  return await query<{ id: string; publication_name: string }>(
    "publications",
    "select=id,publication_name&order=publication_name",
  );
}

export async function fetchSectionCatalog(): Promise<string[]> {
  const rows = await query<{ normalized_section: string | null }>(
    "sections",
    "select=normalized_section&normalized_section=not.is.null&order=normalized_section",
  );
  const seen = new Set<string>();
  const values: string[] = [];
  for (const row of rows) {
    const value = row.normalized_section;
    if (value && !seen.has(value)) {
      seen.add(value);
      values.push(value);
    }
  }
  return values;
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
function normalize(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Simple Levenshtein-ratio close match (replaces difflib.get_close_matches). */
function getCloseMatch(
  word: string,
  candidates: Set<string>,
  cutoff: number,
): string | null {
  let best: string | null = null;
  let bestRatio = 0;
  for (const c of candidates) {
    const ratio = similarity(word, c);
    if (ratio >= cutoff && ratio > bestRatio) {
      bestRatio = ratio;
      best = c;
    }
  }
  return best;
}

function similarity(a: string, b: string): number {
  if (a === b) return 1;
  const longer = a.length > b.length ? a : b;
  const shorter = a.length > b.length ? b : a;
  if (!longer.length) return 1;
  const dist = levenshtein(longer, shorter);
  return (longer.length - dist) / longer.length;
}

function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  const dp: number[] = Array.from({ length: n + 1 }, (_, i) => i);
  for (let i = 1; i <= m; i++) {
    let prev = dp[0];
    dp[0] = i;
    for (let j = 1; j <= n; j++) {
      const temp = dp[j];
      dp[j] = a[i - 1] === b[j - 1] ? prev : 1 + Math.min(prev, dp[j], dp[j - 1]);
      prev = temp;
    }
  }
  return dp[n];
}
