import { getDb } from "./db.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface RoutedQuery {
  mode: "sql" | "semantic" | "hybrid";
  issue_date: string | null;
  edition: string | null;
  section: string | null;
  semantic_query: string | null;
}

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
  let edition = await extractEdition(lowered);
  if (/\bdelhi edition\b/.test(lowered)) edition = "Delhi";
  let section = await extractSection(lowered);
  if (!section && looksLikeSportsIntent(lowered)) section = "Sports";
  if (!section && looksLikeBusinessIntent(lowered)) section = "Business";

  const hasSemantic = SEMANTIC_CUES.some((c) => lowered.includes(c));
  const hasStructured = !!(edition || section) || STRUCTURED_CUES.some((c) => lowered.includes(c));

  let mode: RoutedQuery["mode"];
  if (hasStructured && hasSemantic) mode = "hybrid";
  else if (hasSemantic) mode = "semantic";
  else mode = "sql";

  if (lowered.includes("which sections had the most articles")) mode = "sql";

  const semanticQuery =
    mode === "semantic" || mode === "hybrid"
      ? buildSemanticQuery(normalized, edition, section)
      : null;

  return {
    mode,
    issue_date: issueDate || extractDate(lowered),
    edition,
    section,
    semantic_query: semanticQuery,
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
  const sql = getDb();
  const rows = await sql`
    select id, publication_name from publications order by publication_name
  `;
  return rows as { id: string; publication_name: string }[];
}

export async function fetchSectionCatalog(): Promise<string[]> {
  const sql = getDb();
  const rows = await sql`
    select distinct normalized_section
    from sections
    where normalized_section is not null
    order by normalized_section
  `;
  return rows.map((r) => r.normalized_section as string);
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
