import type { Row } from "./db.ts";
import { chatCompletion } from "./openai.ts";
import {
  isBroadListingQuery,
  isSectionCountQuery,
  routeQuery,
  fetchPublicationCatalog,
} from "./query-router.ts";
import {
  runQuery,
  fetchSqlArticleCount,
  fetchMatchingPublications,
  type QueryResponse,
} from "./query-service.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface ChatResponse {
  answer: string;
  mode: string;
  citations: Record<string, unknown>[];
  session_context: Record<string, unknown> | null;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const SYSTEM_PROMPT = `You answer questions from the TOI e-paper dataset.
Use only the provided article results. If the answer is not supported by the results, say so clearly.
Mention edition and section when useful.
If the user asks a follow-up question, use the prior conversation only as conversational context, but ground factual claims in the retrieved articles.
Sound natural and concise, like a helpful newsroom analyst, not a robot.`;

const STRUCTURED_RESULT_WINDOW = 5000;
const HYBRID_RESULT_WINDOW = 5000;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
export async function answerQuestion(
  question: string,
  issueDate: string | null,
  limit: number,
  sessionFilters: Record<string, unknown> | null = null,
  history: { role: string; content: string }[] | null = null,
  sessionContext: Record<string, unknown> | null = null,
): Promise<ChatResponse> {
  // Edition clarification followup
  const editionClarification = formatEditionFollowupAnswer(question, sessionContext);
  if (editionClarification) {
    editionClarification.session_context = sessionContext;
    return editionClarification;
  }

  // Article text from context
  if (wantsArticleText(question)) {
    const candidate = articleCandidateFromContext(question, sessionContext);
    if (candidate) {
      const resp = formatContextArticleTextAnswer(candidate, sessionContext);
      resp.session_context = sessionContext;
      return resp;
    }
  }

  const retrievalQuestion = augmentFollowupQuestion(question, history, sessionContext);
  let edition = filterValue(sessionFilters, "edition");
  let section = filterValue(sessionFilters, "section");
  const routed = await routeQuery(retrievalQuestion, issueDate);
  edition = edition || contextValue(sessionContext, "edition", question);
  section = section || contextValue(sessionContext, "section", question);

  const broadListing = isBroadListingQuery(question);
  const countQuery = isCountQuery(question);
  const reqArticleCount = requestedArticleCount(question);
  const showRefs = shouldShowReferences(question);

  let resultWindow = limit;
  if (countQuery || isSectionCountQuery(question) || (broadListing && routed.mode === "sql")) {
    resultWindow = STRUCTURED_RESULT_WINDOW;
  } else if (broadListing && routed.mode === "hybrid") {
    resultWindow = HYBRID_RESULT_WINDOW;
  } else if (shouldUseSummaryAnswer(question, routed.mode)) {
    resultWindow = Math.max(limit, 24);
  }

  const queryResponse = await runQuery(retrievalQuestion, issueDate, limit, {
    edition,
    section,
    resultWindow,
  });

  // Ambiguous edition
  const ambiguousAnswer = await formatAmbiguousEditionAnswer(queryResponse);
  if (ambiguousAnswer) {
    ambiguousAnswer.session_context = await buildSessionContext(question, queryResponse, sessionContext);
    return ambiguousAnswer;
  }

  // Section count
  if (isSectionCountQuery(question)) {
    const resp = formatSectionCounts(queryResponse);
    resp.session_context = await buildSessionContext(question, queryResponse, sessionContext);
    return resp;
  }

  // Count
  if (countQuery) {
    const resp = await formatCountAnswer(question, queryResponse);
    resp.session_context = await buildSessionContext(question, queryResponse, sessionContext);
    return resp;
  }

  // Article text
  if (wantsArticleText(question)) {
    const resp = formatArticleTextAnswer(queryResponse);
    resp.session_context = await buildSessionContext(question, queryResponse, sessionContext);
    return resp;
  }

  // Broad listing / story summary
  if (broadListing && (queryResponse.mode === "sql" || queryResponse.mode === "hybrid")) {
    if (wantsExactArticleListing(question) && reqArticleCount) {
      const resp = formatArticleListing(question, queryResponse, reqArticleCount);
      resp.session_context = await buildSessionContext(question, queryResponse, sessionContext);
      return resp;
    }
    const resp = await formatStorySummary(question, queryResponse);
    resp.session_context = await buildSessionContext(question, queryResponse, sessionContext);
    return resp;
  }

  if (shouldUseSummaryAnswer(question, queryResponse.mode) && !showRefs) {
    const resp = await formatStorySummary(question, queryResponse);
    resp.session_context = await buildSessionContext(question, queryResponse, sessionContext);
    return resp;
  }

  // Default: citation answer via LLM
  const contextLines: string[] = [];
  const citations: Record<string, unknown>[] = [];
  for (const item of queryResponse.results.slice(0, limit)) {
    contextLines.push(
      [
        `Headline: ${item.headline}`,
        `Edition: ${item.edition}`,
        `Section: ${item.section}`,
        `Issue Date: ${item.issue_date}`,
        `Excerpt: ${item.excerpt || item.matched_chunk}`,
      ].join("\n"),
    );
    citations.push({
      article_id: item.external_article_id,
      headline: item.headline,
      edition: item.edition,
      section: item.section,
      issue_date: item.issue_date,
      reference_text: item.excerpt || item.matched_chunk || "No supporting excerpt available.",
    });
  }

  const conversationCtx = formatHistory(history);
  const userPrompt =
    (conversationCtx ? `Conversation so far:\n${conversationCtx}\n\n` : "") +
    `Question: ${question}\n\n` +
    "Write a concise answer. Focus on the main story themes and avoid edition-by-edition narration unless it is essential.\n\n" +
    "Retrieved articles:\n\n" +
    contextLines.join("\n\n---\n\n");

  const answer = await chatCompletion(SYSTEM_PROMPT, userPrompt);
  return {
    answer,
    mode: queryResponse.mode,
    citations: showRefs ? citations : [],
    session_context: await buildSessionContext(question, queryResponse, sessionContext),
  };
}

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------
function formatSectionCounts(qr: QueryResponse): ChatResponse {
  const rows = qr.results;
  if (!rows.length) {
    return { answer: "No section counts matched the requested issue date.", mode: "sql", citations: [], session_context: null };
  }
  const top = rows[0];
  const lines = [
    `${top.section || "Unclassified"} had the most articles on March 11 with ${top.article_count} pieces.`,
    "",
    "Full section ranking:",
  ];
  rows.forEach((r, i) => lines.push(`${i + 1}. ${r.section || "Unclassified"}: ${r.article_count}`));
  return { answer: lines.join("\n"), mode: "sql", citations: [], session_context: null };
}

function formatArticleListing(
  _question: string,
  qr: QueryResponse,
  reqCount: number | null,
): ChatResponse {
  const rows = qr.results;
  if (!rows.length) {
    return { answer: "No matching articles were found for that request.", mode: qr.mode, citations: [], session_context: null };
  }
  const display = reqCount ? rows.slice(0, reqCount) : rows;
  const citations: Record<string, unknown>[] = [];
  const lines = [`I found ${rows.length} matching articles.`];
  if (reqCount) {
    lines[0] = `I found ${rows.length} matching articles. Here are ${Math.min(reqCount, rows.length)} worth looking at.`;
  }
  display.forEach((item, i) => {
    lines.push(`${i + 1}. ${item.headline || "Untitled"} | ${item.edition || "Unknown edition"} | ${item.section || "Unknown section"} | ${item.issue_date || "Unknown date"}`);
    citations.push({
      article_id: item.external_article_id,
      headline: item.headline || "Untitled",
      edition: item.edition || "Unknown edition",
      section: item.section || "Unknown section",
      issue_date: item.issue_date || "Unknown date",
      reference_text: item.excerpt || item.matched_chunk || "No supporting excerpt available.",
    });
  });
  let answer = lines.join("\n");
  if (qr.mode === "hybrid") {
    answer = `I found ${rows.length} semantically matched articles after applying the filters you asked for.\n\n` + lines.slice(1).join("\n");
  }
  return { answer, mode: qr.mode, citations, session_context: null };
}

async function formatStorySummary(question: string, qr: QueryResponse): Promise<ChatResponse> {
  const storyGroups = groupUniqueStories(qr.results);
  if (!storyGroups.length) {
    return { answer: "I couldn't find any matching stories for that request.", mode: qr.mode, citations: [], session_context: null };
  }
  const contextBlocks = storyGroups.slice(0, 10).map((s) =>
    [
      `Headline: ${s.headline}`,
      `Section: ${s.section}`,
      `Edition count: ${s.count}`,
      `Representative excerpt: ${s.excerpt}`,
    ].join("\n")
  );
  const prompt =
    `Question: ${question}\n\n` +
    "Write a concise answer that reads like a human analyst. " +
    "Summarize only the unique stories below. " +
    "Merge repeated editions of the same story into one point. " +
    "Do not use numbered labels like Story 1, Story 2, or numbered theme lists. " +
    "Do not output a raw article list, citation list, or repetitive edition-by-edition breakdown. " +
    "Avoid mentioning specific editions unless the user explicitly asked for editions. " +
    "If the user asked for all or broad coverage, mention the total number of matching articles and then summarize the dominant unique stories. " +
    "If one story clearly dominates, say that directly. " +
    "Keep the answer grounded only in the story summaries below.\n\n" +
    `Total matching articles: ${qr.results.length}\n\n` +
    contextBlocks.join("\n\n---\n\n");
  const answer = await chatCompletion(SYSTEM_PROMPT, prompt);
  return { answer, mode: qr.mode, citations: [], session_context: null };
}

function formatArticleTextAnswer(qr: QueryResponse): ChatResponse {
  const rows = qr.results;
  if (!rows.length) {
    return { answer: "I couldn't find a relevant article for that request.", mode: qr.mode, citations: [], session_context: null };
  }
  const item = rows[0];
  const text = item.excerpt || item.matched_chunk || "No article text is available for this result.";
  const answer = `Here is one relevant article excerpt:\n\n${item.headline || "Untitled"}\n${item.section || "Unknown section"} | ${item.issue_date || "Unknown date"}\n\n${text}`;
  return {
    answer,
    mode: qr.mode,
    citations: [{
      article_id: item.external_article_id,
      headline: item.headline,
      edition: item.edition,
      section: item.section,
      issue_date: item.issue_date,
      reference_text: text,
    }],
    session_context: null,
  };
}

function formatContextArticleTextAnswer(
  candidate: Record<string, unknown>,
  sessionContext: Record<string, unknown> | null,
): ChatResponse {
  const headline = String(candidate.headline || "Untitled");
  const section = String(candidate.section || "Unknown section");
  const issueDate = String(candidate.issue_date || "Unknown date");
  const text = String(candidate.reference_text || "No article text is available for this result.");
  const answer = `Here is one relevant article excerpt:\n\n${headline}\n${section} | ${issueDate}\n\n${text}`;
  return {
    answer,
    mode: String((sessionContext || {}).last_mode || "semantic"),
    citations: [{
      article_id: candidate.article_id,
      headline,
      edition: candidate.edition,
      section,
      issue_date: issueDate,
      reference_text: text,
    }],
    session_context: null,
  };
}

async function formatCountAnswer(_question: string, qr: QueryResponse): Promise<ChatResponse> {
  const filters = qr.filters;
  const edition = filters.edition as string | null;
  const section = filters.section as string | null;
  const issueDate = filters.issue_date as string | null;
  const count = qr.mode === "sql"
    ? await fetchSqlArticleCount(issueDate || null, edition || null, section || null)
    : qr.results.length;
  const scopeParts: string[] = [];
  scopeParts.push(section ? `${section} articles` : "articles");
  if (edition) scopeParts.push(`in ${edition}`);
  if (issueDate) scopeParts.push(`on ${issueDate}`);
  return { answer: `There were ${count} ${scopeParts.join(" ")}.`, mode: qr.mode, citations: [], session_context: null };
}

async function formatAmbiguousEditionAnswer(qr: QueryResponse): Promise<ChatResponse | null> {
  const edition = qr.filters.edition as string | null;
  const issueDate = qr.filters.issue_date as string | null;
  if (!edition) return null;
  const catalog = await fetchPublicationCatalog();
  const pubNames = new Set(catalog.map((r) => r.publication_name));
  if (pubNames.has(edition)) return null;
  const matches = await fetchMatchingPublications(issueDate || null, edition);
  if (matches.length <= 1) return null;
  const dateText = issueDate ? ` for ${issueDate}` : "";
  const lines = [
    `I don't see a single exact edition named ${edition} in the dataset${dateText}.`,
    `Under the TOI${edition} publication family${dateText}, I found these editions instead:`,
  ];
  for (const row of matches) lines.push(`- ${row.publication_name}: ${row.article_count} articles`);
  lines.push("Ask for one of those exact editions and I'll give you a precise answer.");
  return { answer: lines.join("\n"), mode: qr.mode, citations: [], session_context: null };
}

function formatEditionFollowupAnswer(
  question: string,
  sessionContext: Record<string, unknown> | null,
): ChatResponse | null {
  if (!sessionContext) return null;
  const matches = sessionContext.ambiguous_publications as Record<string, unknown>[] | undefined;
  const edition = sessionContext.ambiguous_edition as string | undefined;
  const issueDate = sessionContext.issue_date as string | undefined;
  if (!edition || !Array.isArray(matches) || !matches.length) return null;
  if (!wantsEditionClarification(question)) return null;
  const dateText = issueDate ? ` for ${issueDate}` : "";
  const lines = [
    `There isn't a single exact edition named ${edition} in the dataset${dateText}.`,
    `These are the available editions under the TOI${edition} publication family${dateText}:`,
  ];
  for (const row of matches) {
    lines.push(`- ${row.publication_name || "Unknown publication"}: ${row.article_count || 0} articles`);
  }
  lines.push("Ask for one of these exact editions and I'll give you the precise result.");
  return { answer: lines.join("\n"), mode: String(sessionContext.last_mode || "sql"), citations: [], session_context: null };
}

// ---------------------------------------------------------------------------
// Story grouping
// ---------------------------------------------------------------------------
interface StoryGroup {
  headline: string;
  section: string;
  editions: string[];
  count: number;
  excerpt: string;
}

function groupUniqueStories(rows: Row[]): StoryGroup[] {
  const grouped = new Map<string, StoryGroup>();
  for (const row of rows) {
    const headline = String(row.headline || "").trim();
    if (!headline) continue;
    const key = normalizeHeadline(headline);
    if (!key) continue;
    if (!grouped.has(key)) {
      grouped.set(key, {
        headline,
        section: String(row.section || "Unknown section"),
        editions: [],
        count: 0,
        excerpt: String(row.excerpt || row.matched_chunk || ""),
      });
    }
    const story = grouped.get(key)!;
    story.count++;
    const edition = String(row.edition || "Unknown edition");
    if (!story.editions.includes(edition)) story.editions.push(edition);
    const newExcerpt = String(row.excerpt || row.matched_chunk || "");
    if (newExcerpt.length > story.excerpt.length) story.excerpt = newExcerpt;
  }
  return [...grouped.values()].sort((a, b) => b.count - a.count || b.headline.localeCompare(a.headline));
}

// ---------------------------------------------------------------------------
// Session context
// ---------------------------------------------------------------------------
async function buildSessionContext(
  question: string,
  qr: QueryResponse,
  prior: Record<string, unknown> | null,
): Promise<Record<string, unknown>> {
  const base: Record<string, unknown> = { ...(prior || {}) };
  const filters = qr.filters || {};
  if (filters.edition) base.edition = filters.edition;
  if (filters.section) base.section = filters.section;
  if (filters.issue_date) base.issue_date = filters.issue_date;
  base.last_mode = qr.mode;
  base.last_question = question;

  const stories = groupUniqueStories(qr.results);
  const storyTitles = stories.slice(0, 5).map((s) => s.headline);
  const articleCandidates: Record<string, unknown>[] = [];
  for (const item of qr.results.slice(0, 8)) {
    articleCandidates.push({
      article_id: item.external_article_id,
      headline: item.headline,
      edition: item.edition,
      section: item.section,
      issue_date: item.issue_date,
      reference_text: item.excerpt || item.matched_chunk,
    });
  }
  const storyCandidates: Record<string, unknown>[] = [];
  for (const story of stories.slice(0, 20)) {
    storyCandidates.push({
      headline: story.headline,
      edition: story.editions[0] || null,
      section: story.section,
      issue_date: filters.issue_date || null,
      reference_text: story.excerpt,
    });
  }
  if (storyTitles.length) {
    base.story_titles = storyTitles;
    base.last_topic = storyTitles[0];
  }
  if (storyCandidates.length) base.story_candidates = storyCandidates;
  if (articleCandidates.length) base.article_candidates = articleCandidates;

  const edition = filters.edition as string | null;
  const issueDate = filters.issue_date as string | null;
  if (edition) {
    const catalog = await fetchPublicationCatalog();
    const pubNames = new Set(catalog.map((r) => r.publication_name));
    if (!pubNames.has(edition)) {
      const matches = await fetchMatchingPublications(issueDate || null, edition);
      if (matches.length > 1) {
        base.ambiguous_edition = edition;
        base.ambiguous_publications = matches;
      }
    }
  }
  return base;
}

// ---------------------------------------------------------------------------
// Query classification helpers
// ---------------------------------------------------------------------------
function isCountQuery(question: string): boolean {
  const lowered = question.toLowerCase();
  return [/\bhow many\b/, /\bcount\b/, /\bnumber of\b/, /\bnumbers of\b/].some((p) => p.test(lowered));
}

function shouldUseSummaryAnswer(question: string, mode: string): boolean {
  if (mode !== "semantic" && mode !== "hybrid") return false;
  if (wantsExactArticleListing(question) || shouldShowReferences(question)) return false;
  const lowered = question.toLowerCase();
  const patterns = [
    /\bfind\b.*\barticles?\b/, /\bfind\b.*\bnews\b/, /\brelated to\b/,
    /\bnews about\b/, /\btell me news about\b/, /\bwhat was written about\b/,
    /\bwhat happened in\b/, /\bwhat is the news about\b/, /\bwhich stories\b/,
  ];
  return patterns.some((p) => p.test(lowered));
}

function requestedArticleCount(question: string): number | null {
  const match = question.toLowerCase().match(/\b(?:show|give|list)\s+(?:me\s+)?(\d{1,2})\s+articles?\b/);
  return match ? parseInt(match[1]) : null;
}

function wantsArticleText(question: string): boolean {
  const lowered = question.toLowerCase();
  const patterns = [
    /\btext of\b.*\barticle\b/, /\barticle text\b/, /\btext of any article\b/,
    /\bany one article\b/, /\bany one of article\b/, /\bany article\b/,
    /\bshow me text\b/, /\bshow the text\b/, /\bshow an article\b/,
    /\bshow any one\b.*\barticle\b/, /\bshow any\b.*\barticle\b/,
    /\bgive me an article\b/, /\bgive me the article\b/, /\bgive me article\b/,
    /\bgive any one\b.*\barticle\b/, /\bgive any\b.*\barticle\b/,
    /\bshow article text\b/, /\bfull article\b/, /\bexcerpt\b/,
  ];
  return patterns.some((p) => p.test(lowered));
}

function shouldShowReferences(question: string): boolean {
  if (wantsArticleText(question)) return true;
  const lowered = question.toLowerCase();
  const patterns = [
    /\bshow\b.*\barticles?\b/, /\bgive\b.*\bthe article\b/, /\bgive\b.*\ban article\b/,
    /\bgive\b.*\barticle\b/, /\bgive\b.*\barticles?\b/, /\blist\b.*\barticles?\b/,
    /\bwhich articles\b/, /\breferences?\b/, /\bsources?\b/, /\bcitations?\b/, /\bexamples?\b/,
  ];
  return patterns.some((p) => p.test(lowered));
}

function wantsExactArticleListing(question: string): boolean {
  if (wantsArticleText(question)) return false;
  const lowered = question.toLowerCase();
  const patterns = [
    /\bgive me\b.*\bthe article\b/, /\bgive me\b.*\ban article\b/,
    /\bgive me\b.*\barticle\b/, /\bshow me\b.*\barticles?\b/,
    /\bshow\b.*\barticles?\b/, /\bgive me\b.*\barticles?\b/,
    /\blist\b.*\barticles?\b/, /\bwhich articles\b/,
    /\bshow sources\b/, /\bshow references\b/, /\bshow citations\b/,
  ];
  return patterns.some((p) => p.test(lowered));
}

function wantsEditionClarification(question: string): boolean {
  const lowered = question.toLowerCase();
  return [
    /\bwhat exact editions\b/, /\bwhich exact editions\b/,
    /\bwhat editions are available\b/, /\bwhich editions are available\b/,
    /\bavailable editions\b/, /\bexact editions\b/,
  ].some((p) => p.test(lowered));
}

// ---------------------------------------------------------------------------
// Followup augmentation
// ---------------------------------------------------------------------------
function augmentFollowupQuestion(
  question: string,
  history: { role: string; content: string }[] | null,
  sessionContext: Record<string, unknown> | null,
): string {
  if (isGenericArticleRequest(question) && sessionContext?.last_topic) {
    return `${question} ${sessionContext.last_topic}`;
  }
  if (!isReferentialFollowup(question)) return question;
  if (!history) return augmentWithSessionStory(question, sessionContext);
  const title = bestHistoryTitleMatch(question, history, sessionContext);
  if (!title) return augmentWithSessionStory(question, sessionContext);
  return `${question} ${title}`;
}

function bestHistoryTitleMatch(
  question: string,
  history: { role: string; content: string }[] | null,
  sessionContext: Record<string, unknown> | null,
): string | null {
  let candidates = extractHistoryTitles(history);
  candidates.push(...sessionStoryTitles(sessionContext));
  // Deduplicate
  const seen = new Set<string>();
  const deduped: string[] = [];
  for (const c of candidates) {
    const norm = normalizeHeadline(c);
    if (norm && !seen.has(norm)) { seen.add(norm); deduped.push(c); }
  }
  candidates = deduped;
  if (!candidates.length) return null;
  const topic = extractFollowupTopic(question);
  if (!topic) return null;
  const normTopic = normalizeHeadline(topic);
  const scored: [number, string][] = [];
  for (const c of candidates) {
    const normC = normalizeHeadline(c);
    const overlap = tokenOverlapScore(normTopic, normC);
    const sim = sequenceSimilarity(normTopic, normC);
    const score = overlap + sim;
    if (score >= 0.95 || (overlap >= 0.35 && sim >= 0.45)) scored.push([score, c]);
  }
  if (!scored.length) return null;
  scored.sort((a, b) => b[0] - a[0]);
  return scored[0][1];
}

function extractFollowupTopic(question: string): string | null {
  const lowered = question.toLowerCase();
  const patterns = [/(?:regarding|about|on)\s+(.+)/, /(?:the article|article)\s+(?:on|about|regarding)?\s*(.+)/];
  for (const p of patterns) {
    const match = lowered.match(p);
    if (match) {
      let topic = match[1];
      topic = topic.replace(/\b(?:what you have shared above as|what you shared above|you have shared above|you shared above|what you mentioned above|you mentioned above)\b/g, " ");
      topic = topic.replace(/\b(?:what|which|that|those|above|shared|mentioned|have|you|as)\b/g, " ");
      topic = topic.replace(/\s+/g, " ").replace(/^[\s,.?]+|[\s,.?]+$/g, "");
      if (topic) return topic;
    }
  }
  return question;
}

function extractHistoryTitles(history: { role: string; content: string }[] | null): string[] {
  if (!history) return [];
  const titles: string[] = [];
  const seen = new Set<string>();
  const quotedPattern = /["""']([^"""']{8,160})["""']/g;
  const titleCasePattern = /(?:[A-Z][A-Za-z''&.\-]+(?:\s+[A-Z][A-Za-z''&.\-]+){2,8})/g;
  for (const item of [...history].reverse().slice(0, 8)) {
    const content = (item.content || "").trim();
    if (!content) continue;
    const candidates = [...content.matchAll(quotedPattern)].map((m) => m[1]);
    candidates.push(...[...content.matchAll(titleCasePattern)].map((m) => m[0]));
    for (const c of candidates) {
      const cleaned = c.replace(/^[\s,.\-]+|[\s,.\-]+$/g, "");
      if (cleaned.length < 8) continue;
      const norm = normalizeHeadline(cleaned);
      if (norm && !seen.has(norm)) { seen.add(norm); titles.push(cleaned); }
    }
  }
  return titles;
}

function augmentWithSessionStory(
  question: string,
  sessionContext: Record<string, unknown> | null,
): string {
  const titles = sessionStoryTitles(sessionContext);
  if (!titles.length || !isReferentialFollowup(question)) return question;
  return `${question} ${titles[0]}`;
}

function sessionStoryTitles(sessionContext: Record<string, unknown> | null): string[] {
  if (!sessionContext) return [];
  const titles = sessionContext.story_titles;
  if (Array.isArray(titles)) return titles.filter(Boolean).map(String);
  return [];
}

function articleCandidateFromContext(
  question: string,
  sessionContext: Record<string, unknown> | null,
): Record<string, unknown> | null {
  if (!sessionContext) return null;
  const rawCandidates = (sessionContext.article_candidates || []) as Record<string, unknown>[];
  const storyCandidates = (sessionContext.story_candidates || []) as Record<string, unknown>[];
  const candidates = isGenericArticleRequest(question) ? rawCandidates : storyCandidates;
  if (!Array.isArray(candidates) || !candidates.length) return null;
  if (isGenericArticleRequest(question)) {
    const ranked = rankContextArticleCandidates(candidates, sessionContext);
    return ranked[0] || null;
  }
  const topic = extractFollowupTopic(question);
  if (!topic || (isReferentialFollowup(question) && ["that", "those", "this article", "that article", "that story"].includes(topic))) {
    return candidates[0];
  }
  const normTopic = normalizeHeadline(topic);
  const scored: [number, Record<string, unknown>][] = [];
  for (const c of candidates) {
    const normC = normalizeHeadline(String(c.headline || ""));
    const overlap = tokenOverlapScore(normTopic, normC);
    const sim = sequenceSimilarity(normTopic, normC);
    const score = overlap + sim;
    if (score >= 0.7 || (overlap >= 0.2 && sim >= 0.3)) scored.push([score, c]);
  }
  if (!scored.length) return isGenericArticleRequest(question) ? candidates[0] : null;
  scored.sort((a, b) => b[0] - a[0]);
  return scored[0][1];
}

function rankContextArticleCandidates(
  candidates: Record<string, unknown>[],
  sessionContext: Record<string, unknown> | null,
): Record<string, unknown>[] {
  const preferredSection = String((sessionContext || {}).section || "");
  const lastTopic = String((sessionContext || {}).last_topic || "");
  const lastQuestion = String((sessionContext || {}).last_question || "").toLowerCase();
  const normTopic = lastTopic ? normalizeHeadline(lastTopic) : "";
  const scored: [number, number, Record<string, unknown>][] = [];
  for (const c of candidates) {
    const normH = normalizeHeadline(String(c.headline || ""));
    let topicScore = 0;
    if (normTopic && normH) {
      topicScore = tokenOverlapScore(normTopic, normH) + sequenceSimilarity(normTopic, normH);
    }
    const sectionScore = sectionPriorityScore(String(c.section || ""), preferredSection, lastQuestion);
    scored.push([sectionScore, topicScore, c]);
  }
  scored.sort((a, b) => b[0] - a[0] || b[1] - a[1]);
  return scored.map((s) => s[2]);
}

function sectionPriorityScore(section: string, preferred: string, lastQ: string): number {
  const norm = section.toLowerCase();
  if (preferred && norm === preferred.toLowerCase()) return 5;
  if (["iran", "war", "conflict"].some((k) => lastQ.includes(k))) {
    if (norm === "world") return 4.5;
    if (norm === "nation") return 3.5;
    if (norm === "business") return 2.5;
    if (["edit", "feature"].includes(norm)) return 0.5;
  }
  if (["budget", "middle class", "inflation"].some((k) => lastQ.includes(k))) {
    if (norm === "business") return 4.5;
    if (norm === "nation") return 2.5;
    return 1;
  }
  if (["world cup", "sports", "cricket"].some((k) => lastQ.includes(k))) {
    if (norm === "sports") return 4.5;
    return 0.5;
  }
  return 1;
}

// ---------------------------------------------------------------------------
// Followup detection
// ---------------------------------------------------------------------------
function isReferentialFollowup(question: string): boolean {
  const lowered = question.toLowerCase();
  return [
    /\bthat\b/, /\bthose\b/, /\bthese\b/, /\bthe one\b/, /\bthe same\b/,
    /\babove\b/, /\byou shared\b/, /\byou mentioned\b/, /\bwhat about\b/,
    /\bregarding that\b/, /\babout that\b/, /\bthis article\b/,
    /\bthat article\b/, /\bthat story\b/, /\bthose stories\b/, /\bit\b/,
  ].some((p) => p.test(lowered));
}

function isGenericArticleRequest(question: string): boolean {
  const lowered = question.toLowerCase();
  if (/\b(regarding|about|on)\b/.test(lowered)) return false;
  return [
    /\bany one article\b/, /\bany one of article\b/, /\bany article\b/,
    /\bshow any one\b.*\barticle\b/, /\bgive any one\b.*\barticle\b/, /\bgive any\b.*\barticle\b/,
  ].some((p) => p.test(lowered));
}

function contextValue(
  sessionContext: Record<string, unknown> | null,
  key: string,
  question: string,
): string | null {
  if (!sessionContext) return null;
  if (!shouldApplyContext(question)) return null;
  const value = sessionContext[key];
  return value ? String(value) : null;
}

function shouldApplyContext(question: string): boolean {
  const lowered = question.toLowerCase().trim();
  if (!lowered) return false;
  if (isGenericArticleRequest(question) || wantsExactArticleListing(question) || wantsArticleText(question)) return true;
  if (isReferentialFollowup(question)) return true;
  const resetPatterns = [
    /\bnews about\b/, /\barticles about\b/, /\bstories about\b/, /\bfind\b/,
    /\bshow me\b/, /\blist\b/, /\bwhich\b/, /\bwhat\b/, /\bhow many\b/,
    /\bcount\b/, /\bwho\b/, /\bwhen\b/,
  ];
  if (resetPatterns.some((p) => p.test(lowered))) return false;
  if ((lowered.match(/[a-z0-9]+/g) || []).length >= 4) return false;
  return true;
}

function filterValue(
  sessionFilters: Record<string, unknown> | null,
  key: string,
): string | null {
  if (!sessionFilters) return null;
  const value = sessionFilters[key];
  if (value == null || value === "" || value === "all") return null;
  return String(value);
}

function formatHistory(history: { role: string; content: string }[] | null): string {
  if (!history) return "";
  const lines: string[] = [];
  for (const item of history.slice(-8)) {
    const { role, content } = item;
    if (!["user", "assistant"].includes(role) || !content?.trim()) continue;
    lines.push(`${role === "user" ? "User" : "Assistant"}: ${content.trim()}`);
  }
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
function normalizeHeadline(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function tokenOverlapScore(left: string, right: string): number {
  const leftTokens = new Set(left.split(/\s+/).filter((t) => t.length >= 4));
  const rightTokens = new Set(right.split(/\s+/).filter((t) => t.length >= 4));
  if (!leftTokens.size || !rightTokens.size) return 0;
  let common = 0;
  for (const t of leftTokens) if (rightTokens.has(t)) common++;
  return common / Math.max(leftTokens.size, 1);
}

function sequenceSimilarity(a: string, b: string): number {
  if (a === b) return 1;
  const longer = a.length > b.length ? a : b;
  const shorter = a.length > b.length ? b : a;
  if (!longer.length) return 1;
  // Simple character-based similarity
  let matches = 0;
  const used = new Array(longer.length).fill(false);
  for (const ch of shorter) {
    for (let i = 0; i < longer.length; i++) {
      if (!used[i] && longer[i] === ch) { matches++; used[i] = true; break; }
    }
  }
  return (2 * matches) / (a.length + b.length);
}
