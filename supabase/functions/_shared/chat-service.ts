import { chatCompletion } from "./openai.ts";
import { rpc } from "./db.ts";
import { analyzeQuery, isBroadListingQuery, isSectionCountQuery } from "./query-router.ts";
import { runQuery } from "./query-service.ts";
import type { ChatResponseBody, QueryResponse, SessionContext } from "./types.ts";

const SYSTEM_PROMPT =
  "You answer questions from the TOI e-paper dataset. Use only the provided article results. Sound natural and concise, like a helpful newsroom analyst.";

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function arrayValue(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null) : [];
}

function requestedArticleCount(question: string): number | null {
  const match = question.toLowerCase().match(/\b(?:give|show|list)\s+me\s+(\d+)\s+articles?\b/);
  return match ? Number(match[1]) : null;
}

function wantsArticleText(question: string): boolean {
  const lowered = question.toLowerCase();
  return (
    lowered.includes("give me one article") ||
    lowered.includes("show me one article") ||
    lowered.includes("that article") ||
    lowered.includes("the article text") ||
    lowered.includes("full article") ||
    lowered.includes("above conversation")
  );
}

function asksForContext(question: string): boolean {
  const lowered = question.toLowerCase();
  return lowered.includes("in which context") || lowered.includes("in what context") || lowered.includes("what were they about");
}

function shouldShowReferences(question: string): boolean {
  const lowered = question.toLowerCase();
  return /\breference\b|\bcitation\b|\bshow sources\b|\bgive me one article\b|\bshow me one article\b/.test(lowered);
}

function isReferentialFollowup(question: string): boolean {
  const lowered = question.toLowerCase();
  return (
    lowered.includes("that article") ||
    lowered.includes("those stories") ||
    lowered.includes("above conversation") ||
    lowered.includes("from the above conversation") ||
    lowered.includes("which edition did you use")
  );
}

function buildCitations(results: Record<string, unknown>[], limit: number): Record<string, unknown>[] {
  return results.slice(0, limit).map((item) => ({
    article_id: item.external_article_id ?? item.id,
    headline: item.headline ?? "Untitled",
    edition: item.edition ?? "Unknown edition",
    section: item.section ?? "Unknown section",
    issue_date: item.issue_date ?? "Unknown date",
    reference_text: item.excerpt ?? item.matched_chunk ?? "No supporting excerpt available.",
  }));
}

function groupUniqueStories(results: Record<string, unknown>[]): Array<{ headline: string; count: number; section: string; excerpt: string }> {
  const grouped = new Map<string, { headline: string; count: number; section: string; excerpt: string }>();
  for (const item of results) {
    const headline = stringValue(item.headline) || "Untitled";
    const key = headline.toLowerCase();
    const current = grouped.get(key);
    if (current) {
      current.count += 1;
      continue;
    }
    grouped.set(key, {
      headline,
      count: 1,
      section: stringValue(item.section) || "Unknown section",
      excerpt: stringValue(item.excerpt) || stringValue(item.matched_chunk),
    });
  }
  return Array.from(grouped.values()).sort((a, b) => b.count - a.count || a.headline.localeCompare(b.headline));
}

function formatSectionCounts(results: Record<string, unknown>[]): string {
  if (!results.length) {
    return "No section counts matched the requested issue date.";
  }
  const lines = results.map((item, index) => `${index + 1}. ${stringValue(item.section) || "Unclassified"}: ${item.article_count ?? 0}`);
  const top = results[0];
  return `${stringValue(top.section) || "Unclassified"} had the most articles with ${top.article_count ?? 0} pieces.\n\nFull section ranking:\n${lines.join("\n")}`;
}

function formatCountAnswer(question: string, queryResponse: QueryResponse, sessionContext: SessionContext | null): string {
  const exactCount = typeof queryResponse.filters.exact_article_count === "number"
    ? queryResponse.filters.exact_article_count
    : null;
  const total = exactCount ?? queryResponse.results.length;
  if (!total) {
    return "I could not find any matching articles in the current dataset.";
  }
  const entityLabel = stringValue(queryResponse.filters.entity_label) || "that topic";
  const stories = groupUniqueStories(queryResponse.results).slice(0, 4);
  if (asksForContext(question) && stories.length) {
    const storyText = stories.map((story) => story.headline).join("; ");
    return `I found ${total} articles directly focused on ${entityLabel}. They were mainly about ${storyText}.`;
  }
  const edition = stringValue(sessionContext?.edition);
  const editionText = edition ? ` in the ${edition} edition` : "";
  return `There were ${total} matching articles${editionText}.`;
}

function formatArticleTextAnswer(source: Record<string, unknown>): string {
  const headline = stringValue(source.headline) || "Untitled";
  const edition = stringValue(source.edition) || "Unknown edition";
  const section = stringValue(source.section) || "Unknown section";
  const excerpt = stringValue(source.excerpt) || stringValue(source.reference_text) || stringValue(source.matched_chunk) || "No excerpt available.";
  return `${headline}\n${edition} | ${section}\n\n${excerpt}`;
}

function formatArticleListing(queryResponse: QueryResponse, requestedCount: number | null): ChatResponseBody {
  const rows = queryResponse.results;
  if (!rows.length) {
    return {
      answer: "No matching articles were found for that request.",
      mode: queryResponse.mode,
      citations: [],
      confidence_score: queryResponse.confidence_score,
    };
  }
  const display = requestedCount ? rows.slice(0, requestedCount) : rows;
  const lines = [`I found ${rows.length} matching articles.`];
  if (requestedCount) {
    lines[0] = `I found ${rows.length} matching articles. Here are ${Math.min(requestedCount, rows.length)} worth looking at.`;
  }
  display.forEach((item, index) => {
    lines.push(`${index + 1}. ${stringValue(item.headline)} | ${stringValue(item.edition)} | ${stringValue(item.section)} | ${stringValue(item.issue_date)}`);
  });
  return {
    answer: lines.join("\n"),
    mode: queryResponse.mode,
    citations: buildCitations(display, display.length),
    confidence_score: queryResponse.confidence_score,
  };
}

async function formatStorySummary(question: string, queryResponse: QueryResponse): Promise<ChatResponseBody> {
  const uniqueStories = groupUniqueStories(queryResponse.results).slice(0, 10);
  if (!uniqueStories.length) {
    return {
      answer: "I could not find enough evidence to summarize the matching stories.",
      mode: queryResponse.mode,
      citations: [],
      confidence_score: queryResponse.confidence_score,
    };
  }
  const evidence = uniqueStories.map((story, index) =>
    `${index + 1}. ${story.headline}\nSection: ${story.section}\nExcerpt: ${story.excerpt}`
  ).join("\n\n");
  const userPrompt = [
    `Question: ${question}`,
    "",
    "Summarize the main distinct stories covered below. Merge duplicate coverage across editions. Focus on themes, not a raw list unless necessary.",
    "",
    evidence,
  ].join("\n");
  const answer = await chatCompletion(SYSTEM_PROMPT, userPrompt);
  return {
    answer,
    mode: queryResponse.mode,
    citations: buildCitations(queryResponse.results, 6),
    confidence_score: queryResponse.confidence_score,
  };
}

function buildSessionContext(
  question: string,
  queryResponse: QueryResponse,
  previous: SessionContext | null,
  overrides: Partial<SessionContext> = {},
): SessionContext {
  const results = queryResponse.results;
  const uniqueStories = groupUniqueStories(results).slice(0, 10);
  return {
    ...(previous ?? {}),
    ...overrides,
    last_question: question,
    last_mode: queryResponse.mode,
    issue_date: stringValue(queryResponse.filters.issue_date) || (previous?.issue_date ?? null),
    edition: stringValue(queryResponse.filters.edition) || (previous?.edition ?? null),
    section: stringValue(queryResponse.filters.section) || (previous?.section ?? null),
    story_titles: uniqueStories.map((story) => story.headline),
    article_candidates: results.slice(0, 10),
    story_candidates: uniqueStories.map((story) => ({
      headline: story.headline,
      section: story.section,
      reference_text: story.excerpt,
    })),
  };
}

function articleCandidateFromContext(sessionContext: SessionContext | null): Record<string, unknown> | null {
  const candidates = arrayValue(sessionContext?.article_candidates);
  if (candidates.length) {
    return candidates[0];
  }
  const stories = arrayValue(sessionContext?.story_candidates);
  return stories[0] ?? null;
}

function buildAnswerPrompt(
  question: string,
  queryResponse: QueryResponse,
  history: Array<{ role: string; content: string }> | null,
  limit: number,
): string {
  const evidence = queryResponse.results.slice(0, limit).map((item, index) => [
    `[${index + 1}] ${stringValue(item.headline) || "Untitled"}`,
    `Edition: ${stringValue(item.edition) || "Unknown edition"}`,
    `Section: ${stringValue(item.section) || "Unknown section"}`,
    `Issue date: ${stringValue(item.issue_date) || "Unknown date"}`,
    `Excerpt: ${stringValue(item.excerpt) || stringValue(item.matched_chunk) || "No excerpt available."}`,
  ].join("\n")).join("\n\n");

  const historyText = (history ?? []).slice(-6).map((entry) => `${entry.role}: ${entry.content}`).join("\n");
  return [
    `Question: ${question}`,
    "",
    "Conversation context:",
    historyText || "None",
    "",
    "Use only the evidence below:",
    evidence || "No supporting evidence.",
    "",
    "Answer directly. If the evidence is weak or incomplete, say so briefly.",
  ].join("\n");
}

export async function answerQuestion(params: {
  question: string;
  issueDate: string | null;
  limit: number;
  sessionFilters?: Record<string, unknown> | null;
  history?: Array<{ role: string; content: string }> | null;
  sessionContext?: SessionContext | null;
}): Promise<ChatResponseBody> {
  const { question, issueDate, limit, sessionFilters, history, sessionContext } = params;
  const analysis = await analyzeQuery(question, issueDate);
  const routed = analysis.routed;

  if (wantsArticleText(question) && isReferentialFollowup(question)) {
    const cached = articleCandidateFromContext(sessionContext ?? null);
    if (cached) {
      return {
        answer: formatArticleTextAnswer(cached),
        mode: (sessionContext?.last_mode as "sql" | "semantic" | "hybrid" | undefined) ?? "sql",
        citations: [cached],
        confidence_score: 1,
        session_context: sessionContext ?? {},
        debug_trace: { answer_path: "context_article_text" },
      };
    }
  }

  if (question.toLowerCase().includes("which edition did you use")) {
    const edition = stringValue(sessionContext?.edition) || "no edition filter";
    return {
      answer: `I used ${edition}.`,
      mode: (sessionContext?.last_mode as "sql" | "semantic" | "hybrid" | undefined) ?? "sql",
      citations: [],
      confidence_score: 1,
      session_context: sessionContext ?? {},
      debug_trace: { answer_path: "edition_usage" },
    };
  }

  const edition = stringValue(sessionFilters?.edition) || stringValue(sessionContext?.edition) || routed.edition || null;
  const section = stringValue(sessionFilters?.section) || stringValue(sessionContext?.section) || routed.section || null;

  if (analysis.ambiguous_edition && routed.edition) {
    const matches = await rpc<{ publication_name: string; article_count: number }>("fetch_matching_pubs", {
      issue_dt: routed.issue_date,
      edition_term: routed.edition,
    });
    if (matches.length > 1) {
      const names = matches.map((item) => `${item.publication_name} (${item.article_count})`).join(", ");
      return {
        answer: `I found multiple matching editions for "${routed.edition}": ${names}. Please specify which one you want.`,
        mode: "sql",
        citations: [],
        confidence_score: 1,
        session_context: {
          ...(sessionContext ?? {}),
          ambiguous_edition: routed.edition,
          ambiguous_publications: matches.map((item) => item.publication_name),
        },
        debug_trace: { answer_path: "ambiguous_edition" },
      };
    }
  }

  const resultWindow = isSectionCountQuery(question)
    ? 5000
    : (isBroadListingQuery(question) ? 5000 : Math.max(limit, 24));

  const queryResponse = await runQuery(question, issueDate, limit, {
    edition,
    section,
    resultWindow,
    routedOverride: {
      ...routed,
      edition,
      section,
    },
  });

  if (isSectionCountQuery(question)) {
    const nextContext = buildSessionContext(question, queryResponse, sessionContext ?? null);
    return {
      answer: formatSectionCounts(queryResponse.results),
      mode: queryResponse.mode,
      citations: [],
      confidence_score: queryResponse.confidence_score,
      session_context: nextContext,
      debug_trace: { answer_path: "section_count" },
    };
  }

  if (routed.intent === "article_count" || routed.intent === "topic_count" || routed.intent === "author_count") {
    const nextContext = buildSessionContext(question, queryResponse, sessionContext ?? null);
    return {
      answer: formatCountAnswer(question, queryResponse, sessionContext ?? null),
      mode: queryResponse.mode,
      citations: shouldShowReferences(question) ? buildCitations(queryResponse.results, limit) : [],
      confidence_score: queryResponse.confidence_score,
      session_context: nextContext,
      debug_trace: { answer_path: "count_answer" },
    };
  }

  if (wantsArticleText(question)) {
    const first = queryResponse.results[0];
    const nextContext = buildSessionContext(question, queryResponse, sessionContext ?? null);
    if (!first) {
      return {
        answer: "I could not find an article to show from the current result set.",
        mode: queryResponse.mode,
        citations: [],
        confidence_score: queryResponse.confidence_score,
        session_context: nextContext,
        debug_trace: { answer_path: "article_text_empty" },
      };
    }
    return {
      answer: formatArticleTextAnswer(first),
      mode: queryResponse.mode,
      citations: buildCitations([first], 1),
      confidence_score: queryResponse.confidence_score,
      session_context: nextContext,
      debug_trace: { answer_path: "article_text" },
    };
  }

  const articleCount = requestedArticleCount(question);
  if (isBroadListingQuery(question) && articleCount) {
    const response = formatArticleListing(queryResponse, articleCount);
    response.session_context = buildSessionContext(question, queryResponse, sessionContext ?? null);
    response.debug_trace = { answer_path: "article_listing" };
    return response;
  }

  if (isBroadListingQuery(question)) {
    const response = await formatStorySummary(question, queryResponse);
    response.session_context = buildSessionContext(question, queryResponse, sessionContext ?? null);
    response.debug_trace = { answer_path: "story_summary" };
    return response;
  }

  const answer = await chatCompletion(
    SYSTEM_PROMPT,
    buildAnswerPrompt(question, queryResponse, history ?? null, limit),
  );
  const nextContext = buildSessionContext(question, queryResponse, sessionContext ?? null);
  return {
    answer,
    mode: queryResponse.mode,
    citations: shouldShowReferences(question) ? buildCitations(queryResponse.results, limit) : buildCitations(queryResponse.results, Math.min(limit, 6)),
    confidence_score: queryResponse.confidence_score,
    session_context: nextContext,
    debug_trace: {
      answer_path: "llm_answer",
      result_count: queryResponse.results.length,
      filters: queryResponse.filters,
    },
  };
}
