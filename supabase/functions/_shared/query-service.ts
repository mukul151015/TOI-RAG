import { embedTexts } from "./openai.ts";
import { rpc } from "./db.ts";
import {
  analyzeQuery,
  canonicalPersonName,
  expandPersonAliasTerms,
  expandSemanticQueries,
  isSectionCountQuery,
  normalizeUserQuery,
} from "./query-router.ts";
import type { QueryAnalysis, QueryResponse, RoutedQuery } from "./types.ts";

const GENERIC_TERMS = new Set([
  "india",
  "world",
  "stories",
  "story",
  "covered",
  "cover",
  "article",
  "articles",
  "which",
]);

const SPORTS_PHRASES = ["world cup", "t20 world cup", "world champions", "bcci reward"];
const BUDGET_PHRASES = ["budget", "middle class", "inflation", "price rise", "oil prices", "growth", "economists", "tax"];

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function numericValue(value: unknown): number {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
}

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

function overlapCount(query: string, fields: string[]): number {
  const terms = tokenize(query).filter((term) => term.length > 2 && !GENERIC_TERMS.has(term));
  if (!terms.length) {
    return 0;
  }
  const haystack = fields.join(" ").toLowerCase();
  let count = 0;
  for (const term of new Set(terms)) {
    if (haystack.includes(term)) {
      count += 1;
    }
  }
  return count;
}

function phraseBonus(query: string, fields: string[]): number {
  const haystack = fields.join(" ").toLowerCase();
  const lowered = query.toLowerCase();
  let bonus = 0;
  for (const phrase of SPORTS_PHRASES) {
    if (lowered.includes(phrase) && haystack.includes(phrase)) {
      bonus += 0.18;
    }
  }
  for (const phrase of BUDGET_PHRASES) {
    if (lowered.includes(phrase) && haystack.includes(phrase)) {
      bonus += 0.1;
    }
  }
  return Math.min(bonus, 0.36);
}

function exactEntityTermsForTopicCount(analysis: QueryAnalysis, routed: RoutedQuery): string[] {
  if (routed.intent !== "topic_count") {
    return [];
  }
  const out = new Set<string>();
  for (const person of analysis.entities.content_people ?? []) {
    for (const term of expandPersonAliasTerms(person)) {
      out.add(term);
    }
  }
  for (const org of analysis.entities.content_organizations ?? []) {
    out.add(org);
  }
  return Array.from(out);
}

function entityDisplayLabel(analysis: QueryAnalysis): string | null {
  const people = analysis.entities.content_people ?? [];
  if (people.length) {
    return canonicalPersonName(people[0]);
  }
  const orgs = analysis.entities.content_organizations ?? [];
  return orgs[0] ?? null;
}

function dedupeIds(rows: Array<Record<string, unknown>>): number[] {
  const ids = new Set<number>();
  for (const row of rows) {
    const id = numericValue(row.article_id);
    if (id) {
      ids.add(id);
    }
  }
  return Array.from(ids);
}

function computeConfidence(rows: Array<Record<string, unknown>>): number {
  const top = rows.slice(0, 5);
  if (!top.length) {
    return 0;
  }
  const total = top.reduce((sum, row) => sum + numericValue(row.similarity), 0);
  return Number((total / top.length).toFixed(4));
}

function passesTopicGuards(query: string, article: Record<string, unknown>, overlap: number): boolean {
  const lowered = query.toLowerCase();
  const headline = stringValue(article.headline).toLowerCase();
  const excerpt = stringValue(article.excerpt).toLowerCase();
  const body = `${headline} ${excerpt}`;
  if (lowered.includes("world cup")) {
    return /(world cup|t20|cricket|bcci|champion|trophy|india)/.test(body);
  }
  if (lowered.includes("budget")) {
    return /(budget|inflation|econom|tax|price|middle class|growth)/.test(body);
  }
  return overlap > 0;
}

function rankRows(
  vectorRows: Array<Record<string, unknown>>,
  keywordRows: Array<Record<string, unknown>>,
  articleRows: Map<number, Record<string, unknown>>,
  semanticQueries: string[],
): Array<Record<string, unknown>> {
  const ranked = new Map<number, Record<string, unknown>>();

  for (const row of vectorRows) {
    const articleId = numericValue(row.article_id);
    const article = articleRows.get(articleId);
    if (!article) {
      continue;
    }
    const overlap = Math.max(...semanticQueries.map((query) =>
      overlapCount(query, [
        stringValue(article.headline),
        stringValue(article.section),
        stringValue(article.excerpt),
        stringValue(row.chunk_text),
      ])
    ));
    const bonus = Math.max(...semanticQueries.map((query) =>
      phraseBonus(query, [
        stringValue(article.headline),
        stringValue(article.excerpt),
        stringValue(row.chunk_text),
      ])
    ));
    const similarity = numericValue(row.similarity);
    const score = similarity + Math.min(overlap * 0.03, 0.45) + bonus;
    const primaryQuery = semanticQueries[0] ?? "";
    if (!passesTopicGuards(primaryQuery, article, overlap)) {
      continue;
    }
    ranked.set(articleId, {
      article_id: articleId,
      similarity: score,
      chunk_text: stringValue(row.chunk_text),
    });
  }

  for (const row of keywordRows) {
    const articleId = numericValue(row.article_id);
    const article = articleRows.get(articleId);
    if (!article) {
      continue;
    }
    const overlap = Math.max(...semanticQueries.map((query) =>
      overlapCount(query, [
        stringValue(article.headline),
        stringValue(article.section),
        stringValue(article.excerpt),
        stringValue(row.excerpt),
      ])
    ));
    const bonus = Math.max(...semanticQueries.map((query) =>
      phraseBonus(query, [
        stringValue(article.headline),
        stringValue(article.excerpt),
      ])
    ));
    const lexical = numericValue(row.lexical_score);
    const score = 0.35 + Math.min(lexical, 1.2) + Math.min(overlap * 0.04, 0.4) + bonus;
    const primaryQuery = semanticQueries[0] ?? "";
    if (!passesTopicGuards(primaryQuery, article, overlap)) {
      continue;
    }
    const existing = ranked.get(articleId);
    if (!existing || numericValue(existing.similarity) < score) {
      ranked.set(articleId, {
        article_id: articleId,
        similarity: score,
        chunk_text: stringValue(row.excerpt),
      });
    }
  }

  return Array.from(ranked.values()).sort((a, b) => numericValue(b.similarity) - numericValue(a.similarity));
}

export async function runQuery(
  queryText: string,
  issueDate: string | null,
  limit: number,
  options: {
    edition?: string | null;
    section?: string | null;
    resultWindow?: number;
    routedOverride?: RoutedQuery | null;
  } = {},
): Promise<QueryResponse> {
  const analysis = await analyzeQuery(queryText, issueDate);
  const routed = options.routedOverride ?? analysis.routed;
  const edition = options.edition ?? routed.edition;
  const section = options.section ?? routed.section;
  const window = options.resultWindow ?? limit;

  if (isSectionCountQuery(queryText)) {
    const rows = await rpc<Record<string, unknown>>("fetch_section_counts_by_date", {
      issue_dt: routed.issue_date,
    });
    return {
      mode: "sql",
      filters: { issue_date: routed.issue_date, intent: routed.intent },
      results: rows,
      confidence_score: 1,
    };
  }

  if (routed.author) {
    const rows = await rpc<Record<string, unknown>>("fetch_sql_articles_filtered", {
      issue_dt: routed.issue_date,
      publication_filter: edition,
      section_filter: section,
      result_limit: window,
      author_filter: routed.author,
    });
    return {
      mode: "sql",
      filters: {
        ...routed,
        edition,
        section,
      },
      results: rows,
      confidence_score: rows.length ? 1 : 0,
    };
  }

  const exactTerms = exactEntityTermsForTopicCount(analysis, routed);
  if (routed.intent === "topic_count" && exactTerms.length) {
    const personTopic = Boolean((analysis.entities.content_people ?? []).length);
    const rows = await rpc<Record<string, unknown>>("keyword_search_filtered", {
      query_text: exactTerms.join(" OR "),
      issue_dt: routed.issue_date,
      publication_filter: edition,
      section_filter: section,
      match_count: window,
    });

    return {
      mode: "sql",
      filters: {
        ...routed,
        edition,
        section,
        retrieval_strategy: "exact_entity_mentions",
        entity_terms: exactTerms,
        entity_label: entityDisplayLabel(analysis),
        subject_strict: personTopic,
      },
      results: rows,
      confidence_score: rows.length ? 1 : 0,
    };
  }

  if (routed.mode === "sql") {
    const rows = await rpc<Record<string, unknown>>("fetch_sql_articles_filtered", {
      issue_dt: routed.issue_date,
      publication_filter: edition,
      section_filter: section,
      result_limit: window,
    });
    const countRows = await rpc<{ article_count: number | string }>("fetch_sql_article_count_filtered", {
      issue_dt: routed.issue_date,
      publication_filter: edition,
      section_filter: section,
    });
    return {
      mode: "sql",
      filters: {
        ...routed,
        edition,
        section,
        exact_article_count: numericValue(countRows[0]?.article_count) || rows.length,
      },
      results: rows,
      confidence_score: rows.length ? 1 : 0,
    };
  }

  const baseSemantic = routed.semantic_query ?? normalizeUserQuery(queryText);
  const semanticQueries = expandSemanticQueries(baseSemantic);
  const embeddings = await embedTexts(semanticQueries);
  const vectorRows: Array<Record<string, unknown>> = [];
  const keywordRows: Array<Record<string, unknown>> = [];
  const perQueryLimit = Math.min(Math.max(window, limit) * 3, 5000);
  const keywordLimit = Math.min(Math.max(window, limit) * 2, 1000);

  for (let index = 0; index < semanticQueries.length; index += 1) {
    const semanticQuery = semanticQueries[index];
    const embedding = embeddings[index];
    const vector = await rpc<Record<string, unknown>>("match_article_chunks_filtered", {
      query_embedding: embedding,
      issue_dt: routed.issue_date,
      publication_filter: edition,
      section_filter: section,
      match_count: perQueryLimit,
    });
    const keyword = await rpc<Record<string, unknown>>("keyword_search_filtered", {
      query_text: semanticQuery,
      issue_dt: routed.issue_date,
      publication_filter: edition,
      section_filter: section,
      match_count: keywordLimit,
    });
    vectorRows.push(...vector);
    keywordRows.push(...keyword);
  }

  const orderedIds = dedupeIds([...vectorRows, ...keywordRows]);
  const articleRows = new Map<number, Record<string, unknown>>();
  if (orderedIds.length) {
    const hydrated = await rpc<Record<string, unknown>>("fetch_articles_by_ids", {
      article_ids: orderedIds,
    });
    for (const row of hydrated) {
      articleRows.set(numericValue(row.id), row);
    }
  }

  const rankedRows = rankRows(vectorRows, keywordRows, articleRows, semanticQueries);
  const results = rankedRows.slice(0, window).map((row) => {
    const article = articleRows.get(numericValue(row.article_id));
    return {
      ...(article ?? {}),
      similarity: numericValue(row.similarity),
      matched_chunk: stringValue(row.chunk_text).slice(0, 300),
    };
  }).filter((row) => Object.keys(row).length);

  return {
    mode: routed.mode,
    filters: { ...routed, edition, section },
    results,
    confidence_score: computeConfidence(rankedRows),
  };
}
