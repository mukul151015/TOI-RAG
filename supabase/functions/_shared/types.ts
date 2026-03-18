export type QueryMode = "sql" | "semantic" | "hybrid";
export type QueryIntent =
  | "lookup"
  | "article_count"
  | "topic_count"
  | "fact_lookup"
  | "author_lookup"
  | "author_count";

export interface RoutedQuery {
  mode: QueryMode;
  intent: QueryIntent;
  issue_date: string | null;
  edition: string | null;
  section: string | null;
  author: string | null;
  semantic_query: string | null;
}

export interface QueryAnalysis {
  raw_query: string;
  normalized_query: string;
  lowered_query: string;
  routed: RoutedQuery;
  entities: Record<string, string[]>;
  ambiguous_edition: boolean;
}

export interface QueryResponse {
  mode: QueryMode;
  filters: Record<string, unknown>;
  results: Record<string, unknown>[];
  confidence_score: number;
}

export interface ChatResponseBody {
  answer: string;
  mode: QueryMode;
  citations: Record<string, unknown>[];
  confidence_score: number;
  session_context?: Record<string, unknown> | null;
  debug_trace?: Record<string, unknown>;
}

export interface SessionContext {
  edition?: string | null;
  section?: string | null;
  issue_date?: string | null;
  last_mode?: QueryMode | null;
  last_question?: string | null;
  story_titles?: string[];
  article_candidates?: Record<string, unknown>[];
  story_candidates?: Record<string, unknown>[];
  ambiguous_edition?: string | null;
  ambiguous_publications?: string[];
  [key: string]: unknown;
}

export interface AuthenticatedUser {
  id: number;
  email: string;
}

export interface AuthenticatedSession {
  session_id: number;
  user_id: number;
  email: string;
  session_context: SessionContext | null;
}
