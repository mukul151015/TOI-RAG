import {
  getAuthenticatedSession,
  logChatInteraction,
  updateSessionContext,
} from "../_shared/auth.ts";
import { answerQuestion } from "../_shared/chat-service.ts";
import { corsResponse, errorResponse, jsonResponse } from "../_shared/cors.ts";

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return corsResponse();
  }
  if (req.method !== "POST") {
    return errorResponse("Method not allowed", 405);
  }

  try {
    const session = await getAuthenticatedSession(req);
    if (!session) {
      return errorResponse("Authentication required", 401);
    }

    const body = await req.json();
    const question = typeof body.question === "string" ? body.question.trim() : "";
    if (!question) {
      return errorResponse("Question is required", 400);
    }

    const issueDate = typeof body.issue_date === "string" ? body.issue_date : null;
    const sessionFilters = typeof body.session_filters === "object" && body.session_filters !== null
      ? body.session_filters as Record<string, unknown>
      : null;
    const history = Array.isArray(body.history)
      ? body.history.filter((item): item is { role: string; content: string } =>
        typeof item === "object" &&
        item !== null &&
        typeof item.role === "string" &&
        typeof item.content === "string"
      )
      : null;
    const limit = typeof body.limit === "number" && body.limit > 0 ? Math.min(body.limit, 20) : 6;

    const result = await answerQuestion({
      question,
      issueDate,
      limit,
      sessionFilters,
      history,
      sessionContext: session.session_context ?? {},
    });

    await updateSessionContext(session.session_id, result.session_context ?? {});
    await logChatInteraction({
      userId: session.user_id,
      sessionId: session.session_id,
      question,
      answer: result.answer,
      issueDate,
      mode: result.mode,
      sessionFilters,
      citations: result.citations,
      traceData: result.debug_trace ?? null,
    });

    return jsonResponse(result);
  } catch (error) {
    return errorResponse(error instanceof Error ? error.message : "Chat request failed", 500);
  }
});
