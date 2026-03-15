import { corsResponse, jsonResponse, errorResponse } from "../_shared/cors.ts";
import {
  getAuthenticatedSession,
  getAuthenticatedUser,
  updateSessionContext,
  logChatInteraction,
} from "../_shared/auth.ts";
import { answerQuestion } from "../_shared/chat-service.ts";

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return corsResponse();
  try {
    const session = await getAuthenticatedSession(req);
    if (!session) {
      const user = await getAuthenticatedUser(req);
      if (!user) return errorResponse("Authentication required", 401);
    }

    const { question, issue_date, session_filters, history, limit } = await req.json();

    const result = await answerQuestion(
      question,
      issue_date || null,
      limit || 10,
      session_filters || null,
      history || null,
      session?.session_context || null,
    );

    if (session) {
      await updateSessionContext(session.session_id, result.session_context);
      await logChatInteraction({
        userId: session.user_id,
        sessionId: session.session_id,
        question,
        answer: result.answer,
        issueDate: issue_date || null,
        mode: result.mode,
        sessionFilters: session_filters || null,
        citations: result.citations,
      });
    }

    return jsonResponse(result);
  } catch (err) {
    console.error("Chat error:", err);
    return errorResponse((err as Error).message);
  }
});
