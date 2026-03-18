import { getTokenFromRequest, logoutSession } from "../_shared/auth.ts";
import { corsResponse, errorResponse, jsonResponse } from "../_shared/cors.ts";

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return corsResponse();
  }
  if (req.method !== "POST") {
    return errorResponse("Method not allowed", 405);
  }

  try {
    const token = getTokenFromRequest(req);
    if (!token) {
      return errorResponse("Authentication required", 401);
    }
    await logoutSession(token);
    return jsonResponse({ ok: true });
  } catch (error) {
    return errorResponse(error instanceof Error ? error.message : "Logout failed", 500);
  }
});
