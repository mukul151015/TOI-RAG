import { corsResponse, jsonResponse, errorResponse } from "../_shared/cors.ts";
import { getTokenFromRequest, logoutSession } from "../_shared/auth.ts";

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return corsResponse();
  try {
    const token = getTokenFromRequest(req);
    await logoutSession(token);
    return jsonResponse({ ok: true });
  } catch (err) {
    return errorResponse((err as Error).message);
  }
});
