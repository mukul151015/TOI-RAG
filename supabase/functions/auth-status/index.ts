import { corsResponse, jsonResponse, errorResponse } from "../_shared/cors.ts";
import { getAuthenticatedUser } from "../_shared/auth.ts";

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return corsResponse();
  try {
    const user = await getAuthenticatedUser(req);
    if (!user) return jsonResponse({ authenticated: false });
    return jsonResponse({ authenticated: true, email: user.email });
  } catch (err) {
    return errorResponse((err as Error).message);
  }
});
