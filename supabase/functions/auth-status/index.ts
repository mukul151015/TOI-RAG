import { getAuthenticatedUser } from "../_shared/auth.ts";
import { corsResponse, jsonResponse, errorResponse } from "../_shared/cors.ts";

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return corsResponse();
  }
  if (req.method !== "GET") {
    return errorResponse("Method not allowed", 405);
  }

  try {
    const user = await getAuthenticatedUser(req);
    if (!user) {
      return jsonResponse({ authenticated: false });
    }
    return jsonResponse({ authenticated: true, email: user.email });
  } catch (error) {
    return errorResponse(error instanceof Error ? error.message : "Auth status failed", 500);
  }
});
