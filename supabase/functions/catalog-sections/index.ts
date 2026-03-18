import { corsResponse, jsonResponse, errorResponse } from "../_shared/cors.ts";
import { getAuthenticatedUser } from "../_shared/auth.ts";
import { fetchSectionCatalog } from "../_shared/query-router.ts";

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return corsResponse();
  try {
    const user = await getAuthenticatedUser(req);
    if (!user) return errorResponse("Authentication required", 401);
    const sections = await fetchSectionCatalog();
    return jsonResponse(sections);
  } catch (err) {
    return errorResponse((err as Error).message);
  }
});
