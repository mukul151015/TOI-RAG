import { corsResponse, jsonResponse, errorResponse } from "../_shared/cors.ts";
import { getAuthenticatedUser } from "../_shared/auth.ts";
import { fetchPublicationCatalog } from "../_shared/query-router.ts";

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return corsResponse();
  try {
    const user = await getAuthenticatedUser(req);
    if (!user) return errorResponse("Authentication required", 401);
    const catalog = await fetchPublicationCatalog();
    return jsonResponse(catalog.map((r) => r.publication_name));
  } catch (err) {
    return errorResponse((err as Error).message);
  }
});
