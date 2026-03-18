import { getAuthenticatedUser } from "../_shared/auth.ts";
import { corsResponse, errorResponse, jsonResponse } from "../_shared/cors.ts";
import { query } from "../_shared/db.ts";

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
      return errorResponse("Authentication required", 401);
    }
    const rows = await query<{ publication_name: string }>(
      "publications",
      "select=id,publication_name&order=publication_name",
    );
    return jsonResponse(rows.map((row) => row.publication_name));
  } catch (error) {
    return errorResponse(error instanceof Error ? error.message : "Catalog lookup failed", 500);
  }
});
