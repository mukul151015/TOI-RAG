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
    const rows = await query<{ normalized_section: string | null }>(
      "sections",
      "select=normalized_section&normalized_section=not.is.null&order=normalized_section",
    );
    const deduped = Array.from(
      new Set(rows.map((row) => row.normalized_section).filter((value): value is string => Boolean(value))),
    );
    return jsonResponse(deduped);
  } catch (error) {
    return errorResponse(error instanceof Error ? error.message : "Catalog lookup failed", 500);
  }
});
