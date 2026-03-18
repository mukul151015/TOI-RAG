import { corsResponse, jsonResponse, errorResponse } from "../_shared/cors.ts";
import { loginOrCreate } from "../_shared/auth.ts";

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return corsResponse();
  try {
    const { email, password } = await req.json();
    const result = await loginOrCreate(email, password);
    return jsonResponse({ ok: true, email: result.email, token: result.token });
  } catch (err) {
    const msg = (err as Error).message;
    const status = msg.includes("Invalid email") ? 401 : msg.includes("required") ? 400 : 500;
    return errorResponse(msg, status);
  }
});
