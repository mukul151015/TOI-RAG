const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";

if (!SUPABASE_URL) {
  throw new Error("SUPABASE_URL is not set");
}

if (!SUPABASE_SERVICE_ROLE_KEY) {
  throw new Error("SUPABASE_SERVICE_ROLE_KEY is not set");
}

function buildHeaders(extra: HeadersInit = {}): Headers {
  const headers = new Headers(extra);
  headers.set("apikey", SUPABASE_SERVICE_ROLE_KEY);
  headers.set("Authorization", `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`);
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  return headers;
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`PostgREST request failed (${response.status}): ${text}`);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return await response.json() as T;
}

export async function rpc<T>(fn: string, params: Record<string, unknown>): Promise<T[]> {
  const response = await fetch(`${SUPABASE_URL}/rest/v1/rpc/${fn}`, {
    method: "POST",
    headers: buildHeaders(),
    body: JSON.stringify(params),
  });
  return await parseResponse<T[]>(response);
}

export async function query<T>(table: string, qs: string): Promise<T[]> {
  const separator = qs ? `?${qs}` : "";
  const response = await fetch(`${SUPABASE_URL}/rest/v1/${table}${separator}`, {
    method: "GET",
    headers: buildHeaders({
      Accept: "application/json",
    }),
  });
  return await parseResponse<T[]>(response);
}

export async function insert<T>(table: string, data: Record<string, unknown> | Record<string, unknown>[]): Promise<T[]> {
  const response = await fetch(`${SUPABASE_URL}/rest/v1/${table}`, {
    method: "POST",
    headers: buildHeaders({
      Prefer: "return=representation",
    }),
    body: JSON.stringify(data),
  });
  return await parseResponse<T[]>(response);
}

export async function update<T>(table: string, filter: string, data: Record<string, unknown>): Promise<T[]> {
  const response = await fetch(`${SUPABASE_URL}/rest/v1/${table}?${filter}`, {
    method: "PATCH",
    headers: buildHeaders({
      Prefer: "return=representation",
    }),
    body: JSON.stringify(data),
  });
  return await parseResponse<T[]>(response);
}

export async function del(table: string, filter: string): Promise<void> {
  const response = await fetch(`${SUPABASE_URL}/rest/v1/${table}?${filter}`, {
    method: "DELETE",
    headers: buildHeaders(),
  });
  await parseResponse<void>(response);
}
