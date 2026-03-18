export type Row = Record<string, unknown>;

function requireEnv(name: string): string {
  const value = Deno.env.get(name);
  if (!value) throw new Error(`${name} is not set`);
  return value;
}

function baseUrl(): string {
  return `${requireEnv("SUPABASE_URL").replace(/\/+$/, "")}/rest/v1`;
}

function serviceHeaders(extra: Record<string, string> = {}): HeadersInit {
  const key = requireEnv("SUPABASE_SERVICE_ROLE_KEY");
  return {
    apikey: key,
    Authorization: `Bearer ${key}`,
    ...extra,
  };
}

async function parseResponse<T>(resp: Response, label: string): Promise<T> {
  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(`${label} ${resp.status}: ${detail}`);
  }
  if (resp.status === 204) {
    return [] as unknown as T;
  }
  return await resp.json() as T;
}

export async function rpc<T = Row>(
  fn: string,
  params: Record<string, unknown>,
): Promise<T[]> {
  const resp = await fetch(`${baseUrl()}/rpc/${fn}`, {
    method: "POST",
    headers: serviceHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(params),
  });
  return parseResponse<T[]>(resp, `rpc/${fn}`);
}

export async function query<T = Row>(table: string, qs = ""): Promise<T[]> {
  const suffix = qs ? `?${qs}` : "";
  const resp = await fetch(`${baseUrl()}/${table}${suffix}`, {
    headers: serviceHeaders(),
  });
  return parseResponse<T[]>(resp, table);
}

export async function insert<T = Row>(
  table: string,
  data: Record<string, unknown> | Record<string, unknown>[],
): Promise<T[]> {
  const resp = await fetch(`${baseUrl()}/${table}`, {
    method: "POST",
    headers: serviceHeaders({
      "Content-Type": "application/json",
      Prefer: "return=representation",
    }),
    body: JSON.stringify(data),
  });
  return parseResponse<T[]>(resp, `insert/${table}`);
}

export async function update<T = Row>(
  table: string,
  filter: string,
  data: Record<string, unknown>,
): Promise<T[]> {
  const resp = await fetch(`${baseUrl()}/${table}?${filter}`, {
    method: "PATCH",
    headers: serviceHeaders({
      "Content-Type": "application/json",
      Prefer: "return=representation",
    }),
    body: JSON.stringify(data),
  });
  return parseResponse<T[]>(resp, `update/${table}`);
}

export async function del(table: string, filter: string): Promise<void> {
  const resp = await fetch(`${baseUrl()}/${table}?${filter}`, {
    method: "DELETE",
    headers: serviceHeaders({ Prefer: "return=minimal" }),
  });
  await parseResponse(resp, `delete/${table}`);
}
