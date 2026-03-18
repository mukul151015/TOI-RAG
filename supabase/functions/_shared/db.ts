import postgres from "npm:postgres";

let _sql: ReturnType<typeof postgres> | null = null;

export function getDb(): ReturnType<typeof postgres> {
  if (!_sql) {
    const url = Deno.env.get("SUPABASE_DB_URL");
    if (!url) throw new Error("SUPABASE_DB_URL is not set");
    _sql = postgres(url, { max: 3, idle_timeout: 20 });
  }
  return _sql;
}

export type Row = Record<string, unknown>;
