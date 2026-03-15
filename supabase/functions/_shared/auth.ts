import { getDb } from "./db.ts";

const SESSION_DAYS = 7;

export function getTokenFromRequest(req: Request): string | null {
  const header = req.headers.get("authorization");
  if (header?.startsWith("Bearer ")) {
    return header.slice(7);
  }
  return null;
}

export async function getAuthenticatedUser(
  req: Request,
): Promise<{ id: number; email: string } | null> {
  const token = getTokenFromRequest(req);
  if (!token) return null;
  const sql = getDb();
  const rows = await sql`
    select u.id, u.email
    from user_sessions s
    join app_users u on u.id = s.user_id
    where s.session_token = ${token}
      and s.expires_at > now()
  `;
  return rows.length ? (rows[0] as { id: number; email: string }) : null;
}

export async function getAuthenticatedSession(req: Request): Promise<{
  session_id: number;
  user_id: number;
  email: string;
  session_context: Record<string, unknown> | null;
} | null> {
  const token = getTokenFromRequest(req);
  if (!token) return null;
  const sql = getDb();
  const rows = await sql`
    select
      s.id as session_id,
      s.user_id,
      u.email,
      s.session_context
    from user_sessions s
    join app_users u on u.id = s.user_id
    where s.session_token = ${token}
      and s.expires_at > now()
  `;
  if (!rows.length) return null;
  const r = rows[0];
  return {
    session_id: r.session_id as number,
    user_id: r.user_id as number,
    email: r.email as string,
    session_context: r.session_context as Record<string, unknown> | null,
  };
}

export async function loginOrCreate(
  email: string,
  password: string,
): Promise<{ email: string; token: string }> {
  const normalizedEmail = email.trim().toLowerCase();
  if (!normalizedEmail) throw new Error("Email is required");

  const sql = getDb();
  const existing = await sql`
    select id, email, password_hash
    from app_users
    where email = ${normalizedEmail}
  `;

  let userId: number;

  if (existing.length === 0) {
    const passwordHash = await hashPassword(password);
    const inserted = await sql`
      insert into app_users (email, password_hash)
      values (${normalizedEmail}, ${passwordHash})
      returning id
    `;
    userId = inserted[0].id as number;
  } else {
    const user = existing[0];
    const valid = await verifyPassword(
      password,
      user.password_hash as string,
    );
    if (!valid) throw new Error("Invalid email or password");
    userId = user.id as number;
  }

  const token = crypto.randomUUID() + "-" + crypto.randomUUID();
  const expiresAt = new Date(Date.now() + SESSION_DAYS * 24 * 60 * 60 * 1000);
  await sql`
    insert into user_sessions (user_id, session_token, expires_at)
    values (${userId}, ${token}, ${expiresAt})
  `;

  return { email: normalizedEmail, token };
}

export async function logoutSession(token: string | null): Promise<void> {
  if (!token) return;
  const sql = getDb();
  await sql`delete from user_sessions where session_token = ${token}`;
}

export async function updateSessionContext(
  sessionId: number,
  sessionContext: Record<string, unknown> | null,
): Promise<void> {
  if (sessionContext == null) return;
  const sql = getDb();
  await sql`
    update user_sessions
    set session_context = ${JSON.stringify(sessionContext)}::jsonb
    where id = ${sessionId}
  `;
}

export async function logChatInteraction(params: {
  userId: number;
  sessionId: number;
  question: string;
  answer: string;
  issueDate: string | null;
  mode: string | null;
  sessionFilters: Record<string, unknown> | null;
  citations: unknown[] | null;
}): Promise<void> {
  const sql = getDb();
  await sql`
    insert into chat_interactions (
      user_id, session_id, user_question, system_answer,
      issue_date, mode, session_filters, citations
    )
    values (
      ${params.userId}, ${params.sessionId},
      ${params.question}, ${params.answer},
      ${params.issueDate}, ${params.mode},
      ${JSON.stringify(params.sessionFilters || {})}::jsonb,
      ${JSON.stringify(params.citations || [])}::jsonb
    )
  `;
}

async function hashPassword(password: string): Promise<string> {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const keyMaterial = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(password),
    "PBKDF2",
    false,
    ["deriveBits"],
  );
  const bits = await crypto.subtle.deriveBits(
    { name: "PBKDF2", salt, iterations: 200000, hash: "SHA-256" },
    keyMaterial,
    256,
  );
  const digest = new Uint8Array(bits);
  return hexEncode(salt) + "$" + hexEncode(digest);
}

async function verifyPassword(
  password: string,
  storedHash: string,
): Promise<boolean> {
  const parts = storedHash.split("$");
  if (parts.length !== 2) return false;
  const salt = hexDecode(parts[0]);
  const expected = hexDecode(parts[1]);
  const keyMaterial = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(password),
    "PBKDF2",
    false,
    ["deriveBits"],
  );
  const bits = await crypto.subtle.deriveBits(
    { name: "PBKDF2", salt, iterations: 200000, hash: "SHA-256" },
    keyMaterial,
    256,
  );
  const actual = new Uint8Array(bits);
  if (actual.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < actual.length; i++) diff |= actual[i] ^ expected[i];
  return diff === 0;
}

function hexEncode(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function hexDecode(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return bytes;
}
