import { del, insert, query, rpc, update } from "./db.ts";
import type { AuthenticatedSession, AuthenticatedUser, SessionContext } from "./types.ts";

const SESSION_DAYS = 7;
const PBKDF2_ITERATIONS = 200_000;

interface AppUserRow {
  id: number;
  email: string;
  password_hash: string;
}

function hex(bytes: Uint8Array): string {
  return Array.from(bytes).map((value) => value.toString(16).padStart(2, "0")).join("");
}

function hexToBytes(value: string): Uint8Array {
  const out = new Uint8Array(value.length / 2);
  for (let index = 0; index < value.length; index += 2) {
    out[index / 2] = Number.parseInt(value.slice(index, index + 2), 16);
  }
  return out;
}

async function pbkdf2(password: string, salt: Uint8Array): Promise<Uint8Array> {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(password),
    { name: "PBKDF2" },
    false,
    ["deriveBits"],
  );
  const bits = await crypto.subtle.deriveBits(
    {
      name: "PBKDF2",
      hash: "SHA-256",
      iterations: PBKDF2_ITERATIONS,
      salt,
    },
    key,
    256,
  );
  return new Uint8Array(bits);
}

function constantTimeEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) {
    return false;
  }
  let out = 0;
  for (let index = 0; index < a.length; index += 1) {
    out |= a[index] ^ b[index];
  }
  return out === 0;
}

async function hashPassword(password: string): Promise<string> {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const digest = await pbkdf2(password, salt);
  return `${hex(salt)}$${hex(digest)}`;
}

async function verifyPassword(password: string, stored: string): Promise<boolean> {
  const [saltHex, hashHex] = stored.split("$");
  if (!saltHex || !hashHex) {
    return false;
  }
  const salt = hexToBytes(saltHex);
  const expected = hexToBytes(hashHex);
  const actual = await pbkdf2(password, salt);
  return constantTimeEqual(expected, actual);
}

export function getTokenFromRequest(req: Request): string | null {
  const auth = req.headers.get("Authorization") ?? req.headers.get("authorization");
  if (!auth) {
    return null;
  }
  const match = auth.match(/^Bearer\s+(.+)$/i);
  return match ? match[1].trim() : null;
}

export async function getAuthenticatedUser(req: Request): Promise<AuthenticatedUser | null> {
  const token = getTokenFromRequest(req);
  if (!token) {
    return null;
  }
  const rows = await rpc<{ user_id: number; email: string }>("auth_validate_session", { p_token: token });
  const row = rows[0];
  if (!row) {
    return null;
  }
  return {
    id: row.user_id,
    email: row.email,
  };
}

export async function getAuthenticatedSession(req: Request): Promise<AuthenticatedSession | null> {
  const token = getTokenFromRequest(req);
  if (!token) {
    return null;
  }
  const rows = await rpc<{
    session_id: number;
    user_id: number;
    email: string;
    session_context: SessionContext | null;
  }>("auth_get_session", { p_token: token });
  const row = rows[0];
  return row ?? null;
}

export async function loginOrCreate(email: string, password: string): Promise<{ ok: true; email: string; token: string }> {
  const normalizedEmail = email.trim().toLowerCase();
  if (!normalizedEmail) {
    throw new Error("Email is required");
  }
  if (!password || password.length < 6) {
    throw new Error("Password must be at least 6 characters");
  }

  const existing = await query<AppUserRow>(
    "app_users",
    `select=id,email,password_hash&email=eq.${encodeURIComponent(normalizedEmail)}&limit=1`,
  );

  let user = existing[0];
  if (!user) {
    const passwordHash = await hashPassword(password);
    const created = await insert<AppUserRow>("app_users", {
      email: normalizedEmail,
      password_hash: passwordHash,
    });
    user = created[0];
  } else {
    const valid = await verifyPassword(password, user.password_hash);
    if (!valid) {
      throw new Error("Invalid email or password");
    }
  }

  const expiresAt = new Date(Date.now() + SESSION_DAYS * 24 * 60 * 60 * 1000).toISOString();
  const token = `${crypto.randomUUID()}${crypto.randomUUID()}`;
  await insert("user_sessions", {
    user_id: user.id,
    session_token: token,
    expires_at: expiresAt,
    session_context: {},
  });

  return { ok: true, email: normalizedEmail, token };
}

export async function logoutSession(token: string): Promise<void> {
  await del("user_sessions", `session_token=eq.${encodeURIComponent(token)}`);
}

export async function updateSessionContext(sessionId: number, context: SessionContext | null): Promise<void> {
  if (!sessionId) {
    return;
  }
  await update("user_sessions", `id=eq.${sessionId}`, {
    session_context: context ?? {},
  });
}

export async function logChatInteraction(params: {
  userId: number;
  sessionId: number;
  question: string;
  answer: string;
  issueDate: string | null;
  mode: string | null;
  sessionFilters: Record<string, unknown> | null;
  citations: Record<string, unknown>[] | null;
  traceData: Record<string, unknown> | null;
}): Promise<void> {
  await insert("chat_interactions", {
    user_id: params.userId,
    session_id: params.sessionId,
    user_question: params.question,
    system_answer: params.answer,
    issue_date: params.issueDate,
    mode: params.mode,
    session_filters: params.sessionFilters ?? {},
    citations: params.citations ?? [],
    trace_data: params.traceData ?? {},
  });
}
