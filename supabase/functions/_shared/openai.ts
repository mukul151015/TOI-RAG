const EMBEDDING_MODEL = "text-embedding-3-small";
const EMBEDDING_DIMENSIONS = 512;
const CHAT_MODEL = "gpt-4.1-mini";
const MAX_RETRIES = 4;
const RETRY_BASE_DELAY = 1.5;

function apiKey(): string {
  const key = Deno.env.get("OPENAI_API_KEY");
  if (!key) throw new Error("OPENAI_API_KEY is not set");
  return key;
}

export async function embedTexts(texts: string[]): Promise<number[][]> {
  let lastError: Error | null = null;
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const resp = await fetch("https://api.openai.com/v1/embeddings", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey()}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: EMBEDDING_MODEL,
          input: texts,
          dimensions: EMBEDDING_DIMENSIONS,
        }),
      });
      if (!resp.ok) {
        const body = await resp.text();
        throw new Error(`OpenAI embeddings ${resp.status}: ${body}`);
      }
      const data = await resp.json();
      return data.data.map((item: { embedding: number[] }) => item.embedding);
    } catch (err) {
      lastError = err as Error;
      if (attempt < MAX_RETRIES) {
        await new Promise((r) => setTimeout(r, RETRY_BASE_DELAY * attempt * 1000));
      }
    }
  }
  throw lastError!;
}

export async function chatCompletion(
  systemPrompt: string,
  userPrompt: string,
): Promise<string> {
  const resp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey()}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
    }),
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`OpenAI chat ${resp.status}: ${body}`);
  }
  const data = await resp.json();
  return data.choices[0].message.content;
}
