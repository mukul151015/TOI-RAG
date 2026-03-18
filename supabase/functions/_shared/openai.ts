const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY") ?? "";

if (!OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY is not set");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function openAIRequest<T>(path: string, payload: Record<string, unknown>, attempts = 4): Promise<T> {
  let lastError: Error | null = null;
  for (let index = 0; index < attempts; index += 1) {
    try {
      const response = await fetch(`https://api.openai.com/v1/${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
        },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`OpenAI request failed (${response.status}): ${text}`);
      }
      return await response.json() as T;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      if (index === attempts - 1) {
        break;
      }
      await sleep(250 * (2 ** index));
    }
  }
  throw lastError ?? new Error("OpenAI request failed");
}

interface EmbeddingResponse {
  data: Array<{ embedding: number[] }>;
}

interface ChatCompletionResponse {
  choices: Array<{
    message?: {
      content?: string;
    };
  }>;
}

export async function embedTexts(texts: string[]): Promise<number[][]> {
  if (!texts.length) {
    return [];
  }
  const payload: Record<string, unknown> = {
    model: "text-embedding-3-small",
    input: texts,
    dimensions: 512,
  };
  const response = await openAIRequest<EmbeddingResponse>("embeddings", payload);
  return response.data.map((item) => item.embedding);
}

export async function chatCompletion(systemPrompt: string, userPrompt: string): Promise<string> {
  const payload = {
    model: "gpt-4.1-mini",
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    temperature: 0.2,
  };
  const response = await openAIRequest<ChatCompletionResponse>("chat/completions", payload);
  return response.choices[0]?.message?.content?.trim() ?? "";
}
