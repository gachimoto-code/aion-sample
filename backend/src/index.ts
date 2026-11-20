import cors from "cors";
import dotenv from "dotenv";
import express, { type Request, type Response } from "express";

dotenv.config();

const PORT = Number(process.env.PORT ?? 3001);
const LLAMA_BASE_URL = (process.env.LLAMA_BASE_URL ?? "http://127.0.0.1:8080").replace(/\/$/, "");
const MODEL_NAME = process.env.MODEL_NAME ?? "aion-1.0-instruct";

type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

// Health: is the backend up, and can it reach llama-server?
app.get("/api/health", async (_req: Request, res: Response) => {
  let llama: { ok: boolean; models?: string[]; error?: string } = { ok: false };
  try {
    const r = await fetch(`${LLAMA_BASE_URL}/v1/models`, {
      signal: AbortSignal.timeout(3000),
    });
    if (r.ok) {
      const data = (await r.json()) as { data?: Array<{ id?: string }> };
      llama = {
        ok: true,
        models: (data.data ?? []).map((m) => m.id ?? "").filter(Boolean),
      };
    } else {
      llama = { ok: false, error: `llama-server returned ${r.status}` };
    }
  } catch (e) {
    llama = { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
  res.json({
    ok: true,
    model: MODEL_NAME,
    backend: "express-llama.cpp",
    ready: llama.ok,
    detail: llama.ok
      ? `llama-server: ${LLAMA_BASE_URL}`
      : (llama.error ?? "llama-server に接続できません"),
    llamaBaseUrl: LLAMA_BASE_URL,
    llama,
  });
});

// Chat: forwards the conversation to llama-server and streams the reply
// back to the browser as server-sent events: data: {"delta":"..."} lines,
// finished by data: {"done":true}.
app.post("/api/chat", async (req: Request, res: Response) => {
  const body = req.body as {
    messages?: ChatMessage[];
    temperature?: number;
    maxTokens?: number;
  };
  const messages = Array.isArray(body?.messages) ? body.messages : [];
  if (messages.length === 0) {
    res.status(400).json({ error: "messages is required" });
    return;
  }

  let upstream: globalThis.Response;
  try {
    upstream = await fetch(`${LLAMA_BASE_URL}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODEL_NAME,
        messages,
        temperature: typeof body.temperature === "number" ? body.temperature : 0.7,
        max_tokens: typeof body.maxTokens === "number" ? body.maxTokens : 1024,
        stream: true,
      }),
    });
  } catch (e) {
    res.status(502).json({
      error: `llama-server に接続できません (${LLAMA_BASE_URL})`,
      detail: e instanceof Error ? e.message : String(e),
    });
    return;
  }

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text().catch(() => "");
    res.status(502).json({
      error: `llama-server がエラーを返しました (${upstream.status})`,
      detail: text.slice(0, 500),
    });
    return;
  }

  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice("data:".length).trim();
        if (payload === "[DONE]") continue;
        try {
          const json = JSON.parse(payload) as {
            choices?: Array<{ delta?: { content?: string } }>;
          };
          const delta = json.choices?.[0]?.delta?.content;
          if (delta) {
            res.write(`data: ${JSON.stringify({ delta })}\n\n`);
          }
        } catch {
          // Ignore malformed chunks (keep-alive comments etc.)
        }
      }
    }
  } catch (e) {
    res.write(
      `data: ${JSON.stringify({ error: e instanceof Error ? e.message : String(e) })}\n\n`,
    );
  }
  res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
  res.end();
});

app.listen(PORT, () => {
  console.log(`aion-sample backend listening on http://localhost:${PORT}`);
  console.log(`llama-server upstream: ${LLAMA_BASE_URL} (model label: ${MODEL_NAME})`);
});
