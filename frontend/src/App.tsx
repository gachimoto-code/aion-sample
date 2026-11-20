import { useEffect, useRef, useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

type Health = {
  ok: boolean;
  model?: string;
  backend?: string;
  ready?: boolean;
  detail?: string;
};

const SAMPLE_PROMPTS = [
  "次の文を3行に要約して： 人工知能の発展により、文章の要約や翻訳といった作業は急速に自動化が進んでいる。特に大規模言語モデルは、大量のテキストを短時間で処理できるため、業務効率化の手段として注目されている。一方で、生成された文章の正確さを人間が確認する工程は依然として重要であり、完全な自動化には課題も残る。",
  "次の文を丁寧なビジネス文に書き直して： 例の件、明日までにやっといて。できたら今日中だと助かる。",
  "熊本県の阿蘇山について、2〜3文で教えて",
];

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((data: Health) => setHealth(data))
      .catch(() => setHealth({ ok: false }));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send(text: string) {
    const content = text.trim();
    if (!content || busy) return;
    setError(null);
    setInput("");
    const next: Message[] = [...messages, { role: "user", content }];
    setMessages([...next, { role: "assistant", content: "" }]);
    setBusy(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: next }),
      });
      if (!res.ok || !res.body) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(data.error ?? `HTTP ${res.status}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let acc = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;
          const json = JSON.parse(line.slice("data:".length).trim()) as {
            delta?: string;
            error?: string;
          };
          if (json.error) throw new Error(json.error);
          if (json.delta) {
            acc += json.delta;
            setMessages((prev) => {
              const copy = [...prev];
              copy[copy.length - 1] = { role: "assistant", content: acc };
              return copy;
            });
          }
        }
      }
      if (!acc) {
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: "assistant", content: "（応答が空でした）" };
          return copy;
        });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setBusy(false);
    }
  }

  const llamaOk = health?.ready === true;

  return (
    <div className="app">
      <header>
        <h1>Aion 1.0 Instruct ローカルチャット</h1>
        <p className="sub">
          Microsoft の小型LLM（2.3B）を Mac 上でローカル実行するサンプル。文章は端末の外に出ません。
        </p>
        <p className={llamaOk ? "status ok" : "status ng"}>
          {health === null
            ? "接続確認中…"
            : llamaOk
              ? `モデルサーバー接続OK（${health?.model ?? "aion"}）`
              : (health?.detail ??
                "モデルサーバーに接続できません。README の手順で起動してください")}
        </p>
      </header>

      <main>
        {messages.length === 0 && (
          <div className="samples">
            <p>試し文（クリックで送信）:</p>
            {SAMPLE_PROMPTS.map((p) => (
              <button key={p} type="button" onClick={() => void send(p)} disabled={!llamaOk || busy}>
                {p.slice(0, 42)}…
              </button>
            ))}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="role">{m.role === "user" ? "あなた" : "Aion"}</div>
            <div className="body">{m.content || "…"}</div>
          </div>
        ))}
        <div ref={bottomRef} />
      </main>

      {error && <p className="error">エラー: {error}</p>}

      <footer>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              void send(input);
            }
          }}
          placeholder="日本語で質問・要約・書き直しを頼んでみてください（⌘+Enter で送信）"
          rows={3}
        />
        <button type="button" onClick={() => void send(input)} disabled={busy || !input.trim()}>
          {busy ? "生成中…" : "送信"}
        </button>
      </footer>
    </div>
  );
}
