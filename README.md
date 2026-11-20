# Aion 1.0 Instruct ローカルチャット（サンプル）

Microsoft の小型LLM「Aion 1.0 Instruct」（2.3B、日本語対応、MITライセンス）を、
Mac 上でローカル実行してブラウザからチャットするためのサンプルです。

バックエンドは2種類あり、フロントエンドは共通です。文章は端末の外に出ません。

```
A) Express版:  ブラウザ (React) → Express → llama-server (llama.cpp) + Aion GGUF
                  :5173            :3001        :8080
B) Python版:   ブラウザ (React) → FastAPI (Transformers が公式モデルを直接実行)
                  :5173            :3002
```

- frontend/ … Vite + React + TypeScript のチャットUI（ストリーミング表示、試し文ボタン付き）
- backend/ … Express + TypeScript。llama-server の OpenAI 互換APIへ中継し、SSEで返す
- backend-python/ … FastAPI + PyTorch/Transformers。公式モデルをそのまま実行する（llama.cpp 不要）
- モデル実行A … llama.cpp の llama-server が GGUF を読み込む（Apple Silicon は Metal で高速化）

## Mac でのセットアップ

前提: macOS (Apple Silicon)、Node.js 20+、Homebrew、cmake（`brew install cmake`）。

### 1. パッチ版 llama.cpp をビルドする

Aion は独自アーキテクチャ（`aion`）のため、上流の llama.cpp（Homebrew 版や Ollama 等）では
`unknown model architecture: 'aion'` となって読み込めません。
GGUF 配布元の AtomicChat が用意した対応パッチ（`llama.cpp/aion-port.patch`）を
指定コミットの llama.cpp に当ててビルドします（所要は数分程度）。

```bash
bash scripts/build-llama-mac.sh
```

成果物は `.llama-src/llama.cpp-<commit>/build/bin/llama-server` です。
この手順は 2026-11-20 に Linux 上で検証済みです（同パッチ・同コミットでビルドし、
日本語の要約・書き直し・質問応答が動作することを確認）。

### 2. モデル（GGUF）をダウンロードする

公式リポジトリ（microsoft/Aion-1.0-Instruct-2.3B）には PyTorch 形式しかないため、
llama.cpp 用にはコミュニティ変換の GGUF を使います。

- 配布元: https://huggingface.co/AtomicChat/Aion-1.0-Instruct-2.3B-GGUF
- 直接: https://huggingface.co/AtomicChat/Aion-1.0-Instruct-2.3B-GGUF/resolve/main/Aion-1.0-Instruct-2.3B-AD-Q4_K_M.gguf
- `Q4_K_M` のファイル（約1.4GB）を選んでダウンロードし、このフォルダ直下の `models/` に置く

```bash
mkdir -p models
# 上記の直接リンクからダウンロードして models/ へ
```

### 3. llama-server を起動する（ターミナル1）

```bash
.llama-src/llama.cpp-*/build/bin/llama-server -m models/Aion-1.0-Instruct-2.3B-AD-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 8192
```

ブラウザで http://127.0.0.1:8080 を開いて llama.cpp の画面が出れば起動OKです。

### 4. バックエンドを起動する（ターミナル2）

```bash
cd backend
cp .env.example .env   # 既定値のままでOK
npm install
npm run dev            # http://localhost:3001
```

### 5. フロントエンドを起動する（ターミナル3）

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

http://localhost:5173 を開くとチャット画面が出ます。
画面上部のステータスが「モデルサーバー接続OK」になっていれば準備完了です。
試し文ボタン（要約・書き直し・質問応答）か、入力欄から日本語で話しかけてください（⌘+Enter で送信）。

## Python版バックエンド（llama.cpp 不要）

Express版の代わりに、公式モデルを Transformers で直接実行する Python版も用意しています。
GGUF のダウンロードも、パッチ版 llama.cpp のビルドも不要です（モデルは初回起動時に
Hugging Face から自動でダウンロードされます。約4.6GB）。

```bash
cd backend-python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 既定値のままでOK
python main.py         # http://localhost:3002（モデル読み込みに初回は数分）
```

Python版バックエンドは、初回のモデルダウンロード／読み込みの進捗を /api/health の progress フィールドで返し、フロントエンドにプログレスバー（％と取得済み容量）で表示します。Express版は進捗を返さないため、状態文言のみの表示になります。

フロントエンドは接続先を frontend/.env で切り替えます。

```bash
cd frontend
cp .env.example .env
# .env の VITE_API_PROXY を http://localhost:3002 に書き換える
npm run dev
```

（Express版に繋ぐ場合は既定の http://localhost:3001 のままでOKです）

- Apple Silicon では MPS（float16）で実行します。CPU のみの環境では float32 で動きますが低速です
- 設定は backend-python/.env（PORT / MODEL_ID / AION_DTYPE）
- API は Express版と互換（/api/health、/api/chat の SSE 形式は同一）
- 検証状況（2026-11-20）：公式モデルの remote code・config・tokenizer・チャットテンプレートと
  FastAPI の配線は Linux 上で確認済み。実モデル（約4.6GB）での生成確認は Mac (MPS) 側でお願いします

## 設定（backend/.env）

| 変数 | 既定値 | 説明 |
|---|---|---|
| PORT | 3001 | バックエンドのポート |
| LLAMA_BASE_URL | http://127.0.0.1:8080 | llama-server のURL |
| MODEL_NAME | aion-1.0-instruct | llama-server へ送るモデル名ラベル（llama.cpp では任意の文字列で可） |

## API

- `GET /api/health` … バックエンドと llama-server の接続状態
- `POST /api/chat` … `{ "messages": [{"role":"user","content":"..."}], "temperature": 0.7, "maxTokens": 1024 }`
  を受け、llama-server の応答を SSE（`data: {"delta":"..."}` … `data: {"done":true}`）でストリーミング返却

## 注意

- Aion 1.0 Instruct は要約・書き直し・文書理解・質問応答を想定した小型モデルです。長い推論や複雑な指示には向きません。
- GGUF はコミュニティ変換版です。変換元は公式モデル（MIT）ですが、配布物の完全性は各自で確認してください。
- llama.cpp 本体へ上流対応が入れば、パッチ版ビルドは不要になります（2026-11-20 時点では未対応）。
- 2027年1月末までに、Microsoft が最適化版を Windows OS へ標準搭載する予定と発表しています（このサンプルは Mac 用です）。
