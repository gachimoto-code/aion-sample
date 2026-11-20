"""Aion 1.0 Instruct を ONNX Runtime GenAI で実行するバックエンド（llama.cpp / PyTorch 不要）。

公式の ONNX 版（microsoft/Aion-1.0-Instruct-2.3B-ONNX-WebGPU、int4 量子化）を
onnxruntime-genai で読み込んで推論する。genai_config.json の provider_options は
空なので、Mac では CPU 実行になる（モデルは約1.3GBと小さく、量子化済み）。

API は Express 版・Python (Transformers) 版と互換:
  GET  /api/health  -> {"ok", "model", "ready", "detail", "progress"}
  POST /api/chat    -> SSE で data: {"delta": "..."} ... data: {"done": true}

起動: uvicorn main:app --port 3003  （または python main.py）
モデルは起動時にバックグラウンドで読み込む（初回は約1.3GBのダウンロードあり）。
MODEL_DIR を指定すると、ダウンロード済みのローカルフォルダをそのまま使う。
"""

import json
import os
import threading
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "microsoft/Aion-1.0-Instruct-2.3B-ONNX-WebGPU")
MODEL_DIR = os.getenv("MODEL_DIR", "")
PORT = int(os.getenv("PORT", "3003"))
MAX_CONTEXT = 4096  # genai_config.json の search.max_length に合わせる

_state = {
    "status": "loading",  # loading / ready / error
    "model": None,
    "tokenizer": None,
    "model_dir": None,
    "error": None,
}
_load_done = threading.Event()

# フロントエンドの進捗表示用（/api/health の progress フィールド）
_progress = {
    "stage": "downloading",  # downloading / loading / ready / error
    "downloadedBytes": 0,
    "totalBytes": None,
    "percent": None,
}


def _cache_dir_bytes() -> int:
    """HF キャッシュ内のこのモデルの実ファイル合計サイズ（ダウンロード進捗の近似）。"""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        root = os.path.join(HF_HUB_CACHE, "models--" + MODEL_ID.replace("/", "--"))
        total = 0
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
        return total
    except Exception:
        return 0


def _fetch_total_bytes() -> None:
    """HF API でリポジトリの全ファイル合計サイズを取得し、進捗の分母にする。"""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(MODEL_ID, files_metadata=True)
        total = sum(s.size or 0 for s in (info.siblings or []))
        if total > 0:
            _progress["totalBytes"] = total
    except Exception:
        pass  # 取得できなくても読み込み自体は続ける（percent は null のまま）


def _monitor_download() -> None:
    import time

    last_size = -1
    stable_polls = 0
    while not _load_done.is_set():
        size = _cache_dir_bytes()
        _progress["downloadedBytes"] = size
        total = _progress["totalBytes"]
        if total:
            _progress["percent"] = round(min(100.0, size / total * 100), 1)
        if total and size >= total * 0.99:
            if _progress["stage"] == "downloading":
                _progress["stage"] = "loading"
        elif not total and size == last_size and size > 0:
            stable_polls += 1
            if stable_polls >= 3 and _progress["stage"] == "downloading":
                _progress["stage"] = "loading"
        else:
            stable_polls = 0
        last_size = size
        time.sleep(1)


def _load_model() -> None:
    try:
        import onnxruntime_genai as og

        if MODEL_DIR and os.path.isdir(MODEL_DIR):
            model_dir = MODEL_DIR
            _progress.update(stage="loading", percent=100.0)
        else:
            _fetch_total_bytes()
            threading.Thread(target=_monitor_download, daemon=True).start()
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(MODEL_ID)
            _progress.update(stage="loading")

        model = og.Model(model_dir)
        tokenizer = og.Tokenizer(model)
        _state.update(
            status="ready", model=model, tokenizer=tokenizer,
            model_dir=model_dir, error=None,
        )
        _progress.update(stage="ready", percent=100.0)
        print(f"[aion-onnx] model ready: {model_dir}")
    except Exception as e:  # noqa: BLE001
        _state.update(status="error", error=f"{type(e).__name__}: {e}")
        _progress.update(stage="error")
        print(f"[aion-onnx] model load failed: {e}")
    finally:
        _load_done.set()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    threading.Thread(target=_load_model, daemon=True).start()
    yield


app = FastAPI(title="aion-sample backend (onnx)", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    maxTokens: Optional[int] = 1024


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "model": MODEL_ID,
        "backend": "python-onnx-genai",
        "ready": _state["status"] == "ready",
        "detail": (
            "onnxruntime-genai (CPU)"
            if _state["status"] == "ready"
            else _state["error"]
            if _state["status"] == "error"
            else "モデル読み込み中（初回はダウンロードのため数分かかります）"
        ),
        "progress": dict(_progress),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    if _state["status"] == "loading":
        _load_done.wait(timeout=600)
    if _state["status"] != "ready":
        return JSONResponse(
            status_code=502,
            content={"error": "モデルの読み込みに失敗しました", "detail": _state["error"]},
        )

    import onnxruntime_genai as og

    model = _state["model"]
    tokenizer = _state["tokenizer"]

    messages = [m.model_dump() for m in req.messages]
    prompt = tokenizer.apply_chat_template(
        messages=json.dumps(messages, ensure_ascii=False),
        add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(prompt)
    max_new = req.maxTokens or 1024
    temperature = req.temperature if req.temperature is not None else 0.7

    params = og.GeneratorParams(model)
    if temperature and temperature > 0:
        params.set_search_options(
            do_sample=True, temperature=float(temperature), max_length=min(MAX_CONTEXT, len(input_ids) + max_new)
        )
    else:
        params.set_search_options(
            do_sample=False, max_length=min(MAX_CONTEXT, len(input_ids) + max_new)
        )

    def event_stream():
        try:
            generator = og.Generator(model, params)
            generator.append_tokens(input_ids)
            stream = tokenizer.create_stream()
            while not generator.is_done():
                generator.generate_next_token()
                for token in generator.get_next_tokens():
                    text = stream.decode(token)
                    if text:
                        yield f"data: {json.dumps({'delta': text}, ensure_ascii=False)}\n\n"
            yield 'data: {"done": true}\n\n'
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'error': f'生成に失敗しました: {e}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT)
