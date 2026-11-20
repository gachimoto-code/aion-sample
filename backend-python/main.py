"""Aion 1.0 Instruct を Transformers で直接実行するバックエンド（llama.cpp 不要）。

公式リポジトリ（microsoft/Aion-1.0-Instruct-2.3B）に同梱された実装コードを
trust_remote_code=True で読み込み、PyTorch で推論する。
Mac (Apple Silicon) では MPS、なければ CPU で動く。

API は Express 版と互換:
  GET  /api/health  -> {"ok", "model", "ready", "detail"}
  POST /api/chat    -> SSE で data: {"delta": "..."} ... data: {"done": true}

起動: uvicorn main:app --port 3002  （または python main.py）
モデルは起動時にバックグラウンドで読み込む（初回は約4.6GBのダウンロードあり）。
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

MODEL_ID = os.getenv("MODEL_ID", "microsoft/Aion-1.0-Instruct-2.3B")
PORT = int(os.getenv("PORT", "3002"))
DTYPE_ENV = os.getenv("AION_DTYPE", "").lower()  # "float16" / "bfloat16" / "float32"

_state = {
    "status": "loading",  # loading / ready / error
    "model": None,
    "tokenizer": None,
    "device": None,
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
        # ダウンロード完了の目安: 合計が分かれば 99% 到達で読み込み段階へ。
        # 合計不明の場合のみ「サイズが数秒変わらない」を完了の目安にする
        # （回線が止まっているだけの状態を「読み込み中」と誤表示しないため）。
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
    _fetch_total_bytes()
    threading.Thread(target=_monitor_download, daemon=True).start()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
        if DTYPE_ENV == "float16":
            dtype = torch.float16
        elif DTYPE_ENV == "bfloat16":
            dtype = torch.bfloat16
        elif DTYPE_ENV == "float32":
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        _state.update(
            status="ready", model=model, tokenizer=tokenizer, device=device, error=None
        )
        _progress.update(stage="ready", percent=100.0)
        print(f"[aion-python] model loaded on {device} ({dtype})")
    except Exception as e:  # noqa: BLE001
        _state.update(status="error", error=f"{type(e).__name__}: {e}")
        _progress.update(stage="error")
        print(f"[aion-python] model load failed: {e}")
    finally:
        _load_done.set()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    threading.Thread(target=_load_model, daemon=True).start()
    yield


app = FastAPI(title="aion-sample backend (python)", lifespan=_lifespan)
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
        "backend": "python-transformers",
        "ready": _state["status"] == "ready",
        "detail": (
            f"device={_state['device']}"
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

    import torch
    from transformers import TextIteratorStreamer

    model = _state["model"]
    tokenizer = _state["tokenizer"]
    device = _state["device"]

    messages = [m.model_dump() for m in req.messages]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    input_ids = input_ids.to(device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300
    )
    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": req.maxTokens or 1024,
        "streamer": streamer,
    }
    if req.temperature and req.temperature > 0:
        gen_kwargs.update(
            do_sample=True, temperature=float(req.temperature), top_p=0.95
        )
    else:
        gen_kwargs.update(do_sample=False)

    def generate() -> None:
        with torch.no_grad():
            model.generate(**gen_kwargs)

    thread = threading.Thread(target=generate, daemon=True)
    thread.start()

    def event_stream():
        try:
            for text in streamer:
                yield f"data: {json.dumps({'delta': text}, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        yield 'data: {"done": true}\n\n'

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT)
