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


def _load_model() -> None:
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
        print(f"[aion-python] model loaded on {device} ({dtype})")
    except Exception as e:  # noqa: BLE001
        _state.update(status="error", error=f"{type(e).__name__}: {e}")
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
