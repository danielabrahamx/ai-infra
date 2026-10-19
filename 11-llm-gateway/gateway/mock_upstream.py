"""Tiny OpenAI-compatible mock. Streams N tokens with configurable delays."""
import asyncio
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def create_mock(name: str = "mock", ttft_s: float = 0.0, itl_s: float = 0.0, n_tokens: int = 8, fail: bool = False) -> FastAPI:
    app = FastAPI()
    app.state.calls = []

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        app.state.calls.append({"body": body, "auth": req.headers.get("authorization")})
        if fail:
            return JSONResponse({"error": "boom"}, status_code=503)
        toks = [f"{name}-t{i} " for i in range(n_tokens)]
        if not body.get("stream"):
            return {"id": "x", "object": "chat.completion", "model": body.get("model"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(toks)}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": n_tokens, "total_tokens": 5 + n_tokens}}

        async def gen():
            await asyncio.sleep(ttft_s)
            for i, t in enumerate(toks):
                if i:
                    await asyncio.sleep(itl_s)
                chunk = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": body.get("model"),
                         "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
