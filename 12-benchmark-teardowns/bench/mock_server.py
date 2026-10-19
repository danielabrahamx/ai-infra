"""SYNTHETIC OpenAI-compatible mock server. For harness validation only.

Per-token delay grows with in-flight requests, to mimic a shared-capacity engine.
Numbers from this server say nothing about real hardware.
"""
import argparse
import asyncio
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


async def precise_sleep(d: float):
    """Sleep with sub-ms accuracy. asyncio.sleep is coarse (~16 ms) on Windows, so spin the last 20 ms."""
    end = time.perf_counter() + d
    while (left := end - time.perf_counter()) > 0:
        await asyncio.sleep(0 if left < 0.02 else left - 0.02)


def create_app(ttft_s=0.05, itl_s=0.01, slowdown=0.15, n_tokens=32) -> FastAPI:
    app = FastAPI()
    app.state.inflight = 0

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        n = min(int(body.get("max_tokens") or n_tokens), n_tokens)
        usage = (body.get("stream_options") or {}).get("include_usage")

        async def gen():
            app.state.inflight += 1
            try:
                await precise_sleep(ttft_s * (1 + slowdown * (app.state.inflight - 1)))
                for i in range(n):
                    if i:
                        await precise_sleep(itl_s * (1 + slowdown * (app.state.inflight - 1)))
                    c = {"object": "chat.completion.chunk", "created": int(time.time()), "model": body.get("model"),
                         "choices": [{"index": 0, "delta": {"content": f"tok{i} "}, "finish_reason": None}]}
                    yield f"data: {json.dumps(c)}\n\n"
                if usage:
                    yield "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": n}}) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                app.state.inflight -= 1
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8099)
    a = p.parse_args()
    uvicorn.run(create_app(), host="127.0.0.1", port=a.port, log_level="warning")
