"""FastAPI gateway. POST /v1/chat/completions (OpenAI-compatible, stream or not)."""
import json
import time
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from . import classifier as clf
from .config import Settings, from_env
from .metrics import Metrics
from .tracing import span


def prompt_text(messages: list) -> str:
    """Join message contents (handles str and list-of-parts content)."""
    parts = []
    for m in messages:
        c = m.get("content", "")
        parts.append(c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict)))
    return "\n".join(parts)


def delta_text(line: str) -> str | None:
    """Return the content of one SSE data line, or None if it has no token."""
    if not line.startswith("data:") or line.strip() == "data: [DONE]":
        return None
    try:
        ch = json.loads(line[5:])["choices"][0]
        return ch.get("delta", {}).get("content") or None
    except (ValueError, KeyError, IndexError):
        return None


def create_app(settings: Settings | None = None, classify: clf.Classifier = clf.heuristic) -> FastAPI:
    s = settings or from_env()
    m = Metrics()
    app = FastAPI(title="llm-gateway")
    app.state.metrics, app.state.settings = m, s

    def pick(score: float, override: str | None) -> str:
        if override in ("local", "frontier"):
            return override
        return "frontier" if score >= s.threshold else "local"

    def up(route: str):
        return s.local if route == "local" else s.frontier

    def auth(u):
        return {"Authorization": f"Bearer {u.api_key}"} if u.api_key else {}

    def check_slo(route: str, ttft: float | None, itls: list[float]):
        slo = s.slo.get(route)
        if not slo:
            return
        if ttft is not None and ttft > slo.ttft_ms:
            m.slo_viol.labels(route, "ttft").inc()
        if itls and sum(itls) / len(itls) > slo.itl_ms:
            m.slo_viol.labels(route, "itl").inc()

    @app.get("/healthz")
    def health():
        return {"ok": True}

    @app.get("/metrics")
    def metrics():
        return Response(m.render(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        score = clf_score(body)
        route = pick(score, req.headers.get("x-gateway-route"))
        stream = bool(body.get("stream"))
        order = [route] + ([("frontier" if route == "local" else "local")] if s.fallback else [])
        t0 = time.perf_counter()
        last_err = "no upstream"
        for i, r in enumerate(order):
            u = up(r)
            payload = {**body, "model": u.model}
            if i:
                m.fallbacks.labels(order[i - 1]).inc()
            info = {"x-gateway-route": r, "x-gateway-score": str(score)}
            client = httpx.AsyncClient(timeout=s.timeout_s)
            url = f"{u.base_url}/chat/completions"
            try:
                if not stream:
                    with span("gateway.upstream", route=r, score=score):
                        resp = await client.post(url, json=payload, headers=auth(u))
                    await client.aclose()
                    if resp.status_code >= 500:
                        raise RuntimeError(f"upstream {resp.status_code}")
                    m.requests.labels(r, str(resp.status_code)).inc()
                    m.total.labels(r).observe((time.perf_counter() - t0) * 1000)
                    return JSONResponse(resp.json(), status_code=resp.status_code, headers=info)
                # Streaming: open upstream first so we can still fall back before sending bytes.
                resp = await client.send(client.build_request("POST", url, json=payload, headers=auth(u)), stream=True)
                if resp.status_code >= 500:
                    await resp.aclose()
                    raise RuntimeError(f"upstream {resp.status_code}")
                return StreamingResponse(relay(resp, client, r, score, t0), media_type="text/event-stream", headers=info)
            except Exception as ex:  # noqa: BLE001 - any upstream failure triggers fallback
                await client.aclose()
                last_err = str(ex) or type(ex).__name__
                m.requests.labels(r, "error").inc()
        return JSONResponse({"error": {"message": f"all upstreams failed: {last_err}"}}, status_code=502)

    def clf_score(body: dict) -> float:
        score = classify(prompt_text(body.get("messages", [])))
        m.score.observe(score)
        return score

    async def relay(resp: httpx.Response, client, route: str, score: float, t0: float) -> AsyncIterator[bytes]:
        """Pass SSE through while timing TTFT and inter-token gaps."""
        ttft, last, itls = None, None, []
        with span("gateway.stream", route=route, score=score) as sp:
            try:
                async for line in resp.aiter_lines():
                    now = time.perf_counter()
                    if delta_text(line) is not None:
                        if ttft is None:
                            ttft = (now - t0) * 1000
                            m.ttft.labels(route).observe(ttft)
                        else:
                            gap = (now - last) * 1000
                            itls.append(gap)
                            m.itl.labels(route).observe(gap)
                        last = now
                    yield (line + "\n").encode()
            finally:
                await resp.aclose()
                await client.aclose()
                m.requests.labels(route, str(resp.status_code)).inc()
                m.total.labels(route).observe((time.perf_counter() - t0) * 1000)
                check_slo(route, ttft, itls)
                sp.set_attribute("ttft_ms", ttft or -1)

    return app


def main():  # pragma: no cover
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8080)


if __name__ == "__main__":  # pragma: no cover
    main()
