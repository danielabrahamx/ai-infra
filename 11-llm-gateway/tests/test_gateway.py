"""Tests run against REAL mock upstreams started on localhost ports (uvicorn in threads)."""
import socket
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

from gateway import classifier as clf
from gateway.app import create_app
from gateway.config import SLO, Settings, Upstream
from gateway.mock_upstream import create_mock


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, app):
        self.port = free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        while not self.server.started:
            time.sleep(0.01)
        return self

    def __exit__(self, *a):
        self.server.should_exit = True
        self.thread.join(5)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/v1"


def make_gateway(local, frontier, **kw):
    s = Settings(Upstream("local", local.url, "local-model"), Upstream("frontier", frontier.url, "big-model", "sk-test"),
                 slo={"local": SLO(ttft_ms=50, itl_ms=10), "frontier": SLO()}, **kw)
    return TestClient(create_app(s))


def msg(text, **extra):
    return {"messages": [{"role": "user", "content": text}], **extra}


HARD = "Analyze step by step and compare the trade-offs of this architecture, then prove why: ```def f(x): return x**2``` ∫ x dx?"


def metric(client, name):
    text = client.get("/metrics").text
    return sum(float(l.rsplit(" ", 1)[1]) for l in text.splitlines() if l.startswith(name))


# ---- classifier (no network) ----
def test_classifier_simple_vs_complex():
    assert clf.heuristic("What is the capital of France?") < 0.5
    assert clf.heuristic(HARD) >= 0.5


def test_classifier_is_pluggable_and_bounded():
    assert clf.constant(0.9)("x") == 0.9
    assert 0 <= clf.heuristic("word " * 5000 + HARD) <= 1.0


# ---- routing ----
def test_simple_goes_local_hard_goes_frontier():
    with Server(create_mock("L")) as L, Server(create_mock("F")) as F:
        c = make_gateway(L, F)
        r = c.post("/v1/chat/completions", json=msg("Hi, what is 2 plus 2?"))
        assert r.headers["x-gateway-route"] == "local" and "L-t0" in r.json()["choices"][0]["message"]["content"]
        r = c.post("/v1/chat/completions", json=msg(HARD))
        assert r.headers["x-gateway-route"] == "frontier" and "F-t0" in r.json()["choices"][0]["message"]["content"]


def test_model_rewritten_and_api_key_only_to_frontier():
    lm, fm = create_mock("L"), create_mock("F")
    with Server(lm) as L, Server(fm) as F:
        c = make_gateway(L, F)
        c.post("/v1/chat/completions", json=msg("hi", model="user-model"))
        c.post("/v1/chat/completions", json=msg(HARD))
        assert lm.state.calls[0]["body"]["model"] == "local-model" and lm.state.calls[0]["auth"] is None
        assert fm.state.calls[0]["body"]["model"] == "big-model" and fm.state.calls[0]["auth"] == "Bearer sk-test"


def test_override_header():
    with Server(create_mock("L")) as L, Server(create_mock("F")) as F:
        c = make_gateway(L, F)
        r = c.post("/v1/chat/completions", json=msg("hi"), headers={"x-gateway-route": "frontier"})
        assert r.headers["x-gateway-route"] == "frontier"


def test_fallback_when_upstream_fails():
    with Server(create_mock("L", fail=True)) as L, Server(create_mock("F")) as F:
        c = make_gateway(L, F)
        r = c.post("/v1/chat/completions", json=msg("hi"))
        assert r.status_code == 200 and r.headers["x-gateway-route"] == "frontier"
        assert metric(c, "gateway_fallbacks_total{") == 1
        # streaming fallback too
        with c.stream("POST", "/v1/chat/completions", json=msg("hi", stream=True)) as rs:
            assert rs.headers["x-gateway-route"] == "frontier" and "F-t0" in "".join(rs.iter_text())


def test_all_upstreams_down_gives_502():
    with Server(create_mock("L", fail=True)) as L, Server(create_mock("F", fail=True)) as F:
        assert make_gateway(L, F).post("/v1/chat/completions", json=msg("hi")).status_code == 502


def test_no_fallback_flag():
    with Server(create_mock("L", fail=True)) as L, Server(create_mock("F")) as F:
        assert make_gateway(L, F, fallback=False).post("/v1/chat/completions", json=msg("hi")).status_code == 502


# ---- streaming latency + metrics + SLO ----
def test_streaming_passthrough_and_ttft_itl_metrics():
    with Server(create_mock("L", ttft_s=0.2, itl_s=0.05, n_tokens=6)) as L, Server(create_mock("F")) as F:
        c = make_gateway(L, F)
        with c.stream("POST", "/v1/chat/completions", json=msg("hi", stream=True)) as r:
            body = "".join(r.iter_text())
        assert body.count("L-t") == 6 and body.rstrip().endswith("[DONE]")
        # TTFT >= mock delay (200 ms); mean ITL >= mock delay (50 ms). Lower bounds only: no flaky upper bounds.
        ttft_sum = metric(c, 'gateway_ttft_ms_sum{route="local"}')
        itl_sum, itl_n = metric(c, 'gateway_itl_ms_sum{route="local"}'), metric(c, 'gateway_itl_ms_count{route="local"}')
        assert ttft_sum >= 190 and itl_n == 5 and itl_sum / itl_n >= 45
        assert metric(c, 'gateway_requests_total{route="local",status="200"}') == 1


def test_slo_violation_counted():
    # local SLO in make_gateway is ttft<=50ms; mock TTFT is 200 ms -> violation.
    with Server(create_mock("L", ttft_s=0.2, itl_s=0.0)) as L, Server(create_mock("F")) as F:
        c = make_gateway(L, F)
        with c.stream("POST", "/v1/chat/completions", json=msg("hi", stream=True)) as r:
            "".join(r.iter_text())
        assert metric(c, 'gateway_slo_violations_total{route="local",slo="ttft"}') == 1


def test_metrics_and_health_endpoints():
    with Server(create_mock("L")) as L, Server(create_mock("F")) as F:
        c = make_gateway(L, F)
        assert c.get("/healthz").json() == {"ok": True}
        c.post("/v1/chat/completions", json=msg("hi"))
        assert "gateway_complexity_score_count 1.0" in c.get("/metrics").text


def test_tracing_noop_or_real():
    from gateway.tracing import span
    with span("x", a=1) as s:
        s.set_attribute("k", "v")
