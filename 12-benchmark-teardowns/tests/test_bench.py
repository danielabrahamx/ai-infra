"""Tests: unit math, plus a real end-to-end run of the harness against the local mock server."""
import json
import socket
import threading
import time

import pytest
import uvicorn

from bench import cost, harness, mock_server, plot, validate


def test_percentile_nearest_rank():
    v = list(range(1, 101))
    assert harness.pct(v, 50) == 50 and harness.pct(v, 95) == 95 and harness.pct(v, 99) == 99
    assert harness.pct([], 50) is None and harness.pct([7], 99) == 7


def test_prompt_is_deterministic():
    assert harness.make_prompt(20, 1) == harness.make_prompt(20, 1) != harness.make_prompt(20, 2)


def test_cost_math():
    # $3.6/hr at 1000 tok/s = 3.6 / 3.6e6 tok = $1 per 1M tokens
    assert cost.usd_per_million_tokens(3.6, 1000) == pytest.approx(1.0)
    assert cost.usd_per_million_tokens(3.6, 1000, n_gpus=2) == pytest.approx(2.0)
    assert cost.usd_per_million_tokens(3.6, 500, utilization=0.5) == pytest.approx(4 * 1.0)
    assert cost.usd_per_million_tokens(3.6, 0) is None


def test_pareto_front():
    pts = [(10, 100, 1), (20, 300, 2), (30, 250, 4), (40, 400, 8)]   # (latency, throughput, c)
    assert [p[2] for p in plot.pareto_front(pts)] == [1, 2, 8]       # c=4 is dominated by c=2


@pytest.fixture(scope="module")
def server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(mock_server.create_app(ttft_s=0.03, itl_s=0.01, n_tokens=16),
                                        host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    while not srv.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{port}/v1"
    srv.should_exit = True
    t.join(5)


def test_harness_end_to_end(server, tmp_path):
    out = tmp_path / "r.json"
    harness.main(["--base-url", server, "--model", "mock", "--concurrency", "1,4", "--requests-per-level", "8",
                  "--max-tokens", "16", "--warmup", "1", "--validation-only", "--label", "test", "--out", str(out)])
    res = json.loads(out.read_text())
    assert validate.validate(res) == []
    assert res["validation_only"] is True and res["environment"]["python"]
    l1, l4 = res["levels"]
    assert l1["errors"] == l4["errors"] == 0 and l1["output_tokens"] == 8 * 16
    assert l1["ttft_ms"]["p50"] >= 29            # mock sleeps 30 ms before first token
    assert l1["itl_ms"]["mean"] >= 9             # mock sleeps 10 ms between tokens
    assert l4["throughput_tok_s"] > l1["throughput_tok_s"]   # concurrency helps on the mock
    png = tmp_path / "p.png"
    plot.plot([str(out)], str(png))
    assert png.stat().st_size > 1000


def test_harness_counts_errors():
    # Nothing listens on port 9: every request must be recorded as an error, not crash the run.
    args = harness.parse(["--base-url", "http://127.0.0.1:9/v1", "--model", "m", "--out", "x", "--timeout", "2"])
    import asyncio
    lv = asyncio.run(harness.run_level(args, 2, 4, 0))
    assert lv["errors"] == 4 and lv["throughput_tok_s"] == 0


def test_validate_rejects_bad():
    assert validate.validate({"schema_version": 1})
