import os, subprocess, sys, pytest

HERE = os.path.dirname(__file__)


def test_gloo_2proc_runs(tmp_path):
    out = tmp_path / "o.json"
    p = subprocess.run([sys.executable, os.path.join(HERE, "..", "bench_allreduce.py"),
                        "--nprocs", "2", "--backend", "gloo", "--iters", "3",
                        "--warmup", "1", "--out", str(out)],
                       capture_output=True, text=True, timeout=240)
    assert p.returncode == 0, p.stderr
    assert "correctness check OK" in p.stdout
    assert out.exists()
