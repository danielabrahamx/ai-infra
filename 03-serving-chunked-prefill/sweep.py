"""Sweep chunk size (max_num_batched_tokens). Writes results.csv (+ plot.png if matplotlib).
Results are SIMULATED."""
import csv, copy
from sim import make_workload, simulate

BUDGETS = [256, 512, 1024, 2048, 4096, 8192, 32768]
base = make_workload()
rows = []
r = simulate(copy.deepcopy(base), 32768, chunked=False)
rows.append({"mode": "unchunked", "budget": 32768, **r})
for b in BUDGETS:
    rows.append({"mode": "chunked", "budget": b, **simulate(copy.deepcopy(base), b, True)})
with open("results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for x in rows:
    print(f"{x['mode']:9} B={x['budget']:6} TTFTp50={x['ttft_p50_ms']:9.0f} TTFTp99={x['ttft_p99_ms']:9.0f} "
          f"ITLp50={x['itl_p50_ms']:7.1f} ITLp99={x['itl_p99_ms']:8.1f} p99.9={x['itl_p999_ms']:8.1f} tok/s={x['throughput_tok_s']:7.1f}")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ch = [x for x in rows if x["mode"] == "chunked"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k in ("itl_p50_ms", "itl_p99_ms", "itl_p999_ms"):
        ax[0].plot([x["budget"] for x in ch], [x[k] for x in ch], marker="o", label=k)
        ax[0].axhline(rows[0][k], ls="--", color="gray")
    ax[0].set_xscale("log"); ax[0].set_xlabel("chunk budget (tokens)"); ax[0].set_ylabel("ms"); ax[0].legend()
    ax[0].set_title("ITL (dashed = unchunked) SIMULATED")
    ax[1].plot([x["budget"] for x in ch], [x["ttft_p99_ms"] for x in ch], marker="o")
    ax[1].set_xscale("log"); ax[1].set_title("TTFT p99 ms SIMULATED"); ax[1].set_xlabel("chunk budget")
    plt.tight_layout(); plt.savefig("plot.png"); print("wrote plot.png")
except ImportError:
    print("matplotlib missing: skipped plot (optional)")
