"""Pareto plot: latency (x) vs throughput (y). One curve per results file.

A point is Pareto-optimal if no other point has lower-or-equal latency AND higher-or-equal throughput
(with at least one strict). The frontier is drawn as a step line.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

METRICS = {"ttft": ("ttft_ms", "TTFT"), "itl": ("itl_ms", "ITL"), "e2e": ("e2e_ms", "End-to-end")}


def points(result: dict, metric: str = "e2e", pct: str = "p95") -> list[tuple]:
    """Return (latency_ms, throughput_tok_s, concurrency) per level."""
    key = METRICS[metric][0]
    return [(lv[key][pct], lv["throughput_tok_s"], lv["concurrency"]) for lv in result["levels"] if lv[key][pct] is not None]


def pareto_front(pts: list[tuple]) -> list[tuple]:
    """Non-dominated points, sorted by latency ascending."""
    front = []
    for p in pts:
        dominated = any(q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]) for q in pts)
        if not dominated:
            front.append(p)
    return sorted(front)


def plot(paths: list[str], out: str, metric: str = "e2e", pct: str = "p95"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_validation = False
    for path in paths:
        res = json.load(open(path))
        any_validation |= bool(res.get("validation_only"))
        pts = sorted(points(res, metric, pct))
        name = res.get("label") or res["target"]["model"]
        line, = ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", alpha=0.5, label=name)
        fr = pareto_front(pts)
        ax.step([p[0] for p in fr], [p[1] for p in fr], where="post", color=line.get_color(), lw=2.5)
        for lat, thr, c in pts:
            ax.annotate(f"c={c}", (lat, thr), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel(f"{METRICS[metric][1]} latency {pct} (ms)")
    ax.set_ylabel("Throughput (output tokens/s)")
    ax.set_title("Latency vs throughput (thick line = Pareto frontier)")
    if any_validation:
        ax.text(0.5, 0.5, "HARNESS VALIDATION ONLY\nsynthetic mock - not a performance claim", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="red", alpha=0.3, rotation=20)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--metric", choices=list(METRICS), default="e2e")
    p.add_argument("--pct", choices=["mean", "p50", "p95", "p99"], default="p95")
    a = p.parse_args()
    plot(a.results, a.out, a.metric, a.pct)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
