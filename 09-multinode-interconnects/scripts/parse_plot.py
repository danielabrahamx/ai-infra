"""Parse sweep JSONL or nccl-tests stdout; plot latency and busbw."""
import argparse, json, re, sys
from collections import defaultdict

# nccl-tests row: size count type redop root time algbw busbw #wrong time algbw busbw #wrong
_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+\w+\s+\w+\s+(-?\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)")


def parse_nccl_tests(text: str, label: str = "nccl-tests"):
    """Return rows (out-of-place columns) from nccl-tests stdout."""
    rows = []
    for line in text.splitlines():
        m = _ROW.match(line)
        if m:
            rows.append(dict(backend="nccl", topology="unknown", label=label, bytes=int(m[1]),
                             time_us=float(m[4]), algbw_GBps=float(m[5]), busbw_GBps=float(m[6])))
    return rows


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def group(rows):
    """Group rows into curves keyed by (backend, topology, label, world_size)."""
    g = defaultdict(list)
    for r in rows:
        g[(r["backend"], r["topology"], r.get("label", ""), r.get("world_size", 0))].append(r)
    return {k: sorted(v, key=lambda r: r["bytes"]) for k, v in g.items()}


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, v in group(rows).items():
        name = "/".join(str(p) for p in k if p)
        ax[0].plot([r["bytes"] for r in v], [r["time_us"] for r in v], "o-", label=name)
        ax[1].plot([r["bytes"] for r in v], [r["busbw_GBps"] for r in v], "o-", label=name)
    for a, t in zip(ax, ["latency (us)", "bus bandwidth (GB/s)"]):
        a.set_xscale("log", base=2); a.set_xlabel("message bytes"); a.set_ylabel(t); a.grid(alpha=.3); a.legend()
    ax[0].set_yscale("log")
    fig.tight_layout(); fig.savefig(out, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--nccl-tests", action="store_true", help="inputs are nccl-tests stdout logs")
    ap.add_argument("--out", default="results/sweep.png")
    a = ap.parse_args()
    rows = []
    for p in a.inputs:
        rows += parse_nccl_tests(open(p).read(), p) if a.nccl_tests else load_jsonl(p)
    plot(rows, a.out)
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
