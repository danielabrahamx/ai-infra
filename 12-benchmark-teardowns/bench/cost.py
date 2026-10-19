"""Cost per token from measured throughput and a GPU price you supply.

cost per 1M output tokens = (gpu_usd_per_hr * n_gpus) / (tokens_per_s * 3600) * 1e6
Assumes the GPU is fully busy at that throughput. Idle time raises the real cost.
"""
import argparse
import json


def usd_per_million_tokens(gpu_usd_hr: float, tok_s: float, n_gpus: int = 1, utilization: float = 1.0) -> float | None:
    """Return None when throughput is zero. utilization in (0, 1] scales effective throughput."""
    eff = tok_s * utilization
    if eff <= 0:
        return None
    return gpu_usd_hr * n_gpus / (eff * 3600) * 1e6


def cost_table(result: dict, gpu_usd_hr: float, n_gpus: int = 1, utilization: float = 1.0) -> list[dict]:
    return [{"concurrency": lv["concurrency"], "throughput_tok_s": lv["throughput_tok_s"],
             "usd_per_1m_tokens": usd_per_million_tokens(gpu_usd_hr, lv["throughput_tok_s"], n_gpus, utilization)}
            for lv in result["levels"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", help="results JSON from harness")
    p.add_argument("--gpu-usd-hr", type=float, required=True)
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--utilization", type=float, default=1.0)
    a = p.parse_args()
    rows = cost_table(json.load(open(a.result)), a.gpu_usd_hr, a.n_gpus, a.utilization)
    print("concurrency  tok/s      $/1M tokens")
    for r in rows:
        c = r["usd_per_1m_tokens"]
        print(f"{r['concurrency']:<12} {r['throughput_tok_s']:<10.1f} {'n/a' if c is None else f'{c:.4f}'}")


if __name__ == "__main__":
    main()
