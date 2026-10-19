"""All-reduce latency/bandwidth sweep. Run with torchrun.

Writes one JSON line per message size. Formulas follow nccl-tests:
  algbw = bytes / time
  busbw = algbw * 2 * (n - 1) / n     (all-reduce)
"""
import argparse, json, os, time
import torch
import torch.distributed as dist


def busbw_factor(n: int) -> float:
    """nccl-tests bus-bandwidth correction for all-reduce."""
    return 2.0 * (n - 1) / n if n > 1 else 0.0


def topology() -> str:
    """'intra-node' if all ranks share one node, else 'inter-node'."""
    return "intra-node" if int(os.environ["LOCAL_WORLD_SIZE"]) == int(os.environ["WORLD_SIZE"]) else "inter-node"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    ap.add_argument("--min-bytes", type=int, default=1 << 10)
    ap.add_argument("--max-bytes", type=int, default=1 << 24)
    ap.add_argument("--factor", type=int, default=4, help="size multiplier per step")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="results/sweep.jsonl")
    ap.add_argument("--label", default="", help="free-text tag, e.g. 'ib' or 'tcp'")
    a = ap.parse_args()

    dist.init_process_group(a.backend)
    rank, n = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if a.backend == "nccl" else torch.device("cpu")
    if a.backend == "nccl":
        torch.cuda.set_device(dev)
    sync = torch.cuda.synchronize if a.backend == "nccl" else (lambda: None)

    rows, size = [], a.min_bytes
    while size <= a.max_bytes:
        x = torch.ones(size // 4, dtype=torch.float32, device=dev)
        for _ in range(a.warmup):
            dist.all_reduce(x)
        sync(); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            dist.all_reduce(x)
        sync()
        dt = (time.perf_counter() - t0) / a.iters
        # Use the slowest rank, as nccl-tests does for the max time.
        t = torch.tensor([dt], dtype=torch.float64, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        dt = t.item()
        alg = size / dt / 1e9
        rows.append(dict(backend=a.backend, topology=topology(), label=a.label, world_size=n,
                         bytes=size, time_us=dt * 1e6, algbw_GBps=alg,
                         busbw_GBps=alg * busbw_factor(n)))
        size *= a.factor
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        for r in rows:
            print(f"{r['bytes']:>10} B {r['time_us']:>12.1f} us  alg {r['algbw_GBps']:7.3f}  bus {r['busbw_GBps']:7.3f} GB/s")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
