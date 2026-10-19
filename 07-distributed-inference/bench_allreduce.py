"""All-reduce microbenchmark. gloo on CPU, nccl on GPU.

Spawn mode : python bench_allreduce.py --nprocs 4
torchrun   : torchrun --nproc_per_node=4 bench_allreduce.py
"""
import argparse, json, os, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES = [1 << 10, 1 << 14, 1 << 18, 1 << 20, 1 << 22, 1 << 24]  # bytes


def check(world, backend):
    """One all-reduce: rank r holds r+1, so the sum is world*(world+1)/2."""
    dev = "cuda" if backend == "nccl" else "cpu"
    t = torch.full((8,), float(dist.get_rank() + 1), device=dev)
    dist.all_reduce(t)
    assert torch.all(t == world * (world + 1) / 2), t
    if dist.get_rank() == 0:
        print("correctness check OK")


def bench(rank, world, backend, iters, warmup, out):
    dev = "cuda" if backend == "nccl" else "cpu"
    if dev == "cuda":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    res = []
    for nbytes in SIZES:
        x = torch.zeros(nbytes // 4, dtype=torch.float32, device=dev)  # zeros: no overflow
        for _ in range(warmup):
            dist.all_reduce(x)
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        # bus bandwidth as in nccl-tests: 2(n-1)/n * size / time
        res.append({"bytes": nbytes, "time_us": dt * 1e6,
                    "busbw_GBps": 2 * (world - 1) / world * nbytes / dt / 1e9})
    if rank == 0:
        print(f"backend={backend} world={world} iters={iters}")
        print(f"{'bytes':>10} {'time_us':>12} {'busbw_GB/s':>12}")
        for r in res:
            print(f"{r['bytes']:>10} {r['time_us']:>12.1f} {r['busbw_GBps']:>12.3f}")
        if out:
            with open(out, "w") as f:
                json.dump({"backend": backend, "world": world, "results": res}, f, indent=1)


def worker(rank, world, backend, iters, warmup, out):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())  # one GPU per rank
    dist.init_process_group(backend, rank=rank, world_size=world)
    check(world, backend)
    bench(rank, world, backend, iters, warmup, out)
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprocs", type=int, default=2)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    backend = a.backend or ("nccl" if torch.cuda.device_count() > 1 else "gloo")
    if "RANK" in os.environ:  # launched by torchrun
        r, w = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        if backend == "nccl":
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # one GPU per rank
        dist.init_process_group(backend)
        check(w, backend)
        bench(r, w, backend, a.iters, a.warmup, a.out)
        dist.destroy_process_group()
    else:
        mp.spawn(worker, args=(a.nprocs, backend, a.iters, a.warmup, a.out), nprocs=a.nprocs)
