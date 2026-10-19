"""Communication-cost analyzer for TP / PP / EP inference. Pure python.

Model: alpha-beta. time = latency_steps * alpha + bytes_on_wire / bandwidth.
All sizes are per decoded token (times batch_tokens) and per GPU.
"""
from dataclasses import dataclass

# Link presets: (unidirectional GB/s per GPU, latency in microseconds).
# Nominal public numbers. Real achieved bandwidth is lower.
LINKS = {
    "nvlink4": (450.0, 3.0),    # H100 NVLink 4, 900 GB/s bidirectional
    "ib_ndr": (50.0, 5.0),      # 400 Gb/s NDR InfiniBand
    "ib_hdr": (25.0, 5.0),      # 200 Gb/s HDR
    "pcie4": (32.0, 10.0),      # PCIe 4.0 x16
    "pcie5": (64.0, 10.0),      # PCIe 5.0 x16
}


@dataclass(frozen=True)
class Model:
    name: str = "llama-3.1-405b"
    layers: int = 126
    hidden: int = 16384
    moe_layers: int = 0      # layers with expert routing
    top_k: int = 1           # experts per token
    dtype_bytes: int = 2     # bf16


LLAMA_405B = Model()
# Hypothetical MoE of the same size class, for the EP analysis.
MOE_405B = Model("moe-405b-class", layers=126, hidden=16384, moe_layers=126, top_k=2)


@dataclass(frozen=True)
class Plan:
    tp: int = 1
    pp: int = 1
    ep: int = 1
    gpus_per_node: int = 8
    intra: str = "nvlink4"   # link inside a node
    inter: str = "ib_ndr"    # link between nodes
    batch_tokens: int = 1    # tokens in flight per step


def _link(plan, group, inside_node_ok=True):
    """Return the slowest link a group of GPUs must cross."""
    crosses = group > plan.gpus_per_node or not inside_node_ok
    return LINKS[plan.inter] if crosses else LINKS[plan.intra]


def ring_allreduce_bytes(msg, n):
    """Bytes each GPU sends in a ring all-reduce."""
    return 0.0 if n <= 1 else 2.0 * (n - 1) / n * msg


def alltoall_bytes(msg, n):
    """Bytes each GPU sends in an all-to-all of total buffer `msg`."""
    return 0.0 if n <= 1 else (n - 1) / n * msg


def _t(nbytes, steps, link):
    bw, lat_us = link
    return steps * lat_us * 1e-6 + nbytes / (bw * 1e9)


def analyze(model, plan):
    """Return per-token bytes (per GPU) and time (s) for each strategy."""
    act = model.hidden * model.dtype_bytes * plan.batch_tokens  # one activation batch
    out = {}

    # Tensor parallel: 2 all-reduces per layer (attention out, MLP out).
    # A stage owns layers/pp layers. TP groups sit inside a node first.
    n_ar = 2 * (model.layers // plan.pp)
    ar = ring_allreduce_bytes(act, plan.tp)
    steps = 2 * (plan.tp - 1)  # ring: reduce-scatter + all-gather
    out["tp_allreduce"] = {
        "count_per_token": n_ar if plan.tp > 1 else 0,
        "msg_bytes": act if plan.tp > 1 else 0,
        "bytes_per_token": n_ar * ar,
        "time_s": n_ar * _t(ar, steps, _link(plan, plan.tp)) if plan.tp > 1 else 0.0,
    }

    # Pipeline parallel: one point-to-point send per stage boundary.
    # Stages cross nodes when tp*pp does not fit in one node.
    hops = plan.pp - 1
    fits = plan.tp * plan.pp <= plan.gpus_per_node
    out["pp_p2p"] = {
        "count_per_token": hops,
        "msg_bytes": act if hops else 0,
        "bytes_per_token": hops * act,
        "time_s": hops * _t(act, 1, _link(plan, plan.tp * plan.pp, fits)),
    }

    # Expert parallel: dispatch + combine all-to-all per MoE layer.
    if model.moe_layers and plan.ep > 1:
        buf = act * model.top_k
        a2a = alltoall_bytes(buf, plan.ep)
        n_a2a = 2 * model.moe_layers
        out["ep_alltoall"] = {
            "count_per_token": n_a2a,
            "msg_bytes": buf,
            "bytes_per_token": n_a2a * a2a,
            "time_s": n_a2a * _t(a2a, plan.ep - 1, _link(plan, plan.ep)),
        }
    else:
        out["ep_alltoall"] = {"count_per_token": 0, "msg_bytes": 0,
                              "bytes_per_token": 0, "time_s": 0.0}
    parts = list(out.values())
    out["total_bytes_per_token"] = sum(v["bytes_per_token"] for v in parts)
    out["total_time_s"] = sum(v["time_s"] for v in parts)
    return out


def report(model, plan):
    r = analyze(model, plan)
    lines = [f"{model.name} tp={plan.tp} pp={plan.pp} ep={plan.ep} "
             f"intra={plan.intra} inter={plan.inter} batch={plan.batch_tokens}"]
    for k in ("tp_allreduce", "pp_p2p", "ep_alltoall"):
        v = r[k]
        lines.append(f"  {k:13s} n={v['count_per_token']:4d} msg={v['msg_bytes']/1024:8.1f} KiB "
                     f"bytes/tok/GPU={v['bytes_per_token']/1e6:9.2f} MB time={v['time_s']*1e3:8.3f} ms")
    t = r["total_time_s"]
    lines.append(f"  total comm/token: {t*1e3:.3f} ms" + (f" (<= {1/t:.0f} steps/s ceiling)" if t else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    print("NOTE: analytic alpha-beta estimates, not measurements.\n")
    for plan in [
        Plan(tp=8, pp=1),                      # one node
        Plan(tp=8, pp=8),                      # 8 nodes: TP in node, PP across
        Plan(tp=64, pp=1),                     # TP across 8 nodes
        Plan(tp=8, pp=1, intra="pcie4"),       # PCIe box
        Plan(tp=8, pp=8, batch_tokens=64),     # batched decode
    ]:
        print(report(LLAMA_405B, plan), end="\n\n")
    print(report(MOE_405B, Plan(ep=8)))
    print(report(MOE_405B, Plan(ep=64)))
