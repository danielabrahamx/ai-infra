"""Second batch of Modal GPU runs. `modal run tools/modal_run2.py::single` (1xH100: stages 02b, 05b, 03)
and `modal run tools/modal_run2.py::multi` (8xH100 single node: stages 07, 09 NCCL)."""
import subprocess, pathlib, time, urllib.request, os, modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch", "transformers<5", "huggingface-hub<1", "accelerate", "pandas", "pyarrow",
                 "pytest", "numpy", "vllm", "aiohttp", "httpx", "matplotlib")
    .add_local_dir(str(ROOT), "/repo", ignore=["**/.git/**", "**/__pycache__/**", "**/*_stdout.txt", "**/*_log.txt"])
)
app = modal.App("ai-infra-2", image=image)

def sh(cmd, cwd, log):
    p = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True)
    out = f"$ {cmd}\nexit={p.returncode}\n{p.stdout}\n{p.stderr[-2500:]}\n"
    log.append(out); return p

def collect(stage_dirs):
    files = {}
    for d in stage_dirs:
        g = pathlib.Path(f"/repo/{d}/gpu_results")
        if g.exists():
            for f in g.glob("*"): files[f"{d}/gpu_results/{f.name}"] = f.read_bytes()
    return files

@app.function(gpu="H100", timeout=3000)
def single():
    log = []
    R = "/repo"
    os.makedirs(f"{R}/02-inference-physics/gpu_results", exist_ok=True)
    sh("python profile_llm.py --model Qwen/Qwen2.5-7B --device cuda --dtype bfloat16 --new-tokens 32 --prompt-len 512 > gpu_results/run_7b.txt 2>&1; head -8 gpu_results/run_7b.txt",
       f"{R}/02-inference-physics", log)
    os.makedirs(f"{R}/05-quantization-bench/gpu_results", exist_ok=True)
    q = "python bench_quant.py --backend vllm --model Qwen/Qwen2.5-7B --max-tokens 30000 --out gpu_results/results_7b"
    sh(f"{q} --quant none --tag vllm-bf16 && {q} --quant fp8 --tag vllm-fp8", f"{R}/05-quantization-bench", log)
    d3 = f"{R}/03-serving-chunked-prefill"; os.makedirs(f"{d3}/gpu_results", exist_ok=True)
    M = "Qwen/Qwen2.5-7B-Instruct"
    for tag, chunked, mb in [("unchunked", "0", "16384"), ("chunk512", "1", "512"), ("chunk2048", "1", "2048"), ("chunk8192", "1", "8192")]:
        env = dict(os.environ, MODEL=M, TP="1", MAX_BATCHED=mb, CHUNKED=chunked, MAX_LEN="16384", PORT="8000")
        srv = subprocess.Popen(["bash", "launch_vllm.sh"], cwd=d3, env=env, stdout=open(f"{d3}/gpu_results/server_{tag}.log", "w"), stderr=subprocess.STDOUT)
        ok = False
        for _ in range(90):
            try:
                urllib.request.urlopen("http://localhost:8000/health", timeout=2); ok = True; break
            except Exception: time.sleep(5)
        if ok:
            sh(f"python loadtest.py --model {M} --n 60 --rate 4 --long-frac 0.15 --long-words 8000 --max-tokens 128 --label {tag} > gpu_results/load_{tag}.txt 2>&1; tail -12 gpu_results/load_{tag}.txt", d3, log)
        else: log.append(f"{tag}: server did not start\n")
        srv.terminate(); srv.wait(timeout=60); time.sleep(5)
    return "\n".join(log), collect(["02-inference-physics", "05-quantization-bench", "03-serving-chunked-prefill"])

@app.function(gpu="H100:8", timeout=1800)
def multi():
    log = []
    R = "/repo"
    d7 = f"{R}/07-distributed-inference"; os.makedirs(f"{d7}/gpu_results", exist_ok=True)
    sh("nvidia-smi topo -m | head -12; for n in 2 4 8; do python bench_allreduce.py --nprocs $n --backend nccl --out gpu_results/nccl_${n}p; done", d7, log)
    d9 = f"{R}/09-multinode-interconnects"; os.makedirs(f"{d9}/gpu_results", exist_ok=True)
    sh("for n in 2 8; do torchrun --standalone --nproc_per_node=$n scripts/sweep.py --backend nccl --label nvlink_${n}gpu --out gpu_results/sweep_${n}gpu.jsonl; done", d9, log)
    return "\n".join(log), collect(["07-distributed-inference", "09-multinode-interconnects"])

@app.function(gpu="H100:8", timeout=900)
def multi07():
    log = []
    d7 = "/repo/07-distributed-inference"; os.makedirs(f"{d7}/gpu_results", exist_ok=True)
    sh("nvidia-smi topo -m | head -12; for n in 2 4 8; do torchrun --standalone --nproc_per_node=$n bench_allreduce.py --backend nccl --out gpu_results/nccl_${n}p.json; done", d7, log)
    return "\n".join(log), collect(["07-distributed-inference"])

def _save(res, name):
    out, files = res
    for n, b in files.items():
        p = ROOT / n; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b)
    (ROOT / "tools" / f"{name}_log.txt").write_text(out); print(out[-6000:])

@app.local_entrypoint()
def run_single(): _save(single.remote(), "single")
@app.local_entrypoint()
def run_multi(): _save(multi.remote(), "multi")

@app.local_entrypoint()
def run_multi07(): _save(multi07.remote(), "multi07")
