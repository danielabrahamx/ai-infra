"""Third batch: AWQ/GPTQ + Nsight (1xH100), 70B-class chunked prefill (4xH100).
modal run tools/modal_run3.py::run_quant | ::run_big"""
import subprocess, pathlib, time, urllib.request, os, modal
import modal.experimental

ROOT = pathlib.Path(__file__).resolve().parent.parent
IGN = ["**/.git/**", "**/__pycache__/**", "**/*_stdout.txt", "**/*_log.txt"]
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch", "transformers<5", "huggingface-hub<1", "accelerate", "pandas", "pyarrow",
                 "pytest", "numpy", "vllm", "aiohttp", "httpx", "matplotlib")
    .add_local_dir(str(ROOT), "/repo", ignore=IGN)
)
app = modal.App("ai-infra-3", image=image)

def sh(cmd, cwd, log):
    p = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True)
    log.append(f"$ {cmd}\nexit={p.returncode}\n{p.stdout[-4000:]}\n{p.stderr[-2000:]}\n"); return p

def collect(dirs):
    files = {}
    for d in dirs:
        g = pathlib.Path(f"/repo/{d}/gpu_results")
        if g.exists():
            for f in g.glob("*"):
                if f.suffix != ".log": files[f"{d}/gpu_results/{f.name}"] = f.read_bytes()
    return files

@app.function(gpu="H100", timeout=1800)
def quant():
    log = []
    d5 = "/repo/05-quantization-bench"; os.makedirs(f"{d5}/gpu_results", exist_ok=True)
    q = "python bench_quant.py --backend vllm --max-tokens 30000 --out gpu_results/results_7b_int4"
    sh(f"{q} --model Qwen/Qwen2.5-7B-Instruct --quant none --tag vllm-bf16-instruct", d5, log)
    sh(f"{q} --model Qwen/Qwen2.5-7B-Instruct-AWQ --quant awq --tag vllm-awq-int4", d5, log)
    sh(f"{q} --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --quant gptq --tag vllm-gptq-int4", d5, log)
    d2 = "/repo/02-inference-physics"; os.makedirs(f"{d2}/gpu_results", exist_ok=True)
    sh("(apt-get update -qq && apt-get install -y -qq nsight-systems-cli) 2>&1 | tail -n 3; which nsys || ls /opt/nvidia", d2, log)
    sh("nsys profile -o gpu_results/llm --force-overwrite true --trace=cuda,nvtx python profile_llm.py --model Qwen/Qwen2.5-7B --device cuda --dtype bfloat16 --new-tokens 16 --prompt-len 512 --nvtx > gpu_results/nsys_run.txt 2>&1; nsys stats --report cuda_gpu_kern_sum --format csv gpu_results/llm.nsys-rep > gpu_results/nsys_kernels.csv 2>&1; ls -la gpu_results; head -n 12 gpu_results/nsys_kernels.csv", d2, log)
    return "\n".join(log), collect(["05-quantization-bench", "02-inference-physics"])

@app.function(gpu="H100:4", timeout=2700)
def big():
    log = []
    d3 = "/repo/03-serving-chunked-prefill"; os.makedirs(f"{d3}/gpu_results", exist_ok=True)
    M = "Qwen/Qwen2.5-72B-Instruct"
    for tag, chunked, mb in [("72b_unchunked", "0", "32768"), ("72b_chunk1024", "1", "1024"), ("72b_chunk4096", "1", "4096")]:
        env = dict(os.environ, MODEL=M, TP="4", MAX_BATCHED=mb, CHUNKED=chunked, MAX_LEN="32768", PORT="8000")
        srv = subprocess.Popen(["bash", "launch_vllm.sh"], cwd=d3, env=env, stdout=open(f"/tmp/server_{tag}.log", "w"), stderr=subprocess.STDOUT)
        ok = False
        for _ in range(240):
            try:
                urllib.request.urlopen("http://localhost:8000/health", timeout=2); ok = True; break
            except Exception: time.sleep(5)
        if ok:
            sh(f"python loadtest.py --model {M} --n 60 --rate 2 --long-frac 0.2 --long-words 14000 --max-tokens 128 --label {tag} > gpu_results/load_{tag}.txt 2>&1; tail -n 3 gpu_results/load_{tag}.txt", d3, log)
        else:
            log.append(f"{tag}: server did not start\n" + open(f"/tmp/server_{tag}.log").read()[-1500:])
        srv.terminate(); srv.wait(timeout=90); time.sleep(8)
    return "\n".join(log), collect(["03-serving-chunked-prefill"])

@app.function(gpu="H100", timeout=1500)
def sweep7b():
    log = []
    d3 = "/repo/03-serving-chunked-prefill"; os.makedirs(f"{d3}/gpu_results", exist_ok=True)
    M = "Qwen/Qwen2.5-7B-Instruct"
    for tag, chunked, mb in [("7b_unchunked", "0", "16384"), ("7b_chunk512", "1", "512"), ("7b_chunk2048", "1", "2048")]:
        env = dict(os.environ, MODEL=M, TP="1", MAX_BATCHED=mb, CHUNKED=chunked, MAX_LEN="16384", PORT="8000")
        srv = subprocess.Popen(["bash", "launch_vllm.sh"], cwd=d3, env=env, stdout=open(f"/tmp/s_{tag}.log", "w"), stderr=subprocess.STDOUT)
        ok = False
        for _ in range(90):
            try:
                urllib.request.urlopen("http://localhost:8000/health", timeout=2); ok = True; break
            except Exception: time.sleep(5)
        if ok:
            sh(f"python loadtest.py --model {M} --n 120 --rate 3 --long-frac 0.2 --long-words 8000 --max-tokens 128 --label {tag} > gpu_results/load2_{tag}.txt 2>&1; tail -n 2 gpu_results/load2_{tag}.txt", d3, log)
        else: log.append(f"{tag}: server did not start")
        srv.terminate(); srv.wait(timeout=60); time.sleep(5)
    return "\n".join(log), collect(["03-serving-chunked-prefill"])

def _save(res, name):
    out, files = res
    for n, b in files.items():
        p = ROOT / n; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b)
    pathlib.Path(os.environ.get("TEMP", ".")).joinpath(f"{name}.txt").write_text(out)
    print(out[-7000:])

@app.local_entrypoint()
def run_quant(): _save(quant.remote(), "quant")
@app.local_entrypoint()
def run_big(): _save(big.remote(), "big")
@app.local_entrypoint()
def run_sweep7b(): _save(sweep7b.remote(), "sweep7b")
