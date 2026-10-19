"""Run GPU stages on Modal. Usage: modal run tools/modal_run.py --stage 01
Results are written to <stage>/gpu_results/ in this repo."""
import subprocess, pathlib, modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch", "transformers<5", "huggingface-hub<1", "accelerate", "pandas",
                 "pyarrow", "triton", "pytest", "numpy", "vllm")
    .add_local_dir(str(ROOT), "/repo", ignore=["**/.git/**", "**/__pycache__/**"])
)
app = modal.App("ai-infra", image=image)

CMDS = {
    "01": ["bash", "-c", "nvidia-smi -L; nvcc --version | tail -1; make ARCH=sm_90 && ./bench 1000 900 1100 && ./bench 1024 1024 1024 20 && ./bench 4096 4096 4096 5"],
    "02": ["bash", "-c", "nvidia-smi -L; python profile_llm.py --model gpt2 --device cuda"],
    "06": ["bash", "-c", "nvidia-smi -L; python -m pytest -q -rs; mkdir -p gpu_results; python bench.py --out gpu_results/bench"],
    "05": ["bash", "-c", "nvidia-smi -L; mkdir -p gpu_results; M=Qwen/Qwen2.5-1.5B; python bench_quant.py --backend hf --model $M --tag fp16 --out gpu_results/results; python bench_quant.py --backend vllm --model $M --quant none --tag vllm-bf16 --out gpu_results/results; python bench_quant.py --backend vllm --model $M --quant fp8 --tag vllm-fp8 --out gpu_results/results"],
}
DIRS = {"01": "01-cuda-matmul", "02": "02-inference-physics", "05": "05-quantization-bench", "06": "06-triton-rmsnorm"}

@app.function(gpu="H100", timeout=1800)
def run(stage: str) -> tuple:
    p = subprocess.run(CMDS[stage], cwd=f"/repo/{DIRS[stage]}", capture_output=True, text=True)
    d = pathlib.Path(f"/repo/{DIRS[stage]}/gpu_results")
    files = {f.name: f.read_bytes() for f in d.glob("*")} if d.exists() else {}
    return f"exit={p.returncode}\n{p.stdout}\n{p.stderr[-3000:]}", files

@app.local_entrypoint()
def main(stage: str):
    out, files = run.remote(stage)
    d = ROOT / DIRS[stage] / "gpu_results"; d.mkdir(exist_ok=True)
    for n, b in files.items(): (d / n).write_bytes(b)
    (d / "run.txt").write_text(out)
    print(out)
