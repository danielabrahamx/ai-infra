"""Settings. All values come from env vars; SLOs come from a TOML file."""
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Upstream:
    name: str
    base_url: str          # e.g. http://localhost:11434/v1 (Ollama)
    model: str
    api_key: str = ""


@dataclass
class SLO:
    ttft_ms: float = 500.0
    itl_ms: float = 100.0


@dataclass
class Settings:
    local: Upstream
    frontier: Upstream
    threshold: float = 0.5          # score >= threshold -> frontier
    fallback: bool = True           # retry on the other upstream if one fails
    timeout_s: float = 120.0
    slo: dict = field(default_factory=dict)   # route name -> SLO


def load_slo(path: str | None) -> dict:
    out = {"local": SLO(), "frontier": SLO(ttft_ms=1500, itl_ms=150)}
    if path and Path(path).exists():
        data = tomllib.loads(Path(path).read_text())
        for route, vals in data.items():
            out[route] = SLO(**vals)
    return out


def from_env() -> Settings:
    e = os.environ.get
    return Settings(
        local=Upstream("local", e("LOCAL_BASE_URL", "http://localhost:11434/v1"),
                       e("LOCAL_MODEL", "llama3.2:3b-instruct-q4_K_M"), e("LOCAL_API_KEY", "")),
        frontier=Upstream("frontier", e("FRONTIER_BASE_URL", "https://api.openai.com/v1"),
                          e("FRONTIER_MODEL", "gpt-4o"), e("FRONTIER_API_KEY", "")),
        threshold=float(e("COMPLEXITY_THRESHOLD", "0.5")),
        fallback=e("GATEWAY_FALLBACK", "1") == "1",
        timeout_s=float(e("UPSTREAM_TIMEOUT_S", "120")),
        slo=load_slo(e("SLO_FILE", str(Path(__file__).parent.parent / "slo.toml"))),
    )
