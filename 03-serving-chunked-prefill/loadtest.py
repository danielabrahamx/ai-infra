"""Load-test client for a vLLM OpenAI-compatible server. NOT YET RUN against a 70B server.
Mix: many short chats (measure ITL) + a few long-context prompts (cause stalls).
Prints one JSON line: TTFT, ITL, throughput. Needs aiohttp."""
import argparse, asyncio, json, random, time
import aiohttp

async def one(sess, url, model, prompt, max_tokens, stats):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "stream": True, "temperature": 0}
    t0 = time.perf_counter(); last = None
    async with sess.post(url + "/v1/completions", json=body) as r:
        async for line in r.content:
            line = line.strip()
            if not line.startswith(b"data:") or line.endswith(b"[DONE]"): continue
            if not json.loads(line[5:])["choices"][0].get("text"): continue
            now = time.perf_counter()
            if last is None: stats["ttft"].append(now - t0)
            else: stats["itl"].append(now - last)
            stats["tokens"] += 1; last = now

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(len(xs) * p / 100))] if xs else None

async def main(a):
    rng = random.Random(0)
    stats = {"ttft": [], "itl": [], "tokens": 0}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        tasks, t0 = [], time.perf_counter()
        for i in range(a.n):
            long = rng.random() < a.long_frac
            n_words = a.long_words if long else a.short_words   # ~1 token per word here
            tasks.append(asyncio.create_task(
                one(s, a.url, a.model, "lorem " * n_words, a.max_tokens, stats)))
            await asyncio.sleep(rng.expovariate(a.rate))
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    print(json.dumps({"label": a.label, "ttft_p50_s": pct(stats["ttft"], 50),
        "ttft_p99_s": pct(stats["ttft"], 99), "itl_p50_s": pct(stats["itl"], 50),
        "itl_p99_s": pct(stats["itl"], 99), "itl_p999_s": pct(stats["itl"], 99.9), "itl_max_s": max(stats["itl"]), "tok_per_s": stats["tokens"] / wall}))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--rate", type=float, default=1.0, help="requests/s")
    p.add_argument("--long-frac", type=float, default=0.1)
    p.add_argument("--short-words", type=int, default=300)
    p.add_argument("--long-words", type=int, default=24000)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--label", default="run")
    asyncio.run(main(p.parse_args()))
