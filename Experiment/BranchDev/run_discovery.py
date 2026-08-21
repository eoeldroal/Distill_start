"""Collect discovery continuations from every source, in parallel, into one file.

All five sources run concurrently and independently: a slow source never holds
up a fast one, and total wall time is set by the slowest single source rather
than the sum. Every row records the full request context (model, endpoint,
sampling, the provider that actually served it) so the file is self-describing.
"""
import argparse, asyncio, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import or_common
from openai import AsyncOpenAI

SOURCES = {
    "z-ai/glm-5.2":                    "ambient/fp8",
    "deepseek/deepseek-v4-flash-0731": "deepinfra/fp8",
    "qwen/qwen3.8-27b":                "chutes/fp8",
    "minimax/minimax-m3":              "deepinfra/fp8",
    "xiaomi/mimo-v2.5-pro":            "deepinfra/fp8",
    "meta/muse-glimmer-30b":           "deepinfra/bf16",
}
TEMPERATURE = 1.3
TOP_P = 1.0
TOP_K = 0
MAX_TOKENS = 256


async def one(client, sem, model, endpoint, prob, k, counter, total):
    body = {
        "reasoning": {"enabled": True, "exclude": False},
        "transforms": [],
        "provider": {"order": [endpoint], "allow_fallbacks": False},
        "top_p": TOP_P, "top_k": TOP_K,
    }
    async with sem:
        t0 = time.time()
        err = None
        for attempt in range(4):
            try:
                raw = await client.chat.completions.with_raw_response.create(
                    model=model,
                    messages=[{"role": "user",
                               "content": or_common.user_content(prob["problem"])}],
                    temperature=TEMPERATURE, max_tokens=MAX_TOKENS, extra_body=body)
                r = raw.parse()
                break
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:180]}"
                if attempt == 3:
                    counter["done"] += 1
                    return {"model": model, "endpoint": endpoint, "problem_id": prob["id"],
                            "sample_k": k, "error": err}
                await asyncio.sleep(1.5 ** attempt)
        d = r.model_dump()
        msg = r.choices[0].message
        u = d.get("usage") or {}
        counter["done"] += 1
        if counter["done"] % 200 == 0:
            print(f"    {counter['done']}/{total}", flush=True)
        return {
            "model": model,
            "endpoint_requested": endpoint,
            "provider_served": d.get("provider"),
            "problem_id": prob["id"],
            "level": prob.get("level"),
            "type": prob.get("type"),
            "answer": prob.get("answer"),
            "sample_k": k,
            "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
            "max_tokens": MAX_TOKENS,
            "finish_reason": r.choices[0].finish_reason,
            "reasoning": getattr(msg, "reasoning", None),
            "content": msg.content,
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "cost": u.get("cost"),
            "latency_s": round(time.time() - t0, 2),
        }


async def run_source(client, model, endpoint, probs, n, conc, counter, total):
    """One source's whole workload, with its own concurrency budget."""
    sem = asyncio.Semaphore(conc)
    t0 = time.time()
    rows = await asyncio.gather(*[one(client, sem, model, endpoint, p, k, counter, total)
                                  for p in probs for k in range(n)])
    ok = [r for r in rows if "error" not in r]
    print(f"  {model:<34} {len(ok):>4}/{len(rows)} ok  {time.time()-t0:>6.1f}s  "
          f"${sum(r.get('cost') or 0 for r in ok):.4f}", flush=True)
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="samples per problem per source")
    ap.add_argument("--concurrency", type=int, default=64, help="per source")
    ap.add_argument("--tag", default="discovery_pilot")
    a = ap.parse_args()

    with open(os.path.join(or_common.OUT, "pilot_problems.json")) as f:
        ids = json.load(f)["ids"]
    allp = {p["id"]: p for p in or_common.load_problems()}
    probs = [allp[i] for i in ids]
    total = len(probs) * len(SOURCES) * a.n

    print(f"{len(probs)} problems x {len(SOURCES)} sources x {a.n} samples = {total} calls")
    print(f"T={TEMPERATURE} top_p={TOP_P} top_k={TOP_K} max_tokens={MAX_TOKENS} "
          f"concurrency={a.concurrency}/source\n", flush=True)

    client = AsyncOpenAI(base_url=or_common.BASE_URL, api_key=or_common.load_key(),
                         timeout=900.0, max_retries=0)
    counter = {"done": 0}
    t0 = time.time()
    # every source in flight at once
    results = await asyncio.gather(*[
        run_source(client, m, ep, probs, a.n, a.concurrency, counter, total)
        for m, ep in SOURCES.items()])
    wall = time.time() - t0

    rows = [r for sub in results for r in sub]
    rows.sort(key=lambda r: (r["problem_id"], r["model"], r["sample_k"]))
    path = os.path.join(or_common.OUT, f"{a.tag}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in rows if "error" not in r]
    print(f"\nwall {wall:.1f}s | {len(ok)}/{len(rows)} ok | "
          f"${sum(r.get('cost') or 0 for r in ok):.4f} | {path}")
    bad = [r for r in rows if "error" in r]
    if bad:
        from collections import Counter
        print("errors:", Counter(r["model"] for r in bad))
        print("  e.g.", bad[0]["error"][:160])


if __name__ == "__main__":
    asyncio.run(main())
