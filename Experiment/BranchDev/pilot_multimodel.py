"""Same problems, same sampling, same prompt -- only the source model changes.

Every source gets an identical request body apart from `model` and its pinned
endpoint, so any difference in the collected reasoning is attributable to the
model rather than to how we asked. Each endpoint below was checked to accept
temperature, top_p and top_k explicitly, so no sampling parameter is silently
dropped on any arm.
"""
import argparse, asyncio, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import or_common
from openai import AsyncOpenAI

# model -> pinned endpoint. All five accept temperature/top_p/top_k.
SOURCES = {
    # Sources are chosen for three things at once: the pinned endpoint declares
    # temperature/top_p/top_k (so nothing is silently dropped), a 40-request
    # burst goes through with no 429 (no per-minute bucket, which is what makes
    # a large batch feasible), and the reasoning field comes back as text.
    #
    # Dropped after measurement:
    #   z-ai/glm-5.3, qwen/qwen3.7-max, deepseek/deepseek-v4-pro-0813,
    #   qwen/qwen3.8-2.4t-a95b, moonshotai/kimi-k2.6
    #     -> all capped at 10 requests per clock-minute (new-account bucket,
    #        shared across every endpoint of the model, unaffected by credits)
    #   z-ai/glm-5.3 additionally rejects temperature > 1.0
    #   meta/muse-spark-1.2
    #     -> returns reasoning as an encrypted blob, not text
    "deepseek/deepseek-v4-flash-0731": "deepinfra/fp8",
    "minimax/minimax-m3":              "deepinfra/fp8",
    "qwen/qwen3.8-27b":                "chutes/fp8",
}


async def one(client, sem, model, endpoint, prob, k, temp, max_tokens):
    body = {
        "reasoning": {"enabled": True, "exclude": False},
        "transforms": [],
        "provider": {"order": [endpoint], "allow_fallbacks": False},
        "top_p": 1.0,
        "top_k": 0,
    }
    async with sem:
        t0 = time.time()
        last = None
        for attempt in range(5):
            try:
                raw = await client.chat.completions.with_raw_response.create(
                    model=model,
                    messages=[{"role": "user",
                               "content": or_common.user_content(prob["problem"])}],
                    temperature=temp, max_tokens=max_tokens, extra_body=body)
                r = raw.parse()
                break
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt == 4:
                    return {"model": model, "problem_id": prob["id"], "k": k,
                            "temperature": temp, "error": last}
                await asyncio.sleep(2 ** attempt)
        d = r.model_dump(); msg = r.choices[0].message
        u = d.get("usage") or {}
        return {
            "model": model, "endpoint_requested": endpoint,
            "provider_served": d.get("provider"),
            "problem_id": prob["id"], "level": prob.get("level"),
            "type": prob.get("type"), "k": k, "temperature": temp,
            "top_p": 1.0, "top_k": 0, "max_tokens": max_tokens,
            "finish_reason": r.choices[0].finish_reason,
            "reasoning": getattr(msg, "reasoning", None),
            "content": msg.content,
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "cost": u.get("cost"), "latency_s": round(time.time() - t0, 2),
        }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem-ids", default="3,115,180,35,68")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--models", default=",".join(SOURCES))
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    ids = [int(x) for x in a.problem_ids.split(",")]
    allp = {p["id"]: p for p in or_common.load_problems()}
    probs = [allp[i] for i in ids]
    models = [m.strip() for m in a.models.split(",")]

    print(f"T={a.temperature} top_p=1.0 top_k=0 max_tokens={a.max_tokens} "
          f"n={a.n} | {len(probs)} problems x {len(models)} models "
          f"= {len(probs)*len(models)*a.n} calls")
    for m in models:
        print(f"  {m:<36} -> {SOURCES[m]}")
    print(flush=True)

    client = AsyncOpenAI(base_url=or_common.BASE_URL, api_key=or_common.load_key(),
                         timeout=900.0, max_retries=0)
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.time()
    rows = await asyncio.gather(*[
        one(client, sem, m, SOURCES[m], p, k, a.temperature, a.max_tokens)
        for m in models for p in probs for k in range(a.n)])

    os.makedirs(or_common.OUT, exist_ok=True)
    path = os.path.join(or_common.OUT, f"pilot_{a.tag}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"wall {time.time()-t0:.1f}s | wrote {path}\n")
    for m in models:
        sub = [r for r in rows if r["model"] == m]
        ok = [r for r in sub if "error" not in r]
        cost = sum(r.get("cost") or 0 for r in ok)
        served = Counter(r.get("provider_served") for r in ok)
        print(f"  {m:<36} ok {len(ok):>3}/{len(sub)}  ${cost:.4f}  served={dict(served)}")
        for r in sub:
            if "error" in r:
                print(f"      ERROR: {r['error'][:160]}")
                break


if __name__ == "__main__":
    asyncio.run(main())
