"""Pin down the exact shape of the limit on the three bottleneck models.

The burst test showed a ceiling of 10 with x-ratelimit-limit=10, but 12 strictly
sequential calls all succeeded. That rules out a per-minute request cap and points
to a cap on requests *in flight*. This script tests that directly by holding a
fixed number of concurrent requests for a sustained period: if the limit is
concurrency, a steady load at or below the cap never 429s no matter how many
requests pass through it.
"""
import asyncio, sys, time
sys.path.insert(0, '.')
import or_common
from openai import AsyncOpenAI

BOTTLENECKS = [
    ("z-ai/glm-5.3", "z-ai/fp8"),
    ("qwen/qwen3.7-max", "alibaba"),
    ("deepseek/deepseek-v4-pro-0813", "novita/fp8"),
]
P = None


async def call(c, model, ep):
    try:
        await c.chat.completions.create(
            model=model, messages=[{"role": "user", "content": or_common.user_content(P["problem"])}],
            temperature=1.0, max_tokens=48,
            extra_body={"reasoning": {"enabled": True, "exclude": False}, "transforms": [],
                        "provider": {"order": [ep], "allow_fallbacks": False},
                        "top_p": 1.0, "top_k": 0})
        return True
    except Exception as e:
        return False if "429" in str(e) else None


async def sustained(c, model, ep, conc, total):
    """Keep `conc` requests in flight until `total` have completed."""
    done = {"ok": 0, "429": 0, "err": 0}
    sem = asyncio.Semaphore(conc)
    async def worker(_):
        async with sem:
            r = await call(c, model, ep)
            done["ok" if r is True else ("429" if r is False else "err")] += 1
    t0 = time.time()
    await asyncio.gather(*[worker(i) for i in range(total)])
    el = time.time() - t0
    print(f"    conc={conc:>2} total={total:>3}: ok={done['ok']:>3} 429={done['429']:>3} "
          f"err={done['err']:>2}  {el:>5.1f}s  -> {done['ok']/el:.2f} req/s")
    return done["429"] == 0


async def main():
    global P
    P = [p for p in or_common.load_problems() if p["id"] == 3][0]
    c = AsyncOpenAI(base_url=or_common.BASE_URL, api_key=or_common.load_key(),
                    timeout=300, max_retries=0)
    for model, ep in BOTTLENECKS:
        print(f"\n=== {model} @ {ep} ===")
        for conc in (8, 10, 12):
            await sustained(c, model, ep, conc, conc * 3)
            await asyncio.sleep(8)


if __name__ == "__main__":
    asyncio.run(main())
