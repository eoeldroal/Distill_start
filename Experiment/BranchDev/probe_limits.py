"""Measure each source's real concurrency ceiling from the server's own headers.

OpenRouter returns x-ratelimit-limit / -remaining / -reset on 429s, so the limit
does not have to be guessed: fire a burst larger than any plausible cap and read
what comes back.
"""
import asyncio, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import or_common
from openai import AsyncOpenAI

TARGETS = [
    ("deepseek/deepseek-v4-flash-0731", "deepinfra/fp8"),
    ("deepseek/deepseek-v4-pro-0813",   "novita/fp8"),
    ("minimax/minimax-m3",              "deepinfra/fp8"),
    ("z-ai/glm-5.3",                    "z-ai/fp8"),
    ("qwen/qwen3.7-max",                "alibaba"),
]
P = None


async def one(c, model, ep, k):
    body = {"reasoning": {"enabled": True, "exclude": False}, "transforms": [],
            "provider": {"order": [ep], "allow_fallbacks": False},
            "top_p": 1.0, "top_k": 0}
    t0 = time.time()
    try:
        raw = await c.chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": or_common.user_content(P["problem"])}],
            temperature=1.0, max_tokens=48, extra_body=body)
        h = dict(raw.headers)
        return ("ok", time.time() - t0, h, None)
    except Exception as e:
        h = getattr(getattr(e, "response", None), "headers", None)
        return ("429" if "429" in str(e) else "err", time.time() - t0,
                dict(h) if h else {}, str(e)[:120])


async def probe(c, model, ep, burst):
    res = await asyncio.gather(*[one(c, model, ep, k) for k in range(burst)])
    ok = sum(1 for r in res if r[0] == "ok")
    lim = rem = reset = None
    err = None
    for st, _, h, e in res:
        lo = {k.lower(): v for k, v in h.items()}
        lim = lim or lo.get("x-ratelimit-limit")
        rem = rem if rem is not None else lo.get("x-ratelimit-remaining")
        reset = reset or lo.get("x-ratelimit-reset")
        if st != "ok" and not err:
            err = e
    lat = sorted(r[1] for r in res if r[0] == "ok")
    print(f"  {model.split('/')[-1]:<26} burst={burst:>3} ok={ok:>3} "
          f"429={sum(1 for r in res if r[0]=='429'):>3} "
          f"hdr_limit={lim} remaining={rem} "
          f"lat_p50={lat[len(lat)//2]:.1f}s" if lat else "")
    if err:
        print(f"      err: {err}")
    return ok


async def main():
    global P
    P = [p for p in or_common.load_problems() if p["id"] == 3][0]
    c = AsyncOpenAI(base_url=or_common.BASE_URL, api_key=or_common.load_key(),
                    timeout=300, max_retries=0)
    print("=== concurrency ceiling probe (burst of 40 per model) ===")
    for model, ep in TARGETS:
        await probe(c, model, ep, 40)
        await asyncio.sleep(25)   # let any window reset before the next model


if __name__ == "__main__":
    asyncio.run(main())
