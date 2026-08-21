"""Screen candidate discovery sources on the three things that actually matter:
rate limit headroom, whether reasoning comes back as text, and latency.

A source is usable only if a 40-request burst goes through untouched (no
per-minute bucket), the reasoning field carries real text rather than an
encrypted blob, and the pinned endpoint accepts temperature/top_p/top_k.
"""
import asyncio, json, sys, time, urllib.request
sys.path.insert(0, '.')
import or_common
from openai import AsyncOpenAI

NEED = {'temperature', 'top_p', 'top_k'}
CANDIDATES = [
    "openai/gpt-oss-120b",
    "nvidia/nemotron-3-super-120b-a12b",
    "google/gemma-4-31b-it",
    "xiaomi/mimo-v2.5-pro",
    "tencent/hy3",
    "meituan/longcat-2.0",
    "stepfun/step-3.7-flash",
    "inclusionai/ring-2.6-1t",
    "qwen/qwen3.6-35b-a3b",
    "arcee-ai/trinity-large-thinking",
    "mistralai/mistral-small-2603",
    "nousresearch/hermes-4-70b",
]
P = None


def best_endpoint(mid):
    """Cheapest endpoint that declares all three sampling params."""
    try:
        with urllib.request.urlopen(
                f"https://openrouter.ai/api/v1/models/{mid}/endpoints", timeout=30) as f:
            eps = json.load(f)['data']['endpoints']
    except Exception:
        return None, 0, 0
    ok = [e for e in eps if NEED <= set(e.get('supported_parameters', []))]
    if not ok:
        return None, 0, len(eps)
    e = sorted(ok, key=lambda x: float(x['pricing']['completion']))[0]
    return e.get('tag'), len(ok), len(eps)


async def one(c, mid, ep):
    t0 = time.time()
    try:
        raw = await c.chat.completions.with_raw_response.create(
            model=mid, messages=[{"role": "user", "content": or_common.user_content(P['problem'])}],
            temperature=1.3, max_tokens=256,
            extra_body={"reasoning": {"enabled": True, "exclude": False}, "transforms": [],
                        "provider": {"order": [ep], "allow_fallbacks": False},
                        "top_p": 1.0, "top_k": 0})
        r = raw.parse(); m = r.choices[0].message
        d = r.model_dump()
        return ("ok", time.time() - t0, getattr(m, 'reasoning', None) or '',
                (d['choices'][0]['message'].get('reasoning_details')), None)
    except Exception as e:
        return ("429" if "429" in str(e) else "err", time.time() - t0, '', None, str(e)[:120])


async def screen(c, mid):
    ep, nok, ntot = best_endpoint(mid)
    if not ep:
        print(f"  {mid:<40} NO endpoint accepts top_p+top_k ({ntot} total)")
        return
    res = await asyncio.gather(*[one(c, mid, ep) for _ in range(40)])
    ok = [r for r in res if r[0] == "ok"]
    n429 = sum(1 for r in res if r[0] == "429")
    lat = sorted(r[1] for r in ok)
    withtext = sum(1 for r in ok if r[2].strip())
    enc = sum(1 for r in ok if r[3] and any(
        (x or {}).get('type') == 'reasoning.encrypted' for x in (r[3] or [])))
    verdict = "USABLE" if (n429 == 0 and withtext >= len(ok) * 0.9 and ok) else "reject"
    print(f"  {mid:<40} {ep:<20} ok={len(ok):>2}/40 429={n429:>2} "
          f"text={withtext:>2} enc={enc:>2} lat_p50={lat[len(lat)//2] if lat else 0:>5.1f}s  {verdict}")
    if ok and ok[0][2].strip():
        print(f"      {ok[0][2][:150]!r}")
    bad = [r for r in res if r[0] == "err"]
    if bad:
        print(f"      err: {bad[0][4]}")


async def main():
    global P
    P = [p for p in or_common.load_problems() if p['id'] == 3][0]
    c = AsyncOpenAI(base_url=or_common.BASE_URL, api_key=or_common.load_key(),
                    timeout=300, max_retries=0)
    print("=== candidate screen: burst 40, T=1.3, max_tokens=256 ===")
    for mid in CANDIDATES:
        await screen(c, mid)
        await asyncio.sleep(6)


if __name__ == "__main__":
    asyncio.run(main())
