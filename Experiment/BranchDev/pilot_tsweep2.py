"""Temperature sweep with the provider pinned, so quantization and parameter
support stop varying between calls.

Two endpoints are swept separately because no single one gives us both things:
  morph/bf16      - unquantized, but accepts no top_p / top_k
  deepinfra/fp8   - quantized, but accepts the full sampling set

Comparing them at the same temperature separates "temperature broke the text"
from "fp4/fp8 quantization broke the text", which the unpinned sweep could not.
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import WorkPlace.ICLR.BranchDev.or_common as or_common
from openai import AsyncOpenAI


async def one_call(client, sem, model, endpoint, prob, k, temp, max_tokens, sampling):
    body = {
        "reasoning": {"enabled": True, "exclude": False},
        "transforms": [],
        "provider": {"order": [endpoint], "allow_fallbacks": False},
    }
    body.update(sampling)
    async with sem:
        t0 = time.time()
        err = None
        for attempt in range(5):
            try:
                raw = await client.chat.completions.with_raw_response.create(
                    model=model,
                    messages=[{"role": "user",
                               "content": or_common.user_content(prob["problem"])}],
                    temperature=temp,
                    max_tokens=max_tokens,
                    extra_body=body,
                )
                r = raw.parse()
                break
            except Exception as e:
                err = repr(e)[:300]
                if attempt == 4:
                    return {"problem_id": prob["id"], "k": k, "temperature": temp,
                            "endpoint": endpoint, "error": err}
                await asyncio.sleep(2 ** attempt)
        d = r.model_dump()
        msg = r.choices[0].message
        usage = d.get("usage") or {}
        det = usage.get("completion_tokens_details") or {}
        return {
            "problem_id": prob["id"], "k": k, "temperature": temp,
            "endpoint_requested": endpoint,
            "provider_served": d.get("provider"),      # verify the pin held
            "sampling_sent": sampling,
            "finish_reason": r.choices[0].finish_reason,
            "reasoning": getattr(msg, "reasoning", None),
            "content": msg.content,
            "reasoning_tokens": det.get("reasoning_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost": usage.get("cost"),
            "latency_s": round(time.time() - t0, 2),
        }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash-0731")
    ap.add_argument("--endpoint", required=True, help="e.g. morph/bf16")
    ap.add_argument("--problem-id", type=int, default=3)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--temps", default="0.7,1.0,1.3,1.6,2.0")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--full-sampling", action="store_true",
                    help="also send top_p=1.0 and top_k=0 (endpoint must support them)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    sampling = {"top_p": 1.0, "top_k": 0} if a.full_sampling else {}
    prob = next(p for p in or_common.load_problems() if p["id"] == a.problem_id)
    temps = [float(x) for x in a.temps.split(",")]
    print(f"endpoint={a.endpoint} | temps={temps} n={a.n} | "
          f"max_tokens={a.max_tokens} | sampling sent: {sampling or '(temperature only)'}",
          flush=True)

    client = AsyncOpenAI(base_url=or_common.BASE_URL, api_key=or_common.load_key(),
                         timeout=600.0, max_retries=0)
    sem = asyncio.Semaphore(a.concurrency)
    tasks = [one_call(client, sem, a.model, a.endpoint, prob, k, t, a.max_tokens, sampling)
             for t in temps for k in range(a.n)]
    t0 = time.time()
    rows = await asyncio.gather(*tasks)
    wall = time.time() - t0

    os.makedirs(or_common.OUT, exist_ok=True)
    path = os.path.join(or_common.OUT, f"pilot_{a.tag}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in rows if "error" not in r]
    err = [r for r in rows if "error" in r]
    print(f"\nwall {wall:.1f}s | ok {len(ok)} | errors {len(err)} | "
          f"cost ${sum(r.get('cost') or 0 for r in ok):.4f}")
    if err:
        print("sample error:", err[0]["error"][:220])
    from collections import Counter
    print("providers actually served:", Counter(r.get("provider_served") for r in ok))
    print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
