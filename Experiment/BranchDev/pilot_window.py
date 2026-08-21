"""Compare two ways of obtaining the first 256 generated tokens with reasoning on.

  A (direct):  max_tokens=256, take whatever comes back
  B (generous): max_tokens=2048, truncate to the first 256 tokens afterwards

If A and B give the same leading text, A is the cheaper, simpler choice. If they
differ, the model is adapting its reasoning to the budget and only B gives the
natural opening we want to cluster on.
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


async def one_call(client, sem, model, prob, k, arm, max_tokens, temperature):
    async with sem:
        t0 = time.time()
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user",
                               "content": or_common.user_content(prob["problem"])}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"reasoning": {"exclude": False}},
                )
                break
            except Exception as e:
                if attempt == 3:
                    return {"problem_id": prob["id"], "k": k, "arm": arm,
                            "error": repr(e)[:300]}
                await asyncio.sleep(2 ** attempt)
        msg = r.choices[0].message
        usage = r.usage.model_dump() if hasattr(r.usage, "model_dump") else dict(r.usage)
        det = usage.get("completion_tokens_details") or {}
        return {
            "problem_id": prob["id"], "k": k, "arm": arm,
            "max_tokens": max_tokens,
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
    ap.add_argument("--problems", type=int, default=5)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--generous", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--tag", default="window_ab")
    a = ap.parse_args()

    probs = or_common.load_problems(a.problems)
    client = AsyncOpenAI(base_url=or_common.BASE_URL, api_key=or_common.load_key(),
                         timeout=600.0, max_retries=0)
    sem = asyncio.Semaphore(a.concurrency)

    tasks = []
    for p in probs:
        for k in range(a.n):
            tasks.append(one_call(client, sem, a.model, p, k, "A_direct", a.window, a.temperature))
            tasks.append(one_call(client, sem, a.model, p, k, "B_generous", a.generous, a.temperature))
    print(f"dispatching {len(tasks)} calls ({a.model}) "
          f"A=max_tokens {a.window} | B=max_tokens {a.generous}", flush=True)
    t0 = time.time()
    rows = await asyncio.gather(*tasks)
    wall = time.time() - t0

    os.makedirs(or_common.OUT, exist_ok=True)
    path = os.path.join(or_common.OUT, f"pilot_{a.tag}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for arm in ("A_direct", "B_generous"):
        ok = [r for r in rows if r.get("arm") == arm and "error" not in r]
        if not ok:
            continue
        rt = sorted(r["reasoning_tokens"] or 0 for r in ok)
        n = len(rt)
        has_content = sum(1 for r in ok if (r["content"] or "").strip())
        print(f"\n{arm}: n={len(ok)} "
              f"reasoning_tokens p50={rt[n//2]} p90={rt[int(n*0.9)]} max={rt[-1]} | "
              f"non-empty content {has_content}/{len(ok)} | "
              f"cost ${sum(r['cost'] or 0 for r in ok):.4f}")
    print(f"\nwall {wall:.1f}s | wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
