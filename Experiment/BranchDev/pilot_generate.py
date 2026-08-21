"""Pilot: probe one OpenRouter source for reasoning length, budget compliance,
and approach diversity at a fixed state.

Usage:
  python pilot_generate.py --model deepseek/deepseek-v4-flash-0731 \
      --problems 5 --n 8 --budget 1024 --tag deepseek
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


async def one_call(client, sem, model, prob, k, temperature, budget, max_tokens, effort):
    # effort is the tuning knob; reasoning.max_tokens is only a true token budget
    # on some families (docs: others map it to an effort percentage).
    # exclude=False keeps the reasoning text in the response (it is the default,
    # stated explicitly here because that text IS the material we are collecting).
    reasoning = {"effort": effort} if effort else {"max_tokens": budget}
    reasoning["exclude"] = False
    body = {"reasoning": reasoning}
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
                    extra_body=body,
                )
                break
            except Exception as e:
                if attempt == 3:
                    return {"problem_id": prob["id"], "k": k, "error": repr(e)[:300]}
                await asyncio.sleep(2 ** attempt)
        msg = r.choices[0].message
        usage = r.usage.model_dump() if hasattr(r.usage, "model_dump") else dict(r.usage)
        det = usage.get("completion_tokens_details") or {}
        return {
            "problem_id": prob["id"],
            "k": k,
            "source": model,
            "temperature": temperature,
            "finish_reason": r.choices[0].finish_reason,
            "reasoning": getattr(msg, "reasoning", None),
            "content": msg.content,
            "reasoning_tokens": det.get("reasoning_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "cost": usage.get("cost"),
            "latency_s": round(time.time() - t0, 2),
        }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", type=int, default=5)
    ap.add_argument("--n", type=int, default=8, help="continuations per problem")
    ap.add_argument("--budget", type=int, default=1024, help="reasoning max_tokens")
    ap.add_argument("--effort", default=None, help="use reasoning.effort instead of max_tokens")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="safety cap on reasoning+answer combined; default 4096")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    max_tokens = a.max_tokens or 4096
    probs = or_common.load_problems(a.problems)
    client = AsyncOpenAI(base_url=or_common.BASE_URL, api_key=or_common.load_key(),
                         timeout=600.0, max_retries=0)
    sem = asyncio.Semaphore(a.concurrency)

    tasks = [one_call(client, sem, a.model, p, k, a.temperature, a.budget, max_tokens, a.effort)
             for p in probs for k in range(a.n)]
    print(f"dispatching {len(tasks)} calls to {a.model} "
          f"(budget={a.effort or a.budget}, max_tokens={max_tokens}, T={a.temperature})",
          flush=True)
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
    print(f"\nwall {wall:.1f}s | ok {len(ok)} | errors {len(err)}")
    if err:
        print("first error:", err[0]["error"])
    if ok:
        rt = sorted(r["reasoning_tokens"] or 0 for r in ok)
        n = len(rt)
        print(f"reasoning_tokens: min {rt[0]} p50 {rt[n//2]} p90 {rt[int(n*0.9)]} max {rt[-1]}")
        trunc = sum(1 for r in ok if r["finish_reason"] == "length")
        print(f"finish_reason=length: {trunc}/{len(ok)}  (budget respected if ~0)")
        empty = sum(1 for r in ok if not (r["reasoning"] or "").strip())
        print(f"empty reasoning: {empty}/{len(ok)}")
        print(f"total cost: ${sum(r['cost'] or 0 for r in ok):.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
