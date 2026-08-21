"""Temperature sweep on a single problem, with reasoning forced on.

The question this answers: at a fixed state, does raising temperature actually
split the model into different approaches, or does it only perturb wording (and
eventually break the text)? Diversity is what discovery needs; broken text is
what it must avoid.
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


async def one_call(client, sem, model, prob, k, temp, max_tokens, top_p):
    # reasoning.enabled forces the thinking pass on; exclude=False keeps its text.
    body = {"reasoning": {"enabled": True, "exclude": False}}
    if top_p is not None:
        body["top_p"] = top_p
    async with sem:
        t0 = time.time()
        for attempt in range(5):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user",
                               "content": or_common.user_content(prob["problem"])}],
                    temperature=temp,
                    max_tokens=max_tokens,
                    extra_body=body,
                )
                break
            except Exception as e:
                if attempt == 4:
                    return {"problem_id": prob["id"], "k": k, "temperature": temp,
                            "error": repr(e)[:300]}
                await asyncio.sleep(2 ** attempt)
        msg = r.choices[0].message
        usage = r.usage.model_dump() if hasattr(r.usage, "model_dump") else dict(r.usage)
        det = usage.get("completion_tokens_details") or {}
        return {
            "problem_id": prob["id"], "k": k, "temperature": temp, "top_p": top_p,
            "model": model,
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
    ap.add_argument("--problem-id", type=int, default=3)
    ap.add_argument("--n", type=int, default=24, help="samples per temperature")
    ap.add_argument("--temps", default="0.7,1.0,1.3,1.6,2.0")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--tag", default="tsweep_p3")
    a = ap.parse_args()

    allp = or_common.load_problems()
    prob = next(p for p in allp if p["id"] == a.problem_id)
    temps = [float(x) for x in a.temps.split(",")]
    print(f"problem {a.problem_id}: {prob['problem'][:110]}")
    print(f"temps={temps} n={a.n} each, max_tokens={a.max_tokens}, "
          f"top_p={a.top_p}, reasoning forced on", flush=True)

    client = AsyncOpenAI(base_url=or_common.BASE_URL, api_key=or_common.load_key(),
                         timeout=600.0, max_retries=0)
    sem = asyncio.Semaphore(a.concurrency)
    tasks = [one_call(client, sem, a.model, prob, k, t, a.max_tokens, a.top_p)
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
        print("sample error:", err[0]["error"][:200])
    for t in temps:
        sub = [r for r in ok if r["temperature"] == t]
        empty = sum(1 for r in sub if not (r["reasoning"] or "").strip())
        print(f"  T={t}: n={len(sub)} empty-reasoning={empty}")
    print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
