"""Is the single-approach collapse a property of the model, or of the problem?

Problem 3 (Level 2 Number Theory) has one canonical route: lcm then floor-divide.
If diversity is bounded by the problem rather than by sampling, then problems that
genuinely admit several routes should show several routes at the same temperature.
Same model, same endpoint, same settings, only the problem changes.
"""
import argparse, asyncio, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import or_common
from openai import AsyncOpenAI


async def one(client, sem, model, endpoint, prob, k, temp, max_tokens):
    body = {
        "reasoning": {"enabled": True, "exclude": False},
        "transforms": [],
        "provider": {"order": [endpoint], "allow_fallbacks": False},
        "top_p": 1.0, "top_k": 0,
    }
    async with sem:
        for attempt in range(5):
            try:
                raw = await client.chat.completions.with_raw_response.create(
                    model=model,
                    messages=[{"role": "user",
                               "content": or_common.user_content(prob["problem"])}],
                    temperature=temp, max_tokens=max_tokens, extra_body=body)
                r = raw.parse(); break
            except Exception as e:
                if attempt == 4:
                    return {"problem_id": prob["id"], "k": k, "error": repr(e)[:200]}
                await asyncio.sleep(2 ** attempt)
        d = r.model_dump(); msg = r.choices[0].message
        u = d.get("usage") or {}
        return {"problem_id": prob["id"], "level": prob.get("level"),
                "type": prob.get("type"), "k": k, "temperature": temp,
                "provider_served": d.get("provider"),
                "finish_reason": r.choices[0].finish_reason,
                "reasoning": getattr(msg, "reasoning", None),
                "content": msg.content,
                "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cost": u.get("cost")}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash-0731")
    ap.add_argument("--endpoint", default="deepinfra/fp8")
    ap.add_argument("--problem-ids", default="3,115,180,35,68")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--tag", default="probdiv")
    a = ap.parse_args()

    ids = [int(x) for x in a.problem_ids.split(",")]
    allp = {p["id"]: p for p in or_common.load_problems()}
    probs = [allp[i] for i in ids]
    print(f"model={a.model} endpoint={a.endpoint} T={a.temperature} "
          f"max_tokens={a.max_tokens} n={a.n} per problem")
    for p in probs:
        print(f"  id={p['id']:>3} {p['level']:<8} {p['type']:<22} ans={p['answer'][:18]}")
    print(flush=True)

    client = AsyncOpenAI(base_url=or_common.BASE_URL, api_key=or_common.load_key(),
                         timeout=600.0, max_retries=0)
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.time()
    rows = await asyncio.gather(*[one(client, sem, a.model, a.endpoint, p, k,
                                      a.temperature, a.max_tokens)
                                  for p in probs for k in range(a.n)])
    os.makedirs(or_common.OUT, exist_ok=True)
    path = os.path.join(or_common.OUT, f"pilot_{a.tag}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in rows if "error" not in r]
    print(f"wall {time.time()-t0:.1f}s | ok {len(ok)}/{len(rows)} | "
          f"cost ${sum(r.get('cost') or 0 for r in ok):.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
