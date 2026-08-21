"""Are the anchor's uncertain positions method forks, or just wording forks?

The tree design stands or falls on one assumption: that where Base is uncertain, it is choosing
between ways of SOLVING, not between ways of PHRASING. This probe tests that directly and
introduces no new constant. A fork is a position where at least two next-token candidates each
carry mass m_min = 0.10, the same figure Experiments section 1.3 already derived as the minimum
anchor mass worth protecting, so the criterion is a mass condition rather than an entropy
threshold. Entropy is reported alongside for context.

At each fork the probe forces every qualifying candidate in turn and lets Base continue, so the
continuations share their prefix exactly and differ only in the one committed token. Reading them
side by side answers the question by eye before any labelling budget is spent.
"""
import argparse, json, os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
TEACHER_TOKENIZER = "Qwen/Qwen3-14B"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
M_MIN = 0.10          # Experiments section 1.3; reused, not re-invented


def boxed(t):
    i = t.rfind("\\boxed")
    if i < 0:
        return None
    j = t.find("{", i)
    if j < 0:
        return None
    d, out = 0, []
    for ch in t[j:]:
        if ch == "{":
            d += 1
            if d == 1:
                continue
        elif ch == "}":
            d -= 1
            if d == 0:
                break
        out.append(ch)
    return "".join(out).strip() or None


def norm(a):
    a = str(a).strip()
    for x in ["\\left", "\\right", "\\!", "\\,", "\\ ", " ", "$"]:
        a = a.replace(x, "")
    return a.replace("dfrac", "frac").rstrip(".").replace("^\\circ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="180,115,44")
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--forks", type=int, default=2, help="forks to expand per problem")
    ap.add_argument("--cands", type=int, default=3, help="max candidates per fork")
    ap.add_argument("--conts", type=int, default=4, help="continuations per candidate")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--baseline", type=int, default=16,
                    help="positions of preceding context used as the local entropy level")
    ap.add_argument("--window", type=int, default=256,
                    help="only mine forks in the opening window, where the solution is still on track")
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--device", default="cuda:3")
    a = ap.parse_args()

    sys.path.insert(0, ROOT)
    import or_common
    probs = {p["id"]: p for p in or_common.load_problems()}
    tc = AutoTokenizer.from_pretrained(TEACHER_TOKENIZER)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base", dtype=torch.bfloat16).to(a.device).eval()
    stops = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
             if isinstance(i, int) and i >= 0]

    log = open(os.path.join(OUT, "fork_probe.jsonl"), "w")
    for pid in [int(x) for x in a.problems.split(",")]:
        p = probs[pid]
        prompt = tc.apply_chat_template(
            [{"role": "user", "content": p["problem"] + "\n\n" + INSTRUCTION}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        pre = tok(prompt, return_tensors="pt").to(a.device)
        npre = pre["input_ids"].shape[1]

        print("=" * 92)
        print(f"PROBLEM {pid}  [{p.get('level')}, {p.get('type')}]  gold = {p['answer']}")
        print(f"  {p['problem'][:150]}")

        # 1. trunk rollouts
        with torch.no_grad():
            out = model.generate(**{k: v.repeat(a.rollouts, 1) for k, v in pre.items()},
                                 do_sample=True, temperature=a.temp, top_p=1.0, top_k=0,
                                 max_new_tokens=a.max_new, eos_token_id=stops,
                                 pad_token_id=tok.pad_token_id)
        trunks = []
        for j in range(a.rollouts):
            g = [t for t in out[j, npre:].tolist() if t != tok.pad_token_id]
            while g and g[-1] in stops:
                g.pop()
            trunks.append(g)

        # 2. re-score to get the anchor's own distribution at every position.
        # Only rollouts that actually reach an answer are mined: a fork inside a rollout that
        # collapsed is not a choice between methods, it is the model having lost the thread.
        keep = []
        for ri, g in enumerate(trunks):
            txt = tok.decode(g, skip_special_tokens=True)
            b = boxed(txt)
            keep.append((ri, b is not None, b is not None and norm(b) == norm(p["answer"])))
        good = [ri for ri, hasb, ok in keep if hasb]
        print(f"  rollouts reaching a boxed answer: {len(good)}/{len(trunks)}"
              f"  (correct {sum(ok for _,_,ok in keep)})")
        forks = []
        for ri, g in enumerate(trunks):
            if ri not in good:
                continue
            ids = torch.tensor([pre["input_ids"][0].tolist() + g], device=a.device)
            with torch.no_grad():
                lg = model(ids).logits[0].float()
            P = torch.softmax(lg[npre - 1:-1], dim=-1)
            H = (-(P * P.clamp_min(1e-12).log()).sum(-1)).tolist()
            top = torch.topk(P, 5, dim=-1)
            # A fork is a position where entropy JUMPS above its own local level. Draft 3.1 asks
            # for "clearly higher than its surroundings", which is a contrast, not a level: a
            # collapsed stretch of text sits on a uniformly high plateau and has no jump, while a
            # genuine commitment point sits in calm text and spikes. Ranking by the size of the
            # jump needs no threshold; the earliest large jump is the first commitment.
            W = a.baseline
            for t in range(W + 1, min(len(g), len(H))):
                window = sorted(H[t - W:t])
                base_h = window[len(window) // 2]
                prom = H[t] - base_h
                forks.append({
                    "rollout": ri, "pos": t, "H": H[t], "base": base_h, "prom": prom,
                    "cands": [(int(i), tok.decode([int(i)]), round(float(v), 3))
                              for v, i in zip(top.values[t], top.indices[t])
                              if float(v) >= M_MIN],
                    "taken": tok.decode([g[t]]),
                    "prefix_ids": g[:t],
                })
        print(f"\n  {len(trunks)} trunk rollouts, lengths {[len(t) for t in trunks]}")
        import statistics as st
        proms = sorted((f["prom"] for f in forks), reverse=True)
        cut = proms[max(0, int(len(proms) * 0.02))] if proms else 0.0
        spikes = [f for f in forks if f["prom"] >= cut and len(f["cands"]) >= 2]
        spikes.sort(key=lambda f: (f["rollout"], f["pos"]))
        print(f"  scored {len(forks)} positions; jump sizes: median {st.median(proms):+.2f}"
              f"  top-2% cut {cut:+.2f}  max {proms[0]:+.2f}")
        print(f"  spikes in the top 2% of jumps with >=2 candidates over {M_MIN}: {len(spikes)}")
        # the earliest spike in each rollout is the first commitment point
        seen, forks = set(), []
        for f in spikes:
            if f["rollout"] in seen:
                continue
            seen.add(f["rollout"]); forks.append(f)
        print(f"  first spike per rollout: {len(forks)}")
        if not forks:
            print("  NO FORKS -- the anchor never puts 0.10 on two continuations at once here")
            continue
        # Among positions that qualify by mass, prefer the ones where the anchor is CONFIDENT
        # that only a few alternatives exist. High entropy marks confusion, not a fork.
        for f in forks[:6]:
            ctx = tok.decode(f["prefix_ids"][-14:]) if f["prefix_ids"] else ""
            print(f"    r{f['rollout']} pos {f['pos']:>4}  H={f['H']:.2f} vs local {f['base']:.2f}"
                  f"  (jump {f['prom']:+.2f})  ...{ctx!r}")
            print(f"         -> {[(c[1], c[2]) for c in f['cands']]}   took {f['taken']!r}")

        # 3. expand the highest-entropy forks by forcing each candidate
        for f in forks[:a.forks]:
            print(f"\n  --- EXPANDING pos {f['pos']} (H={f['H']:.2f}) ---")
            print(f"  context: ...{tok.decode(f['prefix_ids'][-40:])!r}")
            for cid, ctxt, cp in f["cands"][:a.cands]:
                ids = pre["input_ids"][0].tolist() + f["prefix_ids"] + [cid]
                base = torch.tensor([ids], device=a.device)
                with torch.no_grad():
                    o = model.generate(input_ids=base.repeat(a.conts, 1),
                                       attention_mask=torch.ones_like(base).repeat(a.conts, 1),
                                       do_sample=True, temperature=a.temp, top_p=1.0, top_k=0,
                                       max_new_tokens=a.max_new, eos_token_id=stops,
                                       pad_token_id=tok.pad_token_id)
                oks, texts = 0, []
                for j in range(a.conts):
                    g = [t for t in o[j, len(ids):].tolist() if t != tok.pad_token_id]
                    txt = tok.decode(g, skip_special_tokens=True)
                    b = boxed(txt)
                    ok = b is not None and norm(b) == norm(p["answer"])
                    oks += ok
                    texts.append(txt)
                print(f"    candidate {ctxt!r} (p={cp}):  correct {oks}/{a.conts}")
                print(f"      -> {(ctxt + texts[0])[:200]!r}")
                log.write(json.dumps({"problem_id": pid, "pos": f["pos"], "H": f["H"],
                                      "cand": ctxt, "p": cp, "correct": oks,
                                      "n": a.conts, "sample": texts[0][:1500]},
                                     ensure_ascii=False) + "\n")
                log.flush()
    log.close()
    print(f"\nwrote {OUT}/fork_probe.jsonl")


if __name__ == "__main__":
    main()
