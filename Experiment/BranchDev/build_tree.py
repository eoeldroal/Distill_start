"""Build the anchor's branch tree by walking its own distribution.

The probes established the shape of the thing. Base is certain while it restates the problem
(entropy near zero), becomes uncertain at the moment it announces an approach, and the uncertainty
resolves over the next one or two tokens: the spike itself offers function words ("use", "follow",
"proceed") and only the ones that open a slot are followed by content ("Cauchy", "trig",
"geometric"). Reading the spike alone mistakes the grammar for the choice, which is why the first
three attempts at this found nothing.

So the walk does what that shape asks. Where two or more candidates each hold m_min it branches;
where one candidate dominates it simply follows, because a certain run carries no choice and only
separates one fork from the next. Every leaf is a committed approach reached through a fixed
prefix, and its probability under any checkpoint is the product of the step probabilities along the
path, computed exactly rather than estimated from rollouts.

m_min = 0.10 is the minimum anchor mass Experiments section 1.3 already derived as worth
protecting. Nothing else here is a threshold: the branch-or-follow decision is that same mass
condition, and the entropy at each node is printed for reading rather than used as a cutoff.
"""
import argparse, json, math, os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
M_MIN = 0.10


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
    """Loose enough that a right answer written another way still counts."""
    if a is None:
        return None
    a = str(a).strip()
    for x in ["\\left", "\\right", "\\!", "\\,", "\;", "\\ ", "$", " ", "{", "}"]:
        a = a.replace(x, "")
    a = (a.replace("dfrac", "frac").replace("tfrac", "frac")
          .replace("^\\circ", "").replace("\\degree", "")
          .replace("\\sqrt2", "\\sqrt(2)").rstrip(".").lower())
    try:
        return f"{float(a):.4f}"
    except ValueError:
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="180")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--depth", type=int, default=3, help="forks expanded along one path")
    ap.add_argument("--follow", type=int, default=40, help="tokens walked through a certain run")
    ap.add_argument("--conts", type=int, default=8, help="completions per leaf")
    ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--baseline", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--device", default="cuda:3")
    a = ap.parse_args()

    sys.path.insert(0, ROOT)
    import or_common
    probs = {q["id"]: q for q in or_common.load_problems()}
    tc = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).eval()
    stops = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
             if isinstance(i, int) and i >= 0]

    def dist(ids):
        with torch.no_grad():
            lg = model(torch.tensor([ids], device=a.device)).logits[0, -1].float()
        P = torch.softmax(lg, -1)
        H = float(-(P * P.clamp_min(1e-12).log()).sum())
        tp = torch.topk(P, 10)
        return H, [(int(i), tok.decode([int(i)]), float(v))
                   for v, i in zip(tp.values, tp.indices)]

    out_f = open(os.path.join(OUT, f"tree_{a.tag}.jsonl"), "w")
    for pid in [int(x) for x in a.problems.split(",")]:
        p = probs[pid]
        prompt = tc.apply_chat_template(
            [{"role": "user", "content": p["problem"] + "\n\n" + INSTRUCTION}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        pre = tok(prompt, return_tensors="pt").to(a.device)
        npre = pre["input_ids"].shape[1]
        print("=" * 96)
        print(f"PROBLEM {pid} [{p.get('level')}, {p.get('type')}]  gold = {p['answer']}")
        print(f"  {p['problem'][:160]}")

        with torch.no_grad():
            o = model.generate(**pre, do_sample=True, temperature=a.temp, top_p=1.0, top_k=0,
                               max_new_tokens=300, eos_token_id=stops,
                               pad_token_id=tok.pad_token_id)
        g = [t for t in o[0, npre:].tolist() if t != tok.pad_token_id]
        with torch.no_grad():
            lg = model(torch.tensor([pre["input_ids"][0].tolist() + g],
                                    device=a.device)).logits[0].float()
        P = torch.softmax(lg[npre - 1:-1], -1)
        H = (-(P * P.clamp_min(1e-12).log()).sum(-1)).tolist()
        W = a.baseline
        prom = [(t, H[t] - sorted(H[t - W:t])[W // 2]) for t in range(W + 1, len(H))]
        cut = sorted((v for _, v in prom), reverse=True)[max(0, int(len(prom) * 0.02))]
        root = min(t for t, v in prom if v >= cut)
        print(f"  first spike at token {root}: H {H[root]:.2f} vs local "
              f"{sorted(H[root-W:root])[W//2]:.2f}")
        print(f"  fixed prefix ends: ...{tok.decode(g[max(0,root-45):root])!r}\n")

        leaves = []

        def walk(ids, logp, depth, path):
            walked = 0
            while True:
                h, cands = dist(ids)
                keep = [c for c in cands if c[2] >= M_MIN]
                if len(keep) >= 2 and depth < a.depth:
                    print(f"{'  '*depth}fork after {walked} certain tokens, H={h:.2f}: "
                          f"{[(c[1], round(c[2],3)) for c in keep]}")
                    for cid, ctxt, cp in keep:
                        walk(ids + [cid], logp + math.log(cp), depth + 1, path + [(ctxt, cp)])
                    return
                if walked >= a.follow or not cands:
                    leaves.append({"ids": list(ids), "logp": logp, "path": list(path)})
                    return
                ids = ids + [cands[0][0]]
                walked += 1

        walk(pre["input_ids"][0].tolist() + g[:root], 0.0, 0, [])
        print(f"\n  leaves: {len(leaves)}")
        print(f"\n  {'committed path':<46}{'P(path)':>9}{'correct':>10}")
        for lf in leaves:
            t0 = torch.tensor([lf["ids"]], device=a.device)
            with torch.no_grad():
                oo = model.generate(input_ids=t0.repeat(a.conts, 1),
                                    attention_mask=torch.ones_like(t0).repeat(a.conts, 1),
                                    do_sample=True, temperature=a.temp, top_p=1.0, top_k=0,
                                    max_new_tokens=a.max_new, eos_token_id=stops,
                                    pad_token_id=tok.pad_token_id)
            texts, ok = [], 0
            for j in range(a.conts):
                tx = tok.decode([t for t in oo[j, len(lf["ids"]):].tolist()
                                 if t != tok.pad_token_id], skip_special_tokens=True)
                texts.append(tx)
                ok += (norm(boxed(tx)) == norm(p["answer"]))
            label = "".join(c for c, _ in lf["path"])
            print(f"  {label[:44]!r:<46}{math.exp(lf['logp']):>9.4f}{ok:>7}/{a.conts}")
            print(f"      {(label + texts[0])[:165]!r}")
            out_f.write(json.dumps({
                "problem_id": pid, "answer": p["answer"], "model": a.model,
                "spike_pos": root, "prefix": tok.decode(g[:root]),
                "path": lf["path"], "path_prob": math.exp(lf["logp"]),
                "correct": ok, "n": a.conts,
                "completions": [t[:2000] for t in texts]}, ensure_ascii=False) + "\n")
            out_f.flush()
    out_f.close()
    print(f"\nwrote {OUT}/tree_{a.tag}.jsonl")


if __name__ == "__main__":
    main()
