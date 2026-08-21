"""What fills the slot the spike opens?

The spike marks where the anchor stops being certain, but the token at the spike is a function
word: "we can use", "we need to consider". The method is the word that follows it. So the spike is
the onset of a choice, not the choice itself, and reading only the spike position mistakes the
grammar for the content.

This walks forward from the spike, forcing each candidate in turn and reading the next
distribution, so the tree shows which openings lead to a real slot (many content-bearing
candidates) and which merely continue boilerplate ("proceed" -> "with" -> "the following steps").
"""
import argparse, json, os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.abspath(__file__))
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
M_MIN = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", type=int, default=180)
    ap.add_argument("--rollouts", type=int, default=6)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--baseline", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--device", default="cuda:3")
    a = ap.parse_args()

    sys.path.insert(0, ROOT)
    import or_common
    p = {q["id"]: q for q in or_common.load_problems()}[a.problem]
    tc = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base", dtype=torch.bfloat16).to(a.device).eval()
    stops = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
             if isinstance(i, int) and i >= 0]

    prompt = tc.apply_chat_template(
        [{"role": "user", "content": p["problem"] + "\n\n" + INSTRUCTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pre = tok(prompt, return_tensors="pt").to(a.device)
    npre = pre["input_ids"].shape[1]
    print(f"PROBLEM {a.problem}: {p['problem'][:120]}   gold={p['answer']}\n")

    with torch.no_grad():
        out = model.generate(**{k: v.repeat(a.rollouts, 1) for k, v in pre.items()},
                             do_sample=True, temperature=a.temp, top_p=1.0, top_k=0,
                             max_new_tokens=400, eos_token_id=stops,
                             pad_token_id=tok.pad_token_id)
    g0 = [t for t in out[0, npre:].tolist() if t != tok.pad_token_id]

    def dist(ids):
        with torch.no_grad():
            lg = model(torch.tensor([ids], device=a.device)).logits[0, -1].float()
        P = torch.softmax(lg, -1)
        H = float(-(P * P.clamp_min(1e-12).log()).sum())
        tp = torch.topk(P, 8)
        return H, [(int(i), tok.decode([int(i)]), round(float(v), 3))
                   for v, i in zip(tp.values, tp.indices)]

    # locate the first local entropy spike on the trunk
    ids = pre["input_ids"][0].tolist() + g0
    with torch.no_grad():
        lg = model(torch.tensor([ids], device=a.device)).logits[0].float()
    P = torch.softmax(lg[npre - 1:-1], -1)
    H = (-(P * P.clamp_min(1e-12).log()).sum(-1)).tolist()
    W = a.baseline
    # The FIRST spike, not the largest one: a later spike sits in text whose method is already
    # committed, so it can only be choosing a section title. Rank positions by how far entropy
    # jumps above its local level, keep the top 2%, and take the earliest of those.
    prom = [(t, H[t] - sorted(H[t - W:t])[W // 2]) for t in range(W + 1, len(H))]
    cut = sorted((v for _, v in prom), reverse=True)[max(0, int(len(prom) * 0.02))]
    best = min(t for t, v in prom if v >= cut)
    print(f"first spike at position {best}: H={H[best]:.2f} vs local "
          f"{sorted(H[best-W:best])[W//2]:.2f}")
    print(f"context: ...{tok.decode(g0[max(0,best-30):best])!r}\n")
    print("=" * 92)
    print("WALKING FORWARD FROM THE SPIKE  (indent = depth; H is the entropy at that point)")
    print("=" * 92)

    base_ids = pre["input_ids"][0].tolist() + g0[:best]

    def walk(ids, depth, path):
        h, cands = dist(ids)
        keep = [c for c in cands if c[2] >= M_MIN] or cands[:2]
        for cid, txt, pv in keep:
            slot = "  <-- SLOT" if depth < a.depth else ""
            print(f"{'    '*depth}{txt!r} p={pv}")
            if depth + 1 < a.depth:
                h2, c2 = dist(ids + [cid])
                tag = "content slot" if h2 > 1.5 else "boilerplate"
                print(f"{'    '*(depth+1)}[next H={h2:.2f} -> {tag}] "
                      f"{[(c[1], c[2]) for c in c2[:6]]}")
                walk(ids + [cid], depth + 1, path + [txt])

    walk(base_ids, 0, [])

    print("\n" + "=" * 92)
    print("COMPLETING EACH SLOT FILLER  (4 continuations each, correctness vs gold)")
    print("=" * 92)
    h, cands = dist(base_ids)
    for cid, txt, pv in [c for c in cands if c[2] >= M_MIN]:
        h2, c2 = dist(base_ids + [cid])
        if h2 <= 1.5:
            print(f"\n{txt!r} (p={pv}) -> next H={h2:.2f}, boilerplate; skipping")
            continue
        print(f"\n{txt!r} (p={pv}) -> next H={h2:.2f}, real slot. fillers:")
        for c2id, c2txt, c2p in [c for c in c2 if c[2] >= M_MIN]:
            seq = base_ids + [cid, c2id]
            t0 = torch.tensor([seq], device=a.device)
            with torch.no_grad():
                o = model.generate(input_ids=t0.repeat(4, 1),
                                   attention_mask=torch.ones_like(t0).repeat(4, 1),
                                   do_sample=True, temperature=a.temp, top_p=1.0, top_k=0,
                                   max_new_tokens=420, eos_token_id=stops,
                                   pad_token_id=tok.pad_token_id)
            outs = [tok.decode([t for t in o[j, len(seq):].tolist()
                                if t != tok.pad_token_id], skip_special_tokens=True)
                    for j in range(4)]
            ok = sum(("\\boxed{\\sqrt{2}}" in t.replace(" ", "")
                      or "\\boxed{\\sqrt2}" in t.replace(" ", "")) for t in outs)
            print(f"    {c2txt!r} p={c2p}  correct {ok}/4")
            print(f"        {(txt + c2txt + outs[0])[:190]!r}")


if __name__ == "__main__":
    main()
