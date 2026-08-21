"""Does the prompt format decide the answer, or only the first few tokens?

Prefill scoring compares two checkpoints on ONE fixed string, and the prefix is part of that
string. So the two checkpoints must be given the SAME prefix, or the subtraction that cancels
style and difficulty cancels nothing. That leaves a choice: whose home format?

  chat      the teacher's chat template, what Cal_Beta and Cal_E used. Home for a distilled
            checkpoint, foreign to Base, which never saw a template.
  plain     the problem plus the instruction and nothing else. Home for Base, foreign to a
            checkpoint trained to open with a think block.

Either choice looks like it favours somebody, which is the worry. The question this settles is
whether the favour survives past the opening tokens. Score every traj under both formats with
both checkpoints and compare the arm difference: if the two formats agree once the format tokens
are dropped, the choice is immaterial and the rule is simply "exclude the opening".
"""
import argparse, json, os, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
TEACHER_TOKENIZER = "Qwen/Qwen3-14B"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def prefixes(tok_chat, problem):
    plain = problem + "\n\n" + INSTRUCTION
    chat = tok_chat.apply_chat_template(
        [{"role": "user", "content": plain}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    return {"plain": plain, "chat": chat}


@torch.no_grad()
def score(model, tok, prefix, text, device):
    pre = tok(prefix, add_special_tokens=False).input_ids
    txt = tok(text, add_special_tokens=False).input_ids
    if not txt:
        return None
    ids = torch.tensor([pre + txt], device=device)
    logits = model(ids).logits[0].float()
    lp = torch.log_softmax(logits[len(pre) - 1:-1], dim=-1)
    tgt = torch.tensor(txt, device=device)
    return [round(float(v), 4) for v in lp.gather(1, tgt.unsqueeze(1)).squeeze(1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--limit", type=int, default=0, help="0 = all traj")
    ap.add_argument("--out", default=os.path.join(OUT, "prompt_format_test.jsonl"))
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(OUT, "master_traj.jsonl"))]
    if a.limit:
        rows = rows[::max(1, len(rows) // a.limit)][:a.limit]
    import sys
    sys.path.insert(0, ROOT)
    import or_common
    probs = {p["id"]: p["problem"] for p in or_common.load_problems()}

    tok_chat = AutoTokenizer.from_pretrained(TEACHER_TOKENIZER)
    pre = {pid: prefixes(tok_chat, txt) for pid, txt in probs.items()}
    print(f"{len(rows)} traj, 2 formats x 2 checkpoints")
    print(f"plain prefix ends: {pre[rows[0]['problem_id']]['plain'][-60:]!r}")
    print(f"chat  prefix ends: {pre[rows[0]['problem_id']]['chat'][-60:]!r}\n")

    res = [{} for _ in rows]
    for name, mid in [("base", "Qwen/Qwen3-1.7B-Base"), ("post", "Qwen/Qwen3-1.7B")]:
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.bfloat16).to(a.device).eval()
        for fmt in ("plain", "chat"):
            t0 = time.time()
            for i, r in enumerate(rows):
                res[i][f"{name}_{fmt}"] = score(
                    model, tok, pre[r["problem_id"]][fmt], r["reasoning"], a.device)
                if (i + 1) % 500 == 0:
                    print(f"  {name}/{fmt}: {i+1}/{len(rows)}", flush=True)
            print(f"  {name}/{fmt} done in {time.time()-t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    with open(a.out, "w") as f:
        for r, d in zip(rows, res):
            f.write(json.dumps({
                "problem_id": r["problem_id"], "source_model": r["source_model"],
                "sample_k": r["sample_k"], "approach": r["approach"],
                "n_tokens": r["n_tokens"], **d}) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
