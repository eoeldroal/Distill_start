"""Anchor habitat measurement: does Base spread mass over several solution methods?

Experiments section 1.3 constraint 4 assumes the phenomenon lives at anchor mass 0.05-0.2. That
assumption has never been measured, and it decides whether there is anything for the floor to
protect. This script produces the material: Base's own rollouts under the canonical sampler, to
be labelled by the blind pipeline afterwards.

Temperature is swept rather than assumed. T=1.0 is the value section 0's own rationale implies,
since only there does the rollout distribution equal the policy distribution; the other points
measure what raising temperature buys and where the text falls apart, which section 4 needs
anyway for the recovery analysis.

Prompt is the teacher chat template for every checkpoint, matching the prefill scoring and the
eventual RL harness. Base never saw a template, but it copes (40% correct on these problems in
the pre-analysis rollouts) and the template widens rather than narrows its opening.
"""
import argparse, json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
TEACHER_TOKENIZER = "Qwen/Qwen3-14B"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
PANEL = [3, 28, 34, 35, 44, 53, 54, 68, 100, 114, 115, 127, 154, 155, 158, 159, 160, 180, 182, 195]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--temps", default="0.7,1.0,1.3,1.6,2.0")
    ap.add_argument("--n", type=int, default=24, help="samples per (problem, temperature)")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--problems", default="")
    a = ap.parse_args()

    sys.path.insert(0, ROOT)
    import or_common
    probs = {p["id"]: p for p in or_common.load_problems()}
    ids = [int(x) for x in a.problems.split(",")] if a.problems else PANEL
    temps = [float(x) for x in a.temps.split(",")]

    tc = AutoTokenizer.from_pretrained(TEACHER_TOKENIZER)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).eval()

    # Base may close with either marker; accept both so length is the model's choice.
    stops = [i for i in {tok.eos_token_id,
                         tok.convert_tokens_to_ids("<|im_end|>"),
                         tok.convert_tokens_to_ids("<|endoftext|>")} if isinstance(i, int) and i >= 0]
    print(f"model {a.model} | stop ids {stops} | temps {temps} | n={a.n} | "
          f"{len(ids)} problems | max_new {a.max_new}", flush=True)

    path = os.path.join(OUT, f"habitat_{a.tag}.jsonl")
    f = open(path, "w")
    t_start = time.time()
    for pid in ids:
        p = probs[pid]
        prompt = tc.apply_chat_template(
            [{"role": "user", "content": p["problem"] + "\n\n" + INSTRUCTION}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        enc = tok([prompt], return_tensors="pt").to(a.device)
        for T in temps:
            done = 0
            t0 = time.time()
            while done < a.n:
                b = min(a.batch, a.n - done)
                with torch.no_grad():
                    out = model.generate(
                        **{k: v.repeat(b, 1) for k, v in enc.items()},
                        do_sample=True, temperature=T, top_p=1.0, top_k=0,
                        max_new_tokens=a.max_new, eos_token_id=stops,
                        pad_token_id=tok.pad_token_id)
                gen = out[:, enc["input_ids"].shape[1]:]
                for j in range(b):
                    seq = gen[j].tolist()
                    trimmed = [t for t in seq if t != tok.pad_token_id]
                    hit = any(t in stops for t in trimmed)
                    while trimmed and trimmed[-1] in stops:
                        trimmed.pop()
                    f.write(json.dumps({
                        "problem_id": pid, "level": p.get("level"), "type": p.get("type"),
                        "answer": p.get("answer"), "temperature": T, "sample_k": done + j,
                        "text": tok.decode(trimmed, skip_special_tokens=False),
                        "n_tokens": len(trimmed),
                        "finish": "stop" if hit else "length"}, ensure_ascii=False) + "\n")
                done += b
                f.flush()
            print(f"  p{pid} T={T}: {a.n} samples in {time.time()-t0:.0f}s "
                  f"(elapsed {(time.time()-t_start)/60:.1f}m)", flush=True)
    f.close()
    print(f"\nwrote {path} in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
