"""Score every discovery trajectory under a checkpoint by prefill.

No generation: each trajectory is a fixed string, and the model only assigns
probabilities to it. That is what makes the comparison clean across checkpoints
-- every model is scored on exactly the same text, so nothing about how a model
prefers to write can leak into the number.

Raw log-probabilities are reported, unnormalised. The guarantee this research
makes is q*(v) >= beta * pi_A(v), a statement about raw probabilities, and the
retention law beta^k_bind is a ratio of raw probabilities, so the measurement
stays in those units.

Per-position log-probs are kept alongside the totals because the chat template
is applied to the Base model too, which does not know the <think> opening. That
cost lands in the first few positions and has to be separable from the rest.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
TEACHER_TOKENIZER = "Qwen/Qwen3-14B"   # chat template source, shared by all arms
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prefix(tok, problem):
    """Teacher chat template, identical for every checkpoint including Base."""
    return tok.apply_chat_template(
        [{"role": "user", "content": problem + "\n\n" + INSTRUCTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)


@torch.no_grad()
def score_batch(model, tok, prefixes, texts, device, keep_positions):
    """log P(text | prefix) for each pair, plus the first keep_positions token logprobs."""
    outs = []
    for prefix, text in zip(prefixes, texts):
        pre_ids = tok(prefix, add_special_tokens=False).input_ids
        txt_ids = tok(text, add_special_tokens=False).input_ids
        if not txt_ids:
            outs.append((float("nan"), [], 0))
            continue
        ids = torch.tensor([pre_ids + txt_ids], device=device)
        logits = model(ids).logits[0].float()
        # logits at position i predict token i+1
        lp = torch.log_softmax(logits[len(pre_ids) - 1:-1], dim=-1)
        tgt = torch.tensor(txt_ids, device=device)
        tok_lp = lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        outs.append((float(tok_lp.sum()),
                     [round(float(v), 4) for v in tok_lp],
                     txt_ids))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", default=os.path.join(OUT, "discovery_pilot_v2.jsonl"))
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--keep-positions", type=int, default=0,
                    help="unused; every token logprob is kept")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.input)]
    rows = [r for r in rows if "error" not in r and (r.get("reasoning") or "").strip()]
    print(f"{len(rows)} trajectories to score under {a.model}", flush=True)

    sys.path.insert(0, ROOT)
    import or_common
    problems = {p["id"]: p["problem"] for p in or_common.load_problems()}

    ttok = AutoTokenizer.from_pretrained(TEACHER_TOKENIZER)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map=a.device).eval()

    prefix_cache = {pid: build_prefix(ttok, p) for pid, p in problems.items()}
    # the chat template must tokenise identically under both tokenizers
    for pid, pfx in list(prefix_cache.items())[:5]:
        assert ttok(pfx).input_ids == tok(pfx).input_ids, f"tokenizer mismatch at {pid}"
    print("tokenizer check: teacher template tokenises identically under this model")

    t0 = time.time()
    out_path = os.path.join(OUT, f"prefill_{a.tag}.jsonl")
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            pfx = prefix_cache[r["problem_id"]]
            (total, tok_lp, txt_ids), = score_batch(model, tok, [pfx], [r["reasoning"]],
                                                     a.device, a.keep_positions)
            f.write(json.dumps({
                "model_scored": a.model,
                "problem_id": r["problem_id"],
                "source_model": r["model"],
                "sample_k": r["sample_k"],
                "level": r.get("level"), "type": r.get("type"),
                "n_tokens": len(txt_ids),
                "logp_total": round(total, 4),
                "logp_tokens": tok_lp,
                "token_ids": txt_ids,
            }) + "\n")
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(rows)}  {el:.0f}s  eta {el/(i+1)*(len(rows)-i-1):.0f}s",
                      flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
