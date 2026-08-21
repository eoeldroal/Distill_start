"""Window-level retention: does the vanilla target hard-kill the anchor's actual
opening modes, and how much does the floor rescue?

For each anchor rollout's first 64 generated tokens, score every position under both
frozen models and build the floored target q*. Cumulative products along the window
give, for the literal sequence the anchor actually wrote:

  P_T(window)  -- probability the vanilla target (= teacher) leaves on it
  Q*(window)   -- probability the floored target leaves on it

Both are lower bounds on branch-level mass (a branch is a cluster of many surface
forms), but their ratio is the retention lift, and the count of binding positions is
the empirical k_bind on real entry windows.
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from WorkPlace.ICLR.Experiment.PreAnalysis.common import ANCHOR, OUT, TEACHER
from WorkPlace.ICLR.Experiment.PreAnalysis.cost_beta import solve_log_c

BETAS = [0.1, 0.2, 0.4, 0.8]
W = 64  # window length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rollouts = [json.loads(l) for l in open(os.path.join(OUT, "rollouts.jsonl"))]
    rollouts = [r for r in rollouts if len(r["output_ids"]) >= W]
    if args.limit:
        rollouts = rollouts[: args.limit]
    print(f"{len(rollouts)} rollouts with >= {W} generated tokens")

    device = "cuda"
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16, device_map=device).eval()
    anchor = AutoModelForCausalLM.from_pretrained(ANCHOR, dtype=torch.bfloat16, device_map=device).eval()

    def window_logprobs(model, batch):
        """log-softmax at the W positions predicting each generated window token."""
        seqs = [r["prompt_ids"] + r["output_ids"][:W] for r in batch]
        n = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), n, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):  # left pad -> window is the last W tokens for all
            ids[i, n - len(s):] = torch.tensor(s, device=device)
            mask[i, n - len(s):] = 1
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        # positions n-W-1 .. n-2 predict tokens at n-W .. n-1
        logits = out.logits[:, n - W - 1 : n - 1, :].float()
        return torch.log_softmax(logits, dim=-1), ids[:, n - W :]

    recs = []
    for start in range(0, len(rollouts), args.batch_size):
        batch = rollouts[start : start + args.batch_size]
        log_pT, toks = window_logprobs(teacher, batch)   # [B, W, V], [B, W]
        log_pA, _ = window_logprobs(anchor, batch)
        B, Wn, V = log_pT.shape
        flat_T, flat_A = log_pT.reshape(B * Wn, V), log_pA.reshape(B * Wn, V)
        tok_flat = toks.reshape(B * Wn, 1)

        lpT_tok = flat_T.gather(1, tok_flat).reshape(B, Wn)
        lpA_tok = flat_A.gather(1, tok_flat).reshape(B, Wn)

        per_beta = {}
        for beta in BETAS:
            log_c, log_floor = solve_log_c(flat_T, flat_A, beta)
            log_q = torch.maximum(log_c + flat_T, log_floor)
            log_q = log_q - torch.logsumexp(log_q, -1, keepdim=True)
            lq_tok = log_q.gather(1, tok_flat).reshape(B, Wn)
            bind_tok = (log_floor.gather(1, tok_flat) > (log_c + flat_T.gather(1, tok_flat))).reshape(B, Wn)
            # how far q* is from the arithmetic mixture at each position (full vocab)
            log_mix = torch.logaddexp(np.log(1 - beta) + flat_T, np.log(beta) + flat_A)
            kl_qm = (log_q.exp() * (log_q - log_mix)).sum(-1).reshape(B, Wn)
            per_beta[beta] = (lq_tok.cpu(), bind_tok.cpu(), kl_qm.cpu())

        for i, r in enumerate(batch):
            rec = {
                "rollout_id": r["rollout_id"], "problem_id": r["problem_id"],
                "lpT": lpT_tok[i].cpu().numpy(), "lpA": lpA_tok[i].cpu().numpy(),
            }
            for beta in BETAS:
                lq, bd, km = per_beta[beta]
                rec[f"lq_{beta}"] = lq[i].numpy()
                rec[f"bind_{beta}"] = bd[i].numpy()
                rec[f"klmix_{beta}"] = km[i].numpy()
            recs.append(rec)
        done = min(start + args.batch_size, len(rollouts))
        if done % 40 == 0 or done == len(rollouts):
            print(f"  {done}/{len(rollouts)}", flush=True)

    np.savez_compressed(
        os.path.join(OUT, "window_retention.npz"),
        **{k: np.stack([r[k] for r in recs]) for k in recs[0] if k not in ("rollout_id", "problem_id")},
        rollout_id=np.array([r["rollout_id"] for r in recs]),
        problem_id=np.array([r["problem_id"] for r in recs]),
    )
    print(f"wrote outputs/window_retention.npz  ({len(recs)} windows)")


if __name__ == "__main__":
    main()
