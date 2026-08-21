"""Show, for a few concrete states, what the floor actually does token by token.

Picks representative states from the measured run (opening, expensive internal,
typical internal, free internal) and prints pi_A, pi_T, the floor, q*, and each
token's contribution to KL(q*||pi_T).
"""
import json
import os

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from WorkPlace.Distill_start.Experiment.PreAnalysis.common import ANCHOR, OUT, TEACHER
from WorkPlace.Distill_start.Experiment.PreAnalysis.cost_beta import project

BETA = 0.4


def show(tok, log_pT, log_pA, title, prefix_text, cost, n_show=8):
    log_c, _ = __import__("cost_beta").solve_log_c(log_pT, log_pA, BETA)
    import math
    log_floor = math.log(BETA) + log_pA
    log_q = torch.maximum(log_c + log_pT, log_floor)
    log_q = log_q - torch.logsumexp(log_q, -1, keepdim=True)
    q, pT, pA = log_q.exp()[0], log_pT.exp()[0], log_pA.exp()[0]
    binding = (log_floor > log_c + log_pT)[0]
    contrib = q * (log_q[0] - log_pT[0])

    print("\n" + "=" * 96)
    print(f"### {title}")
    print(f"prefix ends with: ...{prefix_text!r}")
    print(f"H(teacher) = {-(pT * log_pT[0]).sum():.3f}   H(anchor) = {-(pA * log_pA[0]).sum():.3f}"
          f"   KL(q*||pi_T) = {cost:.4f} nats   c = {log_c.exp().item():.4f}")
    cand = torch.unique(torch.cat([pT.topk(n_show).indices, pA.topk(n_show).indices]))
    cand = cand[q[cand].argsort(descending=True)]
    print(f"\n{'token':<16}{'pi_A':>10}{'pi_T':>12}{'floor=b*pi_A':>14}{'q*':>10}{'bind':>6}{'KL share':>11}")
    print("-" * 96)
    for v in cand[:12]:
        v = int(v)
        print(f"{repr(tok.decode([v]))[:15]:<16}{pA[v]:>10.4f}{pT[v]:>12.2e}{BETA*pA[v]:>14.4f}"
              f"{q[v]:>10.4f}{('  yes' if binding[v] else '   no'):>6}"
              f"{contrib[v]/max(cost,1e-12):>10.1%}")
    print(f"{'(all others)':<16}{'':>10}{'':>12}{'':>14}"
          f"{q.sum()-q[cand[:12]].sum():>10.4f}{'':>6}"
          f"{(cost-contrib[cand[:12]].sum())/max(cost,1e-12):>10.1%}")
    print(f"binding tokens: {int(binding.sum())} of {len(binding)};  "
          f"mass on the floor: {(q*binding).sum():.4f}")


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sfx = f".{args.tag}" if args.tag else ""
    df = pd.read_parquet(os.path.join(OUT, f"cost_states{sfx}.parquet"))
    d = df[df.beta == BETA].set_index("state_id")
    rollouts = {r["rollout_id"]: r for r in
                (json.loads(l) for l in open(os.path.join(OUT, f"rollouts{sfx}.jsonl")))}
    states = {s["state_id"]: s for s in
              (json.loads(l) for l in open(os.path.join(OUT, f"states{sfx}.jsonl")))}

    internal = d[d.kind == "internal"].sort_values("cost")
    picks = [
        (int(d[d.kind == "opening"].index[0]), "OPENING STATE (position 0) -- the format ritual"),
        (int(internal.index[-1]), f"INTERNAL STATE, most expensive of {len(internal)}"),
        (int(internal.index[len(internal) // 2]), "INTERNAL STATE, median cost -- the typical case"),
        (int(internal.index[len(internal) // 20]), "INTERNAL STATE, 5th percentile -- floor idle"),
    ]

    tok = AutoTokenizer.from_pretrained(TEACHER)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16, device_map="cuda").eval()
    anchor = AutoModelForCausalLM.from_pretrained(ANCHOR, dtype=torch.bfloat16, device_map="cuda").eval()

    def lp(model, ids):
        with torch.no_grad():
            o = model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1)
        return torch.log_softmax(o.logits[:, -1, :].float(), -1)

    for sid, title in picks:
        st = states[sid]
        r = rollouts[st["rollout_id"]]
        ids = r["prompt_ids"] + r["output_ids"][: st["pos"]]
        txt = tok.decode(ids[-24:]) if st["pos"] else tok.decode(ids[-14:])
        show(tok, lp(teacher, ids), lp(anchor, ids), f"{title}  [pos {st['pos']}]",
             txt, float(d.loc[sid, "cost"]))


if __name__ == "__main__":
    main()
