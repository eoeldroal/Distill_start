"""Step 3: measure Cost(beta) = mean KL(q* || pi_T) and the binding profile.

q*(v) = max(c * pi_T(v), beta * pi_A(v)), with c set so the distribution sums to 1.
Everything is done in log space: teacher probabilities on crushed tokens underflow
float32, and those are exactly the tokens the floor acts on.

Outputs: outputs/cost_states.parquet (per state x beta), outputs/topk.npz
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from WorkPlace.Distill_start.Experiment.PreAnalysis.common import ANCHOR, OUT, TEACHER

BETAS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95
for b in (0.1, 0.2, 0.4, 0.8):                      # sweep grid points
    if b not in BETAS:
        BETAS.append(b)
BETAS = sorted(BETAS)


def solve_log_c(log_pT, log_pA, beta, iters=60):
    """Bisect log c so that sum_v max(c*pi_T, beta*pi_A) == 1. Vectorized over states."""
    log_floor = math.log(beta) + log_pA
    lo = torch.full((log_pT.shape[0], 1), -60.0, device=log_pT.device, dtype=log_pT.dtype)
    hi = torch.zeros_like(lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        total = torch.logsumexp(torch.maximum(mid + log_pT, log_floor), dim=-1, keepdim=True)
        too_big = total > 0
        hi = torch.where(too_big, mid, hi)
        lo = torch.where(too_big, lo, mid)
    return (lo + hi) / 2, log_floor


def project(log_pT, log_pA, beta, top1=None):
    """Return (cost, n_bind, mass_bind, log_c) for one beta over a batch of states.

    With top1 (index of the teacher's argmax) also returns how much of that token the
    target keeps -- at position 0 under a thinking teacher this is q*(<think>), i.e.
    how often the distilled student would still open in the teacher's format.
    """
    log_c, log_floor = solve_log_c(log_pT, log_pA, beta)
    log_q = torch.maximum(log_c + log_pT, log_floor)
    log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)  # kill bisection residue
    q = log_q.exp()
    cost = (q * (log_q - log_pT)).sum(-1)
    binding = log_floor > (log_c + log_pT)
    out = (cost, binding.sum(-1), (q * binding).sum(-1), log_c.squeeze(-1))
    if top1 is None:
        return out
    return out + (q.gather(1, top1[:, None]).squeeze(1),)


def self_test(device):
    """Reproduce the 4-token toy in Document/toy_sims/floor_vs_kl.py."""
    pT = torch.tensor([[0.85, 0.14, 0.008, 0.002]], dtype=torch.float64, device=device)
    pA = torch.tensor([[0.50, 0.30, 0.15, 0.05]], dtype=torch.float64, device=device)
    log_pT, log_pA = pT.log(), pA.log()

    cost, n_bind, mass_bind, log_c = project(log_pT, log_pA, 0.4)
    log_q = torch.maximum(log_c[:, None] + log_pT, math.log(0.4) + log_pA)
    q = log_q.exp()
    odds = (q[0, 0] / q[0, 1]).item()

    print("self-test vs toy_sims/floor_vs_kl.py (beta=0.4)")
    print(f"  q*            = {[round(x, 4) for x in q[0].tolist()]}  (expect [0.79, 0.13, 0.06, 0.02])")
    print(f"  sum q*        = {q.sum().item():.10f}  (expect 1)")
    print(f"  odds A:B      = {odds:.3f}  (expect 6.071 = teacher odds, preserved)")
    print(f"  KL(q*||pi_T)  = {cost.item():.4f}  (expect 0.0995)")
    print(f"  n_bind        = {int(n_bind.item())}  (expect 2: tokens C and D)")

    ok = (abs(cost.item() - 0.0995) < 1e-3 and abs(odds - 6.071) < 1e-2
          and abs(q.sum().item() - 1) < 1e-9 and int(n_bind.item()) == 2)

    zero, _, _, _ = project(log_pT, log_pA, 1e-12)
    mono = [project(log_pT, log_pA, b)[0].item() for b in BETAS]
    inc = all(mono[i] <= mono[i + 1] + 1e-12 for i in range(len(mono) - 1))
    print(f"  beta->0 cost  = {zero.item():.2e}  (expect ~0)")
    print(f"  monotone in beta: {inc}")
    ok = ok and zero.item() < 1e-6 and inc
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--self-test-only", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not self_test(device):
        raise SystemExit("self-test failed; not running the measurement")
    if args.self_test_only:
        return

    from transformers import AutoModelForCausalLM

    suffix = f".{args.tag}" if args.tag else ""
    rollouts = {r["rollout_id"]: r for r in
                (json.loads(l) for l in open(os.path.join(OUT, f"rollouts{suffix}.jsonl")))}
    states = [json.loads(l) for l in open(os.path.join(OUT, f"states{suffix}.jsonl"))]
    for s in states:
        r = rollouts[s["rollout_id"]]
        s["ids"] = r["prompt_ids"] + r["output_ids"][: s["pos"]]
        s["problem_id"] = r["problem_id"]
    states.sort(key=lambda s: len(s["ids"]))  # length-bucketed batches

    print(f"\nloading models on {device} (bf16)")
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16, device_map=device).eval()
    anchor = AutoModelForCausalLM.from_pretrained(ANCHOR, dtype=torch.bfloat16, device_map=device).eval()

    def last_logprobs(model, batch):
        n = max(len(s["ids"]) for s in batch)
        ids = torch.zeros(len(batch), n, dtype=torch.long, device=device)
        mask = torch.zeros(len(batch), n, dtype=torch.long, device=device)
        for i, s in enumerate(batch):  # left pad: last position is the state
            ids[i, n - len(s["ids"]):] = torch.tensor(s["ids"], device=device)
            mask[i, n - len(s["ids"]):] = 1
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
        return torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)

    rows, topk_store = [], []
    for start in range(0, len(states), args.batch_size):
        batch = states[start : start + args.batch_size]
        log_pT = last_logprobs(teacher, batch)
        log_pA = last_logprobs(anchor, batch)

        t1 = log_pT.argmax(-1)
        pT_top1 = log_pT.gather(1, t1[:, None]).squeeze(1).exp()
        for beta in BETAS:
            cost, n_bind, mass_bind, log_c, q_top1 = project(log_pT, log_pA, beta, t1)
            for i, s in enumerate(batch):
                rows.append(
                    {
                        "state_id": s["state_id"], "problem_id": s["problem_id"],
                        "rollout_id": s["rollout_id"], "pos": s["pos"], "kind": s["kind"],
                        "beta": beta, "cost": cost[i].item(), "n_bind": int(n_bind[i].item()),
                        "mass_bind": mass_bind[i].item(), "log_c": log_c[i].item(),
                        "pT_top1": pT_top1[i].item(), "q_at_Ttop1": q_top1[i].item(),
                    }
                )

        # Agreement diagnostics + a small slice of both distributions for inspection.
        tv, ti = log_pT.topk(args.topk, dim=-1)
        av, ai = log_pA.topk(args.topk, dim=-1)
        agree = (ti[:, 0] == ai[:, 0])
        pA_at_T1 = log_pA.gather(1, ti[:, :1]).squeeze(1)
        pT_at_A1 = log_pT.gather(1, ai[:, :1]).squeeze(1)
        for i, s in enumerate(batch):
            topk_store.append(
                {
                    "state_id": s["state_id"], "kind": s["kind"], "pos": s["pos"],
                    "T_ids": ti[i].cpu().numpy(), "T_logp": tv[i].cpu().numpy(),
                    "A_ids": ai[i].cpu().numpy(), "A_logp": av[i].cpu().numpy(),
                    "argmax_agree": bool(agree[i].item()),
                    "logpA_at_Ttop1": pA_at_T1[i].item(), "logpT_at_Atop1": pT_at_A1[i].item(),
                    "H_T": -(log_pT[i].exp() * log_pT[i]).sum().item(),
                    "H_A": -(log_pA[i].exp() * log_pA[i]).sum().item(),
                }
            )
        done = min(start + args.batch_size, len(states))
        if done % (args.batch_size * 10) == 0 or done == len(states):
            print(f"  {done}/{len(states)} states", flush=True)

    import pandas as pd

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, f"cost_states{suffix}.parquet")
    df.to_parquet(path, index=False)
    print(f"\nwrote {len(df)} rows ({df.state_id.nunique()} states x {len(BETAS)} betas) -> {path}")

    np.savez_compressed(
        os.path.join(OUT, f"topk{suffix}.npz"),
        **{k: np.array([t[k] for t in topk_store]) for k in topk_store[0]},
    )
    print("Cost(beta) by state kind (unweighted means):")
    kinds = [k for k in ("opening", "early", "internal") if (df.kind == k).any()]
    print("  beta  | " + " | ".join(f"{k:>9}" for k in kinds))
    for b in (0.1, 0.2, 0.4, 0.8):
        sub = df[df.beta == b]
        print(f"  {b:<5} | " + " | ".join(f"{sub[sub.kind==k].cost.mean():>9.4f}" for k in kinds))
    print("\nq*(teacher argmax) at pos 0 -- format retention of the distilled target:")
    op = df[df.kind == "opening"]
    for b in (0.1, 0.2, 0.4, 0.8):
        print(f"  beta={b}: {op[op.beta==b].q_at_Ttop1.mean():.3f}")


if __name__ == "__main__":
    main()
