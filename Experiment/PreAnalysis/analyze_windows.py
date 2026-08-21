"""Aggregate window_retention.npz into the retention story."""
import json
import os

import numpy as np

from WorkPlace.ICLR.Experiment.PreAnalysis.common import OUT

BETAS = [0.1, 0.2, 0.4, 0.8]
BUDGET = 1 / 80  # run-level budget floor: 1/(G*T_eff)
LOG10 = np.log(10)

d = np.load(os.path.join(OUT, "window_retention.npz"))
lpT, lpA = d["lpT"], d["lpA"]          # [N, 64] natural-log probs of the taken token
N = lpT.shape[0]
print(f"{N} anchor opening windows, 64 tokens each\n")

def cum(x, lo, hi):  # cumulative log10 prob over positions [lo, hi)
    return x[:, lo:hi].sum(1) / LOG10

def q(x, ps=(10, 25, 50, 75, 90)):
    return " ".join(f"p{p}={np.percentile(x, p):7.2f}" for p in ps)

print("=== 1. Crush depth of the VANILLA target (log10 P_T of the anchor's own window) ===")
print(f"  full window [0,64):     {q(cum(lpT, 0, 64))}")
print(f"  first 32    [0,32):     {q(cum(lpT, 0, 32))}")
print(f"  excl pos0   [1,32):     {q(cum(lpT, 1, 32))}")
print(f"  anchor's own view of same windows:")
print(f"  P_A [0,32):             {q(cum(lpA, 0, 32))}")
print(f"  P_A [1,32):             {q(cum(lpA, 1, 32))}")
print(f"  pos0 alone (log10):     {q(lpT[:, 0] / LOG10)}   <- the <think>-vs-'To' gap")
print(f"  budget floor log10(1/80) = {np.log10(BUDGET):.2f}")

print("\n=== 2. Retention under the FLOORED target ===")
print(f"{'beta':>5} {'med log10 Q*[0,32)':>20} {'med log10 Q*[1,32)':>20} {'med lift vs vanilla [1,32) (log10)':>36}")
for b in BETAS:
    lq = d[f"lq_{b}"]
    lift = cum(lq, 1, 32) - cum(lpT, 1, 32)
    print(f"{b:>5} {np.median(cum(lq, 0, 32)):>20.2f} {np.median(cum(lq, 1, 32)):>20.2f} {np.median(lift):>36.2f}")

print("\n=== 3. Share of anchor windows kept above the budget floor (entry prob > 1/80) ===")
print("   (literal-sequence lower bound; branch-level mass is higher)")
print(f"{'':>10} {'[0,32) window':>16} {'[1,32) within-mode':>20}")
print(f"{'vanilla':>10} {np.mean(cum(lpT,0,32) > np.log10(BUDGET)):>16.1%} {np.mean(cum(lpT,1,32) > np.log10(BUDGET)):>20.1%}")
for b in BETAS:
    lq = d[f"lq_{b}"]
    print(f"{'q* b='+str(b):>10} {np.mean(cum(lq,0,32) > np.log10(BUDGET)):>16.1%} {np.mean(cum(lq,1,32) > np.log10(BUDGET)):>20.1%}")

print("\n=== 4. k_bind on real entry windows (taken-token binding counts, beta=0.4) ===")
bd = d["bind_0.4"]
for lo, hi, lab in [(0, 32, "front [0,32)"), (1, 32, "front excl pos0 [1,32)"), (32, 64, "back [32,64)")]:
    k = bd[:, lo:hi].sum(1)
    print(f"  {lab:<24} median {np.median(k):.0f}   p25 {np.percentile(k,25):.0f}  p75 {np.percentile(k,75):.0f}  p90 {np.percentile(k,90):.0f}")
k1 = bd[:, 1:32].sum(1)
lift1 = (d["lq_0.4"][:, 1:32].sum(1) - lpT[:, 1:32].sum(1))
pred = k1 * np.log(0.4)
print(f"  beta^k_bind check [1,32): median actual lift {np.median(lift1):.2f} nats vs beta^k prediction {np.median(pred):.2f} nats")
print(f"  binding rate at pos0: {bd[:, 0].mean():.1%}")

print("\n=== 5. Where does q* differ from the arithmetic mixture? KL(q*||mix), beta=0.4 ===")
km = d["klmix_0.4"]
print(f"  pos 0:        mean {km[:,0].mean():.4f}  median {np.median(km[:,0]):.4f}")
print(f"  pos [1,64):   mean {km[:,1:].mean():.4f}  median {np.median(km[:,1:]):.4f}  p90 {np.percentile(km[:,1:],90):.4f}")

print("\n=== 6. Binding position profile along the window (beta=0.4, mean binding rate) ===")
for lo, hi in [(0,1),(1,2),(2,4),(4,8),(8,16),(16,32),(32,64)]:
    print(f"  pos [{lo:>2},{hi:>2}): {bd[:, lo:hi].mean():.1%}")
