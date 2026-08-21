"""Compare two checkpoints on the same fixed trajectories.

The quantity that matters is the difference, log P(post) - log P(base), on a
trajectory both models scored. Whatever makes a trajectory intrinsically hard or
unusual cancels in that difference, which is exactly why prefill scoring is
usable even though the absolute numbers are dominated by style.
"""
import json, sys
import numpy as np
from collections import defaultdict


def load(tag):
    d = {}
    for l in open(f'outputs/prefill_{tag}.jsonl'):
        r = json.loads(l)
        d[(r['problem_id'], r['source_model'], r['sample_k'])] = r
    return d


base = load(sys.argv[1] if len(sys.argv) > 1 else 'base')
post = load(sys.argv[2] if len(sys.argv) > 2 else 'qwen3_1p7b')
keys = sorted(set(base) & set(post))
print(f"{len(keys)} trajectories scored by both\n")

delta = np.array([post[k]['logp_total'] - base[k]['logp_total'] for k in keys])
ntok = np.array([base[k]['n_tokens'] for k in keys])
per = delta / ntok

print("=== log P(Qwen3-1.7B) - log P(Base), per trajectory ===")
print(f"  total    p10 {np.percentile(delta,10):+8.1f}  p50 {np.percentile(delta,50):+8.1f}"
      f"  p90 {np.percentile(delta,90):+8.1f}")
print(f"  per-token p10 {np.percentile(per,10):+7.3f}  p50 {np.percentile(per,50):+7.3f}"
      f"  p90 {np.percentile(per,90):+7.3f}")
print(f"  fraction where post-training made the trajectory MORE likely: "
      f"{100*np.mean(delta>0):.1f}%")

print("\n=== by source: did post-training move toward compressed styles? ===")
print(f"{'source':<26}{'n':>5}{'Base/tok':>11}{'Post/tok':>11}{'delta/tok':>12}{'tokens':>9}")
by = defaultdict(list)
for k in keys:
    by[k[1]].append(k)
rows = []
for m, ks in by.items():
    b = np.median([base[k]['logp_total'] / base[k]['n_tokens'] for k in ks])
    p = np.median([post[k]['logp_total'] / post[k]['n_tokens'] for k in ks])
    t = np.median([base[k]['n_tokens'] for k in ks])
    rows.append((p - b, m, len(ks), b, p, t))
for d, m, n, b, p, t in sorted(rows, reverse=True):
    print(f"{m.split('/')[-1]:<26}{n:>5}{b:>11.3f}{p:>11.3f}{d:>+12.3f}{t:>9.0f}")

print("\n=== the first tokens: does post-training learn the <think> opening? ===")
hb = np.array([base[k]['logp_head'][:6] for k in keys if len(base[k]['logp_head']) >= 6])
hp = np.array([post[k]['logp_head'][:6] for k in keys if len(post[k]['logp_head']) >= 6])
print(f"{'pos':>5}{'Base':>10}{'Post':>10}{'delta':>10}")
for i in range(6):
    print(f"{i:>5}{np.median(hb[:,i]):>10.3f}{np.median(hp[:,i]):>10.3f}"
          f"{np.median(hp[:,i])-np.median(hb[:,i]):>+10.3f}")

print("\n=== spread: does post-training compress or widen the range across trajectories? ===")
bl = np.array([base[k]['logp_total'] / base[k]['n_tokens'] for k in keys])
pl = np.array([post[k]['logp_total'] / post[k]['n_tokens'] for k in keys])
print(f"  Base per-token: p10 {np.percentile(bl,10):.3f}  p90 {np.percentile(bl,90):.3f}"
      f"  spread {np.percentile(bl,90)-np.percentile(bl,10):.3f}")
print(f"  Post per-token: p10 {np.percentile(pl,10):.3f}  p90 {np.percentile(pl,90):.3f}"
      f"  spread {np.percentile(pl,90)-np.percentile(pl,10):.3f}")
