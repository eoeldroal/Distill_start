"""Test the claim that the embedding space tracks approach rather than source.

With human-grade approach labels available for one problem, the decisive
comparison becomes possible: are two texts that share an APPROACH closer than
two texts that share only a SOURCE? If style dominated the space, same-source
pairs would win regardless of approach.
"""
import json, sys, itertools
import numpy as np
from collections import Counter, defaultdict

X = np.load('outputs/emb_qwen3emb8b_raw.npy')
idx = [json.loads(l) for l in open('outputs/emb_qwen3emb8b_raw_index.jsonl')]
src = [json.loads(l) for l in open('outputs/discovery_pilot_v2.jsonl')]
src = [r for r in src if 'error' not in r and (r.get('reasoning') or '').strip()]

PID = 68
sel = [i for i, m in enumerate(idx) if m['problem_id'] == PID]
rows = [src[i] for i in sel]
order = sorted(range(len(rows)), key=lambda j: (rows[j]['model'], rows[j]['sample_k']))
sel = [sel[j] for j in order]
rows = [rows[j] for j in order]

labels = {}
for f in sys.argv[1:]:
    for rec in json.load(open(f)):
        labels[rec['id']] = rec['label']
print(f"labels loaded: {len(labels)}/{len(rows)}")
print("label distribution:", dict(Counter(labels.values()).most_common()))

Xp = X[sel]
S = Xp @ Xp.T

use = [i for i in range(len(rows)) if labels.get(i) not in (None, 'unclear', 'garbled')]
print(f"\nusable (approach identified): {len(use)}/{len(rows)}")
print("per-source usable:", dict(Counter(rows[i]['model'].split('/')[-1] for i in use)))
print("per-approach:", dict(Counter(labels[i] for i in use)))

cats = defaultdict(list)
for a, b in itertools.combinations(use, 2):
    same_app = labels[a] == labels[b]
    same_src = rows[a]['model'] == rows[b]['model']
    cats[(same_app, same_src)].append(S[a, b])

print(f"\n{'pair type':<38}{'n':>6}{'mean cos':>11}{'median':>10}")
names = {(True, True): 'same approach, same source',
         (True, False): 'same approach, DIFFERENT source',
         (False, True): 'DIFFERENT approach, same source',
         (False, False): 'different approach, different source'}
for k in [(True, True), (True, False), (False, True), (False, False)]:
    v = cats.get(k, [])
    if v:
        print(f"{names[k]:<38}{len(v):>6}{np.mean(v):>11.4f}{np.median(v):>10.4f}")

a = cats.get((True, False), [])   # approach shared only
b = cats.get((False, True), [])   # source shared only
if a and b:
    print(f"\nDECISIVE COMPARISON")
    print(f"  same approach + different source : {np.mean(a):.4f}  (n={len(a)})")
    print(f"  different approach + same source : {np.mean(b):.4f}  (n={len(b)})")
    d = np.mean(a) - np.mean(b)
    print(f"  difference: {d:+.4f}")
    print(f"  -> {'APPROACH wins: the space tracks method, not style' if d > 0 else 'SOURCE wins: style dominates the space'}")
    # permutation test on the difference
    pool = np.array(a + b); n1 = len(a)
    rng = np.random.default_rng(20260820)
    null = []
    for _ in range(20000):
        p = rng.permutation(pool)
        null.append(p[:n1].mean() - p[n1:].mean())
    null = np.array(null)
    pv = (np.sum(np.abs(null) >= abs(d)) + 1) / (len(null) + 1)
    print(f"  permutation test p = {pv:.4f}  (20000 shuffles)")

# nearest neighbour restricted to labelled points
print(f"\nnearest neighbour among labelled points only (n={len(use)}):")
Su = S[np.ix_(use, use)].copy()
np.fill_diagonal(Su, -np.inf)
nn = Su.argmax(axis=1)
same_src = sum(1 for j, k in enumerate(nn) if rows[use[j]]['model'] == rows[use[k]]['model'])
same_app = sum(1 for j, k in enumerate(nn) if labels[use[j]] == labels[use[k]])
print(f"  NN shares source  : {100*same_src/len(use):.1f}%")
print(f"  NN shares approach: {100*same_app/len(use):.1f}%")
napp = len(set(labels[i] for i in use)); nsrc = len(set(rows[i]['model'] for i in use))
print(f"  (chance: source {100/nsrc:.1f}%, approach depends on distribution)")
