"""Pool the labelled problems and test approach-vs-source across all of them.

One problem gave a direction but not significance. Pooling several problems
raises the pair counts by an order of magnitude, and it also guards against the
finding being a quirk of one problem type: if the effect is real it should hold
per problem, not only in aggregate.
"""
import glob, itertools, json, os
import numpy as np
from collections import Counter, defaultdict

X = np.load('outputs/emb_qwen3emb8b_raw.npy')
idx = [json.loads(l) for l in open('outputs/emb_qwen3emb8b_raw_index.jsonl')]
src = [json.loads(l) for l in open('outputs/discovery_pilot_v2.jsonl')]
src = [r for r in src if 'error' not in r and (r.get('reasoning') or '').strip()]

files = sorted(glob.glob('outputs/labels_json/p*.json'))
print(f"labelled problems: {[os.path.basename(f) for f in files]}\n")

allpairs = defaultdict(list)
per_problem = []

for f in files:
    pid = int(os.path.basename(f)[1:-5])
    labels = {r['id']: r['label'] for r in json.load(open(f))}
    sel = [i for i, m in enumerate(idx) if m['problem_id'] == pid]
    rows = [src[i] for i in sel]
    order = sorted(range(len(rows)), key=lambda j: (rows[j]['model'], rows[j]['sample_k']))
    sel = [sel[j] for j in order]; rows = [rows[j] for j in order]
    Xp = X[sel]; S = Xp @ Xp.T
    use = [i for i in range(len(rows)) if labels.get(i) not in (None, 'unclear', 'garbled')]
    if len(use) < 10:
        print(f"p{pid}: only {len(use)} labelled, skipped"); continue

    cats = defaultdict(list)
    for a, b in itertools.combinations(use, 2):
        k = (labels[a] == labels[b], rows[a]['model'] == rows[b]['model'])
        cats[k].append(S[a, b]); allpairs[k].append(S[a, b])
    app_only = cats.get((True, False), []); src_only = cats.get((False, True), [])
    napp = len(set(labels[i] for i in use))
    d = (np.mean(app_only) - np.mean(src_only)) if (app_only and src_only) else float('nan')
    per_problem.append((pid, len(use), napp, np.mean(app_only) if app_only else np.nan,
                        np.mean(src_only) if src_only else np.nan, d,
                        len(app_only), len(src_only)))
    print(f"p{pid}: {len(use)}/{len(rows)} labelled, {napp} distinct approaches "
          f"{dict(Counter(labels[i] for i in use).most_common())}")

print("\n" + "="*104)
print("PER-PROBLEM: is 'same approach, different source' closer than 'different approach, same source'?")
print("="*104)
print(f"{'pid':>5}{'n':>5}{'appr':>6}{'app-only cos':>15}{'src-only cos':>15}{'diff':>10}"
      f"{'n_app':>8}{'n_src':>8}  verdict")
for pid, n, na, a, b, d, npa, nps in per_problem:
    v = "approach" if d > 0 else "SOURCE"
    print(f"{pid:>5}{n:>5}{na:>6}{a:>15.4f}{b:>15.4f}{d:>+10.4f}{npa:>8}{nps:>8}  {v}")

print("\n" + "="*104)
print("POOLED ACROSS ALL LABELLED PROBLEMS")
print("="*104)
names = {(True, True): 'same approach, same source',
         (True, False): 'same approach, DIFFERENT source',
         (False, True): 'DIFFERENT approach, same source',
         (False, False): 'different approach, different source'}
for k in [(True, True), (True, False), (False, True), (False, False)]:
    v = allpairs.get(k, [])
    if v:
        print(f"  {names[k]:<40}{len(v):>7}  mean {np.mean(v):.4f}   median {np.median(v):.4f}")

a = allpairs[(True, False)]; b = allpairs[(False, True)]
d = np.mean(a) - np.mean(b)
print(f"\n  approach-only {np.mean(a):.4f} (n={len(a)})  vs  source-only {np.mean(b):.4f} (n={len(b)})")
print(f"  difference {d:+.4f}")
pool = np.array(a + b); n1 = len(a)
rng = np.random.default_rng(20260820)
null = np.array([(lambda p: p[:n1].mean() - p[n1:].mean())(rng.permutation(pool))
                 for _ in range(20000)])
pv = (np.sum(np.abs(null) >= abs(d)) + 1) / (len(null) + 1)
print(f"  permutation test p = {pv:.5f}  (20000 shuffles)")
print(f"  -> {'APPROACH dominates' if d > 0 else 'SOURCE dominates'}"
      f"{' (significant)' if pv < 0.05 else ' (not significant)'}")
