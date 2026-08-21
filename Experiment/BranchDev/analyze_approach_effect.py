"""Separate the effect of approach from the effect of source.

The trap in comparing sources directly is that each source also favours certain
approaches, so a difference between sources cannot be attributed to either
factor alone. Holding the source fixed and varying only the approach removes
that ambiguity: within one source, style is a constant, so any remaining
difference in how much post-training moved away from a trajectory has to come
from what the trajectory does mathematically.
"""
import json
import numpy as np
from collections import defaultdict

rows = [json.loads(l) for l in open('outputs/master_traj.jsonl')]
rows = [r for r in rows if r.get('approach') and r['approach'] not in ('unclear', 'garbled')
        and len(r['logp']) == 2]
print(f"{len(rows)} labelled trajectories with both checkpoints scored\n")

for r in rows:
    r['delta'] = (r['logp']['qwen3_1p7b'] - r['logp']['base']) / r['n_tokens']

print("="*100)
print("WITHIN-SOURCE: holding the writer fixed, does the approach change the drop?")
print("="*100)
cells = defaultdict(list)
for r in rows:
    cells[(r['problem_id'], r['source_model'], r['approach'])].append(r['delta'])

within = defaultdict(dict)
for (pid, src, app), v in cells.items():
    within[(pid, src)][app] = (np.mean(v), len(v))

spreads = []
print(f"{'problem':>8} {'source':<24} approaches (mean delta/token, n)")
for (pid, src), d in sorted(within.items()):
    if len(d) < 2:
        continue
    tot = sum(n for _, n in d.values())
    if tot < 6:
        continue
    parts = "  ".join(f"{a}:{m:+.3f}({n})" for a, (m, n) in
                      sorted(d.items(), key=lambda x: -x[1][0]))
    sp = max(m for m, _ in d.values()) - min(m for m, _ in d.values())
    spreads.append(sp)
    print(f"{pid:>8} {src.split('/')[-1]:<24} {parts}")
print(f"\nwithin-source spread across approaches: median {np.median(spreads):.3f} nats/token "
      f"(n={len(spreads)} source-problem cells)")

print("\n" + "="*100)
print("WITHIN-APPROACH: holding the method fixed, does the source change the drop?")
print("="*100)
wa = defaultdict(dict)
for (pid, src, app), v in cells.items():
    wa[(pid, app)][src] = (np.mean(v), len(v))
spreads2 = []
print(f"{'problem':>8} {'approach':<18} sources (mean delta/token, n)")
for (pid, app), d in sorted(wa.items()):
    if len(d) < 2:
        continue
    tot = sum(n for _, n in d.values())
    if tot < 6:
        continue
    parts = "  ".join(f"{s.split('/')[-1][:9]}:{m:+.3f}({n})" for s, (m, n) in
                      sorted(d.items(), key=lambda x: -x[1][0]))
    sp = max(m for m, _ in d.values()) - min(m for m, _ in d.values())
    spreads2.append(sp)
    print(f"{pid:>8} {app:<18} {parts}")
print(f"\nwithin-approach spread across sources: median {np.median(spreads2):.3f} nats/token "
      f"(n={len(spreads2)} approach-problem cells)")

print("\n" + "="*100)
print("VERDICT")
print("="*100)
a, b = np.median(spreads), np.median(spreads2)
print(f"  varying the APPROACH while fixing source : {a:.3f} nats/token")
print(f"  varying the SOURCE while fixing approach : {b:.3f} nats/token")
print(f"  -> {'APPROACH matters more' if a > b else 'SOURCE matters more'} "
      f"(ratio {max(a,b)/max(min(a,b),1e-9):.2f}x)")

# two-way ANOVA-style variance decomposition on the same cells
print("\nvariance of cell means attributable to each factor (problem-centred):")
import itertools
bysrc, byapp = defaultdict(list), defaultdict(list)
for (pid, src, app), v in cells.items():
    if len(v) < 2: continue
    m = np.mean(v)
    bysrc[src].append(m); byapp[app].append(m)
print(f"  between-source variance : {np.var([np.mean(v) for v in bysrc.values()]):.5f}")
print(f"  between-approach variance: {np.var([np.mean(v) for v in byapp.values()]):.5f}")
