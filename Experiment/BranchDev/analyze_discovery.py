"""Read the discovery pilot and answer the three questions it was run for.

  1. do the five sources land in the same space, or does style split them?
  2. which problems actually admit several approaches?
  3. how many samples per source are enough?

Approaches are detected with lightweight lexical signatures rather than real
clustering. That is deliberate: clustering settings are branch-dev's own
decision, and this pass only needs a first-order read on whether the material
is workable.
"""
import json, os, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import or_common

# Generic method markers: they name a mathematical move, not a problem.
MARKERS = {
    "lcm_gcd":      r"\blcm\b|least common multiple|\bgcd\b|greatest common divisor",
    "prime_fact":   r"prime factor|factoriz|=\s*\d+\s*[\*×·]\s*\d+\^?\d?",
    "modular":      r"\bmod\b|congruen|remainder when|divisib",
    "enumerate":    r"list (all|the)|enumerat|case \d|try \w+ ?=|check each",
    "symmetry":     r"by symmetry|symmetr|wlog|without loss of generality",
    "coordinate":   r"coordinate|slope|equation of (the )?line|\(x, ?y\)|place .* at (the )?origin",
    "trig":         r"\bcos\b|\bsin\b|\btan\b|theta|θ|trigonometr",
    "calculus":     r"derivative|differentiat|lagrange|critical point|\bd/dx\b",
    "inequality":   r"cauchy|schwarz|\bam-?gm\b|\bqm\b|triangle inequality|inequal",
    "algebraic":    r"substitut|let\s+[a-z]\s*=|rearrang|expand|square (both|the)",
    "geometric":    r"similar triangle|pythagor|area of|congruent|circle|midpoint|median",
    "combinatoric": r"\bbinom|\bchoose\b|combination|permutation|factorial|\bn!\b",
    "series":       r"geometric (series|sum)|arithmetic (series|sequence)|telescop",
    "numeric":      r"≈|approx|estimate|plug in|numerically",
}


def sig(text):
    t = (text or "").lower()
    return frozenset(k for k, p in MARKERS.items() if re.search(p, t, re.I))


def gini_simpson(counter):
    """Probability two random draws differ. 0 = all identical, ->1 = spread out."""
    n = sum(counter.values())
    if n < 2:
        return 0.0
    return 1.0 - sum(c * (c - 1) for c in counter.values()) / (n * (n - 1))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(or_common.OUT, "discovery_pilot.jsonl")
    rows = [json.loads(l) for l in open(path)]
    ok = [r for r in rows if "error" not in r and (r.get("reasoning") or "").strip()]
    print(f"{len(ok)} usable of {len(rows)} rows\n")

    models = sorted({r["model"] for r in ok})
    by_pm = defaultdict(list)
    for r in ok:
        by_pm[(r["problem_id"], r["model"])].append(r)
    by_p = defaultdict(list)
    for r in ok:
        by_p[r["problem_id"]].append(r)

    # ---- 2. which problems admit several approaches ----
    print("=" * 108)
    print("PER-PROBLEM DIVERSITY  (pooled over all sources; div = Gini-Simpson on approach signatures)")
    print("=" * 108)
    print(f"{'id':>4} {'level':<9}{'type':<24}{'n':>4}{'sigs':>6}{'top%':>6}{'div':>7}   per-source distinct sigs")
    prof = []
    for pid in sorted(by_p):
        rs = by_p[pid]
        c = Counter(sig(r["reasoning"]) for r in rs)
        top = 100 * c.most_common(1)[0][1] // len(rs)
        div = gini_simpson(c)
        per = " ".join(f"{m.split('/')[-1][:8]}:{len(Counter(sig(r['reasoning']) for r in by_pm[(pid,m)]))}"
                       for m in models if by_pm[(pid, m)])
        r0 = rs[0]
        print(f"{pid:>4} {str(r0['level']):<9}{str(r0['type'])[:22]:<24}{len(rs):>4}"
              f"{len(c):>6}{top:>6}{div:>7.3f}   {per}")
        prof.append((div, pid, r0["level"], r0["type"], len(c), top))

    print("\n  most diverse:", ", ".join(f"{p[1]}({p[0]:.2f})" for p in sorted(prof, reverse=True)[:5]))
    print("  least diverse:", ", ".join(f"{p[1]}({p[0]:.2f})" for p in sorted(prof)[:5]))

    # ---- 1. do sources share approaches, or does style split them? ----
    print("\n" + "=" * 108)
    print("SOURCE MIXING  (per problem: of the approach signatures seen, how many are used by >1 source?)")
    print("=" * 108)
    shared_tot = solo_tot = 0
    rowsout = []
    for pid in sorted(by_p):
        sig2models = defaultdict(set)
        for r in by_p[pid]:
            sig2models[sig(r["reasoning"])].add(r["model"])
        shared = sum(1 for s, ms in sig2models.items() if len(ms) > 1)
        solo = sum(1 for s, ms in sig2models.items() if len(ms) == 1)
        shared_tot += shared; solo_tot += solo
        # mass-weighted: what share of samples sits on a shared signature
        n_shared = sum(1 for r in by_p[pid] if len(sig2models[sig(r["reasoning"])]) > 1)
        rowsout.append((pid, shared, solo, 100 * n_shared // len(by_p[pid])))
    print(f"{'id':>4}{'shared sigs':>13}{'solo sigs':>11}{'% samples on shared sig':>26}")
    for pid, sh, so, pct in rowsout:
        print(f"{pid:>4}{sh:>13}{so:>11}{pct:>26}")
    tot_pct = 100 * sum(r[3] for r in rowsout) // len(rowsout)
    print(f"\n  overall: {shared_tot} shared vs {solo_tot} solo signatures; "
          f"{tot_pct}% of samples sit on a signature that more than one source produced")
    print("  (high share = sources land in the same space; low = style is splitting them)")

    # ---- 3. sample-count sensitivity ----
    print("\n" + "=" * 108)
    print("SAMPLE SENSITIVITY  (distinct signatures per problem, pooled, as sample budget shrinks)")
    print("=" * 108)
    for take in (4, 8, 12, 16):
        tot = 0
        for pid in sorted(by_p):
            s = set()
            for m in models:
                for r in sorted(by_pm[(pid, m)], key=lambda x: x["sample_k"])[:take]:
                    s.add(sig(r["reasoning"]))
            tot += len(s)
        print(f"  {take:>2} samples/source: {tot/len(by_p):>5.2f} distinct signatures per problem "
              f"(total calls {take*len(models)*len(by_p)})")

    # ---- per-source character ----
    print("\n" + "=" * 108)
    print("PER-SOURCE")
    print("=" * 108)
    print(f"{'model':<34}{'n':>5}{'sigs/prob':>11}{'chars p50':>11}{'rt p50':>8}{'lat p50':>9}{'$':>9}")
    for m in models:
        rs = [r for r in ok if r["model"] == m]
        spp = sum(len(Counter(sig(r["reasoning"]) for r in by_pm[(pid, m)]))
                  for pid in by_p if by_pm[(pid, m)]) / len(by_p)
        ch = sorted(len(r["reasoning"]) for r in rs)
        rt = sorted((r.get("reasoning_tokens") or 0) for r in rs)
        la = sorted(r["latency_s"] for r in rs)
        print(f"{m:<34}{len(rs):>5}{spp:>11.2f}{ch[len(ch)//2]:>11}{rt[len(rt)//2]:>8}"
              f"{la[len(la)//2]:>9.1f}{sum(r.get('cost') or 0 for r in rs):>9.4f}")


if __name__ == "__main__":
    main()
