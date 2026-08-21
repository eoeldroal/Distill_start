"""Quantify what temperature actually buys at one state.

Three things are separated:
  1. surface variation  - do samples differ as strings at all (distinct openings)
  2. approach variation - do they differ in the mathematical move they make
  3. text integrity     - is the text still well-formed at high temperature

(2) is what discovery needs. (1) without (2) is worthless, and (3) is the price.
"""
import json
import re
import sys
from collections import Counter, defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/pilot_tsweep_p3.jsonl"
rows = [json.loads(l) for l in open(path)]
ok = [r for r in rows if "error" not in r and (r.get("reasoning") or "").strip()]
by_t = defaultdict(list)
for r in ok:
    by_t[r["temperature"]].append(r)

# Approach signature for problem 3 (coupon / lcm problem): which mathematical
# object does the sample reach for? These are detected, not prescribed -- a
# sample can hit several or none.
SIGS = {
    "lcm": r"\blcm\b|least common multiple",
    "prime_factor": r"prime factor|3\s*\*\s*5\^?2|2\s*\*\s*3\s*\*\s*5",
    "gcd": r"\bgcd\b|greatest common divisor",
    "floor_div": r"floor\(|⌊|integer part|divide .* by 150",
    "list_multiples": r"multiples of|150, ?300|list them",
    "inclusion_exclusion": r"inclusion|exclusion|venn",
    "modular": r"\bmod\b|congruen|remainder",
    "enumerate": r"count how many|enumerat|step through",
}

def sig(text):
    t = text.lower()
    return frozenset(k for k, p in SIGS.items() if re.search(p, t))

def opening(text, n=60):
    return re.sub(r"\s+", " ", text.strip())[:n]

# text integrity: fraction of non-ascii junk, repeated-token runs, broken latex
def junk_ratio(t):
    if not t:
        return 1.0
    bad = sum(1 for c in t if ord(c) > 0x2500 or (ord(c) < 32 and c not in "\n\t\r"))
    return bad / len(t)

def longest_repeat_run(t):
    toks = t.split()
    best = cur = 1
    for i in range(1, len(toks)):
        cur = cur + 1 if toks[i] == toks[i-1] else 1
        best = max(best, cur)
    return best

print(f"{'T':>5} {'n':>4} {'uniq open':>10} {'uniq sig':>9} {'top signature (share)':<38} "
      f"{'junk%':>7} {'maxrep':>7} {'chars p50':>10}")
print("-"*100)
for t in sorted(by_t):
    rs = by_t[t]
    texts = [r["reasoning"] for r in rs]
    opens = Counter(opening(x) for x in texts)
    sigs = Counter(sig(x) for x in texts)
    top_sig, top_n = sigs.most_common(1)[0]
    lens = sorted(len(x) for x in texts)
    junk = 100 * sum(junk_ratio(x) for x in texts) / len(texts)
    rep = max(longest_repeat_run(x) for x in texts)
    label = ",".join(sorted(top_sig)) or "(none)"
    print(f"{t:>5} {len(rs):>4} {len(opens):>10} {len(sigs):>9} "
          f"{label[:30]+' '+str(round(100*top_n/len(rs)))+'%':<38} "
          f"{junk:>7.3f} {rep:>7} {lens[len(lens)//2]:>10}")

print("\n=== signature distribution per temperature ===")
for t in sorted(by_t):
    rs = by_t[t]
    sigs = Counter(sig(r["reasoning"]) for r in rs)
    print(f"\nT={t} ({len(rs)} samples)")
    for s, c in sigs.most_common(6):
        print(f"   {c:>3}x  {','.join(sorted(s)) or '(none)'}")

print("\n=== sample openings at each temperature ===")
for t in sorted(by_t):
    print(f"\n--- T={t} ---")
    for r in by_t[t][:4]:
        print(f"   {opening(r['reasoning'], 130)}")
