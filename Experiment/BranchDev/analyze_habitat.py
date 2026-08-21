"""Read the anchor habitat sweep before any labelling.

Labels are what settle the habitat question, but three things can be read from the raw text and
they decide which temperatures are worth labelling at all:

  degradation   the sweep already showed T=1.6 turning into multilingual token soup, so the usable
                ceiling has to be located rather than assumed
  language      Base is multilingually pretrained and, given a chat template with no language
                anchor, sometimes leaves English entirely. That is drift, not method diversity,
                and it has to be counted separately or it inflates every diversity measure
  execution     whether the rollout reaches a boxed answer, and whether it is right. Section 1.3
                assumes the phenomenon lives where anchor mass is 0.05-0.2; a problem the anchor
                cannot execute at all is a different regime and should be visible

Surface diversity is reported too, but only as a foil: the whole point of Cal_E section 4 is that
surface variation is not method variation, so a rise here proves nothing on its own.
"""
import argparse, json, os, re, collections, math

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")


def boxed(t):
    i = t.rfind("\\boxed")
    if i < 0:
        return None
    j = t.find("{", i)
    if j < 0:
        return None
    d, out = 0, []
    for ch in t[j:]:
        if ch == "{":
            d += 1
            if d == 1:
                continue
        elif ch == "}":
            d -= 1
            if d == 0:
                break
        out.append(ch)
    return "".join(out).strip() or None


def norm(a):
    a = str(a).strip()
    for x in ["\\left", "\\right", "\\!", "\\,", "\\ ", " ", "$"]:
        a = a.replace(x, "")
    return a.replace("dfrac", "frac").replace("tfrac", "frac").rstrip(".") \
            .replace("^\\circ", "").replace("^{\\circ}", "")


CJK = re.compile(r"[　-鿿가-힯Ѐ-ӿ؀-ۿ฀-๿]")
LATIN = re.compile(r"[A-Za-z]")


def nonlatin_frac(t):
    """Share of letter-ish characters that are not Latin. Drift and soup both raise this."""
    c = len(CJK.findall(t)); l = len(LATIN.findall(t))
    return c / (c + l) if (c + l) else 0.0


def mathiness(t):
    """Share of characters that belong to math prose. Token soup drives this down."""
    if not t:
        return 0.0
    good = sum(ch.isalnum() or ch in " \n\t.,;:()[]{}$\\^_+-*/=<>|'\"" for ch in t)
    return good / len(t)


def distinct_openings(texts, n=8):
    """How many distinct first-n-word openings. A surface measure, reported as a foil."""
    return len({" ".join(t.split()[:n]) for t in texts})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="base_T3,base_T16")
    a = ap.parse_args()
    rows = []
    for tag in a.tags.split(","):
        p = os.path.join(OUT, f"habitat_{tag}.jsonl")
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p)]
    print(f"{len(rows)} rollouts over "
          f"{len({r['problem_id'] for r in rows})} problems, "
          f"temperatures {sorted({r['temperature'] for r in rows})}\n")

    for r in rows:
        b = boxed(r["text"])
        r["_boxed"] = b is not None
        r["_ok"] = b is not None and norm(b) == norm(r["answer"])
        r["_nl"] = nonlatin_frac(r["text"])
        r["_math"] = mathiness(r["text"])

    print("=" * 84)
    print("1. USABILITY BY TEMPERATURE")
    print("=" * 84)
    print(f"{'T':>5}{'n':>6}{'med tok':>9}{'stopped':>9}{'boxed':>8}{'correct':>9}"
          f"{'non-Latin>0.3':>15}{'mathiness':>11}")
    for T in sorted({r["temperature"] for r in rows}):
        s = [r for r in rows if r["temperature"] == T]
        med = sorted(x["n_tokens"] for x in s)[len(s) // 2]
        print(f"{T:>5}{len(s):>6}{med:>9}"
              f"{sum(x['finish']=='stop' for x in s)/len(s):>8.0%}"
              f"{sum(x['_boxed'] for x in s)/len(s):>8.0%}"
              f"{sum(x['_ok'] for x in s)/len(s):>9.1%}"
              f"{sum(x['_nl']>0.3 for x in s)/len(s):>15.0%}"
              f"{sum(x['_math'] for x in s)/len(s):>11.3f}")

    print("\n" + "=" * 84)
    print("2. PER PROBLEM: execution vs temperature  (correct rate)")
    print("=" * 84)
    Ts = sorted({r["temperature"] for r in rows})
    print(f"{'prob':>5}{'lvl':>5}" + "".join(f"{f'T={t}':>9}" for t in Ts) + f"{'drift@1.0':>11}")
    for pid in sorted({r["problem_id"] for r in rows}):
        s = [r for r in rows if r["problem_id"] == pid]
        line = f"{pid:>5}{(s[0].get('level') or '?').replace('Level ','L'):>5}"
        for t in Ts:
            v = [x for x in s if x["temperature"] == t]
            line += f"{(sum(x['_ok'] for x in v)/len(v) if v else float('nan')):>9.2f}"
        v10 = [x for x in s if x["temperature"] == 1.0]
        line += f"{(sum(x['_nl']>0.3 for x in v10)/len(v10) if v10 else 0):>11.0%}"
        print(line)

    print("\n" + "=" * 84)
    print("3. SURFACE DIVERSITY (a foil, not the habitat measure)")
    print("=" * 84)
    print(f"{'T':>5}{'distinct 8-word openings / n':>32}")
    for T in Ts:
        vals = []
        for pid in sorted({r["problem_id"] for r in rows}):
            s = [r["text"] for r in rows if r["problem_id"] == pid and r["temperature"] == T]
            if s:
                vals.append(distinct_openings(s) / len(s))
        print(f"{T:>5}{sum(vals)/len(vals):>32.2f}")

    print("\nNOTE: the habitat verdict needs the blind method labels. What this table settles is"
          "\nwhich temperatures are worth labelling, and how much of any apparent diversity is"
          "\nlanguage drift rather than a different way of solving the problem.")


if __name__ == "__main__":
    main()
