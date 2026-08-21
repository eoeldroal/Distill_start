"""Where does the floor actually act, as a function of who is uncertain?

Splits states by teacher entropy x anchor entropy and asks, in each cell, how much
probability the floor rescues and whether the rescued tokens are ones the teacher also
considers plausible (its own top-64) or ones it has effectively ruled out.
"""
import argparse
import os

import numpy as np
import pandas as pd

from WorkPlace.ICLR.Experiment.PreAnalysis.common import OUT

BETA = 0.4
LOW, HIGH = 0.2, 1.0
LABEL = {0: "decisive (H<0.2)", 1: "mixed (0.2-1)", 2: "uncertain (H>1)"}


def load(tag):
    sfx = f".{tag}" if tag else ""
    tk = np.load(os.path.join(OUT, f"topk{sfx}.npz"))
    df = pd.read_parquet(os.path.join(OUT, f"cost_states{sfx}.parquet"))
    d = df[df.beta == BETA].set_index("state_id")
    n = len(tk["state_id"])

    # How much of each model's own top-64 mass sits on tokens the *other* model also
    # ranks in its top-64. Low overlap means the two are proposing different candidates.
    ov, a_share, t_share = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        ti, ai = tk["T_ids"][i], tk["A_ids"][i]
        inter = np.intersect1d(ti, ai, assume_unique=False)
        ov[i] = len(inter)
        a_share[i] = np.exp(tk["A_logp"][i][np.isin(ai, inter)]).sum()
        t_share[i] = np.exp(tk["T_logp"][i][np.isin(ti, inter)]).sum()

    g = pd.DataFrame({
        "state_id": tk["state_id"], "kind": tk["kind"], "pos": tk["pos"],
        "H_T": tk["H_T"], "H_A": tk["H_A"], "agree": tk["argmax_agree"],
        "overlap": ov, "anchor_mass_shared": a_share, "teacher_mass_shared": t_share,
    }).set_index("state_id")
    g = g.join(d[["cost", "n_bind", "mass_bind"]])
    g["bT"] = np.digitize(g.H_T, [LOW, HIGH])
    g["bA"] = np.digitize(g.H_A, [LOW, HIGH])
    return g


def table(g, title):
    print(f"\n{'='*104}\n{title}   (n={len(g)}, beta={BETA})\n{'='*104}")
    print(f"{'teacher':<20}{'anchor':<20}{'n':>6}{'share':>8}{'cost':>9}"
          f"{'floor mass':>12}{'agree':>8}{'top64 ov':>10}{'A mass shared':>15}")
    print("-" * 104)
    for bt in (0, 1, 2):
        for ba in (0, 1, 2):
            s = g[(g.bT == bt) & (g.bA == ba)]
            if len(s) < 5:
                continue
            print(f"{LABEL[bt]:<20}{LABEL[ba]:<20}{len(s):>6}{len(s)/len(g):>7.1%}"
                  f"{s.cost.mean():>9.3f}{s.mass_bind.mean():>12.3f}{s.agree.mean():>8.0%}"
                  f"{s.overlap.mean():>10.1f}{s.anchor_mass_shared.mean():>15.3f}")


def spotlight(g, title):
    print(f"\n--- {title} ---")
    q1 = g[(g.bT == 2)]                       # teacher itself is uncertain: a real fork
    q3 = g[(g.bT == 0) & (g.bA == 2)]         # teacher decisive, anchor spread out
    both = g[(g.bT == 2) & (g.bA == 2)]
    for s, name in [(q1, "teacher uncertain (H_T>1)"),
                    (both, "both uncertain"),
                    (q3, "teacher decisive, anchor uncertain")]:
        if not len(s):
            continue
        print(f"{name:<38} n={len(s):>4} ({len(s)/len(g):>5.1%})  "
              f"cost {s.cost.mean():>6.3f}  floor mass {s.mass_bind.mean():.3f}  "
              f"share of all floor mass {s.mass_bind.sum()/g.mass_bind.sum():>5.1%}  "
              f"anchor mass on shared candidates {s.anchor_mass_shared.mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    g = load(a.tag)
    name = "TEACHER-generated states" if a.tag else "ANCHOR-generated states"
    table(g, name)
    spotlight(g, name)
    g.to_parquet(os.path.join(OUT, f"quadrants{'.'+a.tag if a.tag else ''}.parquet"))
