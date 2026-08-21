"""Pick a stratified sample of problems for the discovery pilot.

Stratified rather than hand-picked: the point of this pilot is to learn which
problems admit several approaches, and that question cannot be answered from a
set where the answer was assumed in advance.
"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import or_common

SEED = 20260820
N = 20
# Problems already measured in earlier pilots; kept so the new run is comparable.
CARRY = [3, 115, 180, 35, 68]


def main():
    probs = or_common.load_problems()
    by = defaultdict(list)
    for p in probs:
        by[(p.get("level"), p.get("type"))].append(p)

    import random
    rng = random.Random(SEED)
    chosen = {i: next(p for p in probs if p["id"] == i) for i in CARRY}

    # fill the rest by walking level x type cells, least-represented first
    cells = sorted(by, key=lambda c: (str(c[0]), str(c[1])))
    rng.shuffle(cells)
    lvl_count = defaultdict(int)
    typ_count = defaultdict(int)
    for p in chosen.values():
        lvl_count[p.get("level")] += 1
        typ_count[p.get("type")] += 1

    while len(chosen) < N:
        cells.sort(key=lambda c: (lvl_count[c[0]], typ_count[c[1]], rng.random()))
        placed = False
        for cell in cells:
            cands = [p for p in by[cell] if p["id"] not in chosen]
            if not cands:
                continue
            p = rng.choice(cands)
            chosen[p["id"]] = p
            lvl_count[p.get("level")] += 1
            typ_count[p.get("type")] += 1
            placed = True
            break
        if not placed:
            break

    ids = sorted(chosen)
    out = os.path.join(or_common.OUT, "pilot_problems.json")
    os.makedirs(or_common.OUT, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"seed": SEED, "ids": ids, "carried_over": CARRY}, f, indent=2)

    print(f"{len(ids)} problems (seed {SEED}), {len(CARRY)} carried over\n")
    print(f"{'id':>4}  {'level':<9} {'type':<24} {'answer':<16} problem")
    for i in ids:
        p = chosen[i]
        mark = "*" if i in CARRY else " "
        print(f"{i:>4}{mark} {str(p.get('level')):<9} {str(p.get('type')):<24} "
              f"{p['answer'][:14]:<16} {p['problem'][:64].replace(chr(10),' ')}")
    print("\nlevel:", dict(sorted(lvl_count.items(), key=lambda x: str(x[0]))))
    print("type :", dict(sorted(typ_count.items(), key=lambda x: str(x[0]))))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
