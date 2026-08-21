"""Step 1: sample MATH train problems and wrap them in the shared prompt.

Output: outputs/problems.jsonl  (id, problem, answer, level, type, prompt)
"""
import argparse
import json
import os
import random

from datasets import load_dataset
from transformers import AutoTokenizer

from WorkPlace.Distill_start.Experiment.PreAnalysis.common import ANCHOR, OUT, SEED, TEACHER, build_prompt, extract_boxed

SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(OUT, "problems.jsonl"))
    args = ap.parse_args()

    rows = []
    for subject in SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        for r in ds:
            answer = extract_boxed(r["solution"])
            if answer is None:
                continue
            rows.append(
                {
                    "problem": r["problem"],
                    "answer": answer,
                    "level": r["level"],
                    "type": r["type"],
                }
            )
    print(f"MATH train pool: {len(rows)} problems with a parseable answer")

    # Stratify over subject x level so the state sample is not dominated by one
    # corner of the distribution.
    strata = {}
    for r in rows:
        strata.setdefault((r["type"], r["level"]), []).append(r)
    keys = sorted(strata)
    rng = random.Random(SEED)
    for k in keys:
        rng.shuffle(strata[k])

    picked, i = [], 0
    while len(picked) < args.n:
        progressed = False
        for k in keys:
            if len(picked) >= args.n:
                break
            if i < len(strata[k]):
                picked.append(strata[k][i])
                progressed = True
        if not progressed:
            break
        i += 1
    rng.shuffle(picked)

    # The teacher tokenizer defines the prompt; assert the anchor tokenizes it
    # identically, otherwise the two models are not conditioned on the same state.
    tok_t = AutoTokenizer.from_pretrained(TEACHER)
    tok_a = AutoTokenizer.from_pretrained(ANCHOR)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    mismatch = 0
    with open(args.out, "w") as f:
        for idx, r in enumerate(picked):
            prompt = build_prompt(tok_t, r["problem"])
            if tok_t(prompt)["input_ids"] != tok_a(prompt)["input_ids"]:
                mismatch += 1
            f.write(json.dumps({"id": idx, "prompt": prompt, **r}) + "\n")

    dist = {}
    for r in picked:
        dist[r["level"]] = dist.get(r["level"], 0) + 1
    print(f"wrote {len(picked)} problems -> {args.out}")
    print(f"level distribution: {dict(sorted(dist.items()))}")
    if mismatch:
        raise SystemExit(f"FATAL: {mismatch} prompts tokenize differently in the two models")
    print("tokenizer check: teacher and anchor produce identical token ids for every prompt")


if __name__ == "__main__":
    main()
