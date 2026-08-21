"""Turn Hendrycks MATH into the parquet verl trains on.

verl ships examples/data_preprocess/math_dataset.py, but it cannot be used here for two
reasons, and both of them are about keeping the training runs comparable to what was already
measured.

First, the instruction string. Cost(beta) was measured under PreAnalysis/common.py's
INSTRUCTION, and the branch-development scripts use the same one. verl's example appends a
different sentence, so a run built on it would train under a prompt no measurement covers.
This script imports INSTRUCTION rather than restating it, so the two cannot drift apart.

Second, the split. The pre-analysis drew its 200 problems from the MATH *train* split, and the
branch panel will be drawn the same way. Training on those problems and then measuring E and V
on them is contamination, so they are held out here and written to heldout.jsonl for the panel
to build on.

The prompt is stored as a chat message, which is the format verl's dataset expects. Rendering
it as plain text (which the Base anchor needs; a chat template opening costs 9.97 nats against
0.277) is a rollout-time concern, set by
data.apply_chat_template_kwargs.chat_template="{{ messages[0]['content'] }}".

data_source routes the reward: "DigitalLearningGmbH/MATH-lighteval" reaches math_reward, which
scores 0 or 1 by comparing the last \\boxed{} against the ground truth.
"""
import argparse
import json
import os
import random

import datasets

from WorkPlace.Distill_start.Experiment.PreAnalysis.common import INSTRUCTION, extract_boxed

ROOT = os.path.dirname(os.path.abspath(__file__))
PREANALYSIS_PROBLEMS = os.path.join(
    os.path.dirname(os.path.dirname(ROOT)), "Experiment", "PreAnalysis", "outputs", "problems.jsonl"
)

SOURCE = "EleutherAI/hendrycks_math"
DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"   # reward_score/__init__.py -> math_reward
SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def load_split(split):
    """Every subject of one split, with the boxed answer pulled out of the solution."""
    rows, unparseable = [], 0
    for subject in SUBJECTS:
        for r in datasets.load_dataset(SOURCE, subject, split=split):
            answer = extract_boxed(r["solution"])
            if answer is None:
                unparseable += 1
                continue
            rows.append(
                {
                    "problem": r["problem"],
                    "answer": answer,
                    "solution": r["solution"],
                    "level": r["level"],
                    "type": r["type"],
                }
            )
    print(f"{split}: {len(rows)} problems ({unparseable} dropped for an unparseable answer)")
    return rows


def held_out_problems():
    """Problem texts the pre-analysis already measured on, matched by exact text."""
    if not os.path.exists(PREANALYSIS_PROBLEMS):
        raise SystemExit(f"missing {PREANALYSIS_PROBLEMS}; run PreAnalysis/prepare_problems.py first")
    with open(PREANALYSIS_PROBLEMS) as f:
        return {json.loads(line)["problem"] for line in f}


def to_verl(rows, split):
    out = []
    for idx, r in enumerate(rows):
        out.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": r["problem"] + "\n\n" + INSTRUCTION}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": r["answer"]},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "level": r["level"],
                    "type": r["type"],
                    "answer": r["answer"],
                    "question": r["problem"],
                },
            }
        )
    return datasets.Dataset.from_list(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--val-n", type=int, default=500,
                    help="validation problems kept; the full test split is too slow to score every eval")
    ap.add_argument("--seed", type=int, default=20260819, help="matches PreAnalysis SEED")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    heldout_texts = held_out_problems()
    train_rows = load_split("train")
    kept, heldout_rows = [], []
    for r in train_rows:
        (heldout_rows if r["problem"] in heldout_texts else kept).append(r)
    print(f"held out {len(heldout_rows)} of {len(heldout_texts)} pre-analysis problems; train keeps {len(kept)}")
    if len(heldout_rows) != len(heldout_texts):
        print(f"  note: {len(heldout_texts) - len(heldout_rows)} pre-analysis problems were not "
              f"found in the train pool (their answers may be the ones dropped above)")

    test_rows = load_split("test")
    rng = random.Random(a.seed)
    rng.shuffle(test_rows)
    val_rows = test_rows[: a.val_n]

    to_verl(kept, "train").to_parquet(os.path.join(a.out, "train.parquet"))
    to_verl(val_rows, "test").to_parquet(os.path.join(a.out, "test.parquet"))
    with open(os.path.join(a.out, "heldout.jsonl"), "w") as f:
        for r in heldout_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(kept)} train, {len(val_rows)} val, {len(heldout_rows)} heldout -> {a.out}")
    print(f"instruction: {INSTRUCTION!r}")


if __name__ == "__main__":
    main()
