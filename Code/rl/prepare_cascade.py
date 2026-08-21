"""Turn nvidia/Nemotron-Cascade-RL-Math into the parquet verl trains on.

This is the dataset the runs use. It was chosen by measurement, not by reputation: on 300 of its
problems under the plain prompt the pipeline actually uses, Qwen3-1.7B-Base reaches pass@1 0.069
and Qwen3-14B reaches 0.312, with all-fail groups falling from 0.823 to 0.527. Distillation only
has something to move where the teacher succeeds and the student does not, and that is the gap.
MATH was replaced because it sits above that band for this student.

It also fits the paper's own framing: Draft section 1 cites Nemotron-Cascade 2 for putting OPD
between RL stages, which is the handoff this work studies, so the runs use the RL math data
released with that system.

Only four fields go in, because only four are read.

    data_source     picks the scoring function in verl/utils/reward_score/__init__.py. Not
                    metadata: an unlisted string raises NotImplementedError before step 1. Ours
                    is registered alongside the numina family, which routes to prime_math. That
                    verifier matters here even though the distillation loss never sees a reward,
                    because training/groups/{all_fail,informative} is read off it and is the last
                    link of the mechanism chain the paper reports. On the same generations
                    prime_math scores the teacher at 0.312 where math_dapo scores 0.112, since
                    math_dapo rejects 14 against a ground truth of 14.0 and Cascade carries
                    answers like 999.998976.
    prompt          the problem plus the instruction imported from PreAnalysis/common.py, so the
                    pre-analysis, the difficulty probe and training cannot drift apart. Kept as a
                    chat message; rendering it as plain text is a rollout-time concern that
                    conf/prompt_format/plain.yaml owns, which keeps the format switchable per arm.
    reward_model    ground_truth is Cascade's own short answer. No \\boxed extraction is needed.
    extra_info      source travels with each problem so per-source results stay recoverable at
                    analysis time. The four sources differ in difficulty (14B pass@1 runs 0.15 to
                    0.34 across them), and carrying the label is strictly better than filtering on
                    it: an earlier read that openmathreasoning had no teacher-student gap turned
                    out to rest on 13 problems.

`ability` is omitted. verl never reads it; the example scripts carry it out of habit.

실행: PYTHONPATH=/home/eoeldroal python prepare_cascade.py
"""

import argparse
import os
import random

import datasets
from WorkPlace.Distill_start.Experiment.PreAnalysis.common import INSTRUCTION, SEED

ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE = "nvidia/Nemotron-Cascade-RL-Math"
DATA_SOURCE = "nemotron_cascade_math"  # reward_score/__init__.py -> prime_math


def to_verl(rows, split):
    return datasets.Dataset.from_list(
        [
            {
                "data_source": DATA_SOURCE,
                "prompt": [
                    {"role": "user", "content": r["problem"] + "\n\n" + INSTRUCTION}
                ],
                "reward_model": {"style": "rule", "ground_truth": str(r["answer"])},
                "extra_info": {"split": split, "index": i, "source": r["source"]},
            }
            for i, r in enumerate(rows)
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument(
        "--val-n", type=int, default=500, help="train 에서 떼어 낼 검증 문제 수"
    )
    ap.add_argument("--seed", type=int, default=SEED, help="사전 분석과 같은 seed")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ds = datasets.load_dataset(SOURCE, split="train")
    rows = [dict(r) for r in ds]
    random.Random(a.seed).shuffle(rows)
    val, train = rows[: a.val_n], rows[a.val_n :]

    to_verl(train, "train").to_parquet(os.path.join(a.out, "train.parquet"))
    to_verl(val, "test").to_parquet(os.path.join(a.out, "test.parquet"))

    print(f"{SOURCE}: {len(ds)} problems")
    print(f"  train {len(train)}  val {len(val)}  -> {a.out}")
    print(f"  data_source: {DATA_SOURCE}")
    print(f"  instruction: {INSTRUCTION!r}")
    for name, part in (("train", train), ("val", val)):
        dist = {}
        for r in part:
            dist[r["source"]] = dist.get(r["source"], 0) + 1
        print(f"  {name} source: {dict(sorted(dist.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
