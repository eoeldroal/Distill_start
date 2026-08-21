"""Is the dataset in the band where distillation has something to move?

Distillation only has value where the teacher succeeds and the student does not. Too easy and
there is nothing to learn; too hard and the teacher's distribution is not a useful target either.
So the quantity that decides a dataset is not its absolute difficulty but the gap between the two
models, measured under the prompt format training will actually use.

That format is plain text, not the Qwen3 chat template, because the anchor is a Base model
(Cal_Beta_Before_train.md; a chat-template opening costs 9.97 nats against 0.277). Plain text also
takes the Post models off their native format, which is exactly the condition the runs will face,
so measuring them any other way would flatter the setup.

The 1.7B Post model is here as a floor on what the student side can reach. Base alone would fail
almost everything and could not tell "the dataset is too hard" apart from "Base cannot follow the
task at all".

Scoring is reported under two verifiers because they disagree: prime_math accepts 14 against a
ground truth of 14.0, math_dapo rejects it. Which one the training run uses follows from the
data_source we assign, so both numbers are worth seeing before that choice is fixed.

Generation and scoring are separate phases because they fail differently. Generation costs GPU
hours; scoring is string math that can be rerun for free. And scoring inside the generating
process is actively broken here: verl's prime_math wraps are_equal_under_sympy in a
multiprocessing timeout, and under the spawn start method sglang installs, pickling the wrapped
function fails its identity check and every sympy equivalence silently scores as wrong. A plain
process (fork by default on Linux) has no such problem. So generation writes every completion to
disk and scoring reads it back.

실행:
    python probe_difficulty.py --phase generate --n-problems 300
    python probe_difficulty.py --phase score
"""

import argparse
import json
import os
import time
from collections import defaultdict

from WorkPlace.Distill_start.Experiment.PreAnalysis.common import INSTRUCTION

ROOT = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    ("1.7B-Base", "Qwen/Qwen3-1.7B-Base", 1),
    ("1.7B-Post", "Qwen/Qwen3-1.7B", 1),
    ("14B-Post", "Qwen/Qwen3-14B", 2),
]


def load_cascade(n, seed):
    """nvidia/Nemotron-Cascade-RL-Math: problem, answer, source. No traces, no difficulty label."""
    import random

    from datasets import load_dataset

    ds = load_dataset("nvidia/Nemotron-Cascade-RL-Math", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    rows = [ds[i] for i in idx[:n]]
    print(f"Cascade-RL-Math: {len(ds)} total, sampled {len(rows)}")
    dist = defaultdict(int)
    for r in rows:
        dist[r["source"]] += 1
    print("  source:", dict(sorted(dist.items(), key=lambda kv: -kv[1])))
    return rows


def score(text, gt):
    """(prime_math, math_dapo). They disagree on numeric formatting, so keep both."""
    from verl.utils.reward_score import math_dapo, prime_math

    # 채점자가 이상한 LaTeX 에서 터지는 것을 오답으로 넘긴다. 한 문제 때문에 probe 전체를
    # 잃지 않기 위한 것이고, 터진 비율은 아래 all_fail 에 흡수된다.
    try:
        p = bool(prime_math.compute_score(text, gt)[0])
    except Exception:  # noqa: BLE001
        p = False
    try:
        d = bool(math_dapo.compute_score(text, gt)["acc"])
    except Exception:  # noqa: BLE001
        d = False
    return p, d


def generate(tag, path, tp, prompts, rows, a):
    import sglang as sgl

    print(f"\n{'=' * 78}\n{tag}  ({path}, tp={tp})\n{'=' * 78}", flush=True)
    t0 = time.time()
    llm = sgl.Engine(
        model_path=path,
        tp_size=tp,
        dp_size=a.dp,
        mem_fraction_static=a.mem_fraction,
        max_running_requests=a.max_running,
        random_seed=a.seed,
        log_level="error",
    )
    load_s = time.time() - t0

    # n samples per problem in one flat batch, so sglang schedules them together.
    flat = [p for p in prompts for _ in range(a.n_samples)]
    t0 = time.time()
    outs = llm.generate(
        prompt=flat,
        sampling_params={
            "temperature": a.temp,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": a.max_new,
        },
    )
    gen_s = time.time() - t0
    llm.shutdown()

    out_path = os.path.join(a.out, f"probe_gen_{tag}.jsonl")
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            chunk = outs[i * a.n_samples : (i + 1) * a.n_samples]
            f.write(
                json.dumps(
                    {
                        "problem": r["problem"],
                        "answer": str(r["answer"]),
                        "source": r["source"],
                        "completions": [o["text"] for o in chunk],
                        "meta": [
                            {
                                "completion_tokens": o.get("meta_info", {}).get(
                                    "completion_tokens"
                                ),
                                "finish_reason": o.get("meta_info", {})
                                .get("finish_reason", {})
                                .get("type"),
                            }
                            for o in chunk
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"  load {load_s:.1f}s  gen {gen_s:.1f}s  -> {out_path}", flush=True)


def score_phase(a):
    """Read every saved completion back and score it. Free to rerun."""
    import glob

    paths = sorted(glob.glob(os.path.join(a.out, "probe_gen_*.jsonl")))
    if not paths:
        raise SystemExit(f"no probe_gen_*.jsonl in {a.out}; run --phase generate first")

    results, k = [], None
    for path in paths:
        tag = os.path.basename(path)[len("probe_gen_") : -len(".jsonl")]
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        per_problem, lens, trunc, n = [], [], 0, 0
        for r in rows:
            hp, hd = [], []
            for text, meta in zip(r["completions"], r["meta"], strict=True):
                p, d = score(text, r["answer"])
                hp.append(p)
                hd.append(d)
                lens.append(meta["completion_tokens"] or 0)
                trunc += meta["finish_reason"] == "length"
                n += 1
            per_problem.append((r["source"], hp, hd))
        k = len(per_problem[0][1])
        by_source = {}
        for src in {s2 for s2, _, _ in per_problem}:
            sel = [h for s2, h, _ in per_problem if s2 == src]
            by_source[src] = round(sum(sum(h) for h in sel) / (len(sel) * k), 4)
        results.append(
            {
                "model": tag,
                "problems": len(per_problem),
                "pass@1_prime": sum(sum(h) for _, h, _ in per_problem) / n,
                "pass@1_dapo": sum(sum(d) for _, _, d in per_problem) / n,
                f"pass@{k}_prime": sum(any(h) for _, h, _ in per_problem)
                / len(per_problem),
                "all_fail_prime": sum(not any(h) for _, h, _ in per_problem)
                / len(per_problem),
                "all_pass_prime": sum(all(h) for _, h, _ in per_problem)
                / len(per_problem),
                "resp_len_mean": sum(lens) / len(lens),
                "trunc_ratio": trunc / n,
                "by_source_pass@1_prime": by_source,
            }
        )

    hdr = [
        "model",
        "problems",
        "pass@1_prime",
        "pass@1_dapo",
        f"pass@{k}_prime",
        "all_fail_prime",
        "resp_len_mean",
        "trunc_ratio",
    ]
    print(f"\n{'=' * 100}\nCascade-RL-Math, 평문 프롬프트\n{'=' * 100}")
    print("  " + "".join(f"{h:>16}" for h in hdr))
    for r in results:
        cells = "".join(
            f"{r[h]:>16.4f}" if isinstance(r[h], float) else f"{r[h]:>16}"
            for h in hdr[1:]
        )
        print("  " + f"{r['model']:>16}" + cells)
    print("\n  소스별 pass@1 (prime_math)")
    for r in results:
        print(f"    {r['model']:<12} {r['by_source_pass@1_prime']}")
    with open(os.path.join(a.out, "probe_summary.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}/probe_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["generate", "score"], default="generate")
    ap.add_argument("--n-problems", type=int, default=300)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--temp", type=float, default=1.0, help="rollout 설정과 같게 둔다")
    ap.add_argument("--tp", type=int, default=None, help="MODELS 의 tp 를 덮어쓴다")
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--mem-fraction", type=float, default=0.80)
    ap.add_argument("--max-running", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--only", default=None, help="쉼표로 태그 일부만")
    # 같은 모델을 형식/예산만 바꿔 여러 번 돌릴 때 산출 파일이 서로를 덮어쓰지 않게 한다.
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--prompt-format", choices=["plain", "chat"], default="plain")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.phase == "score":
        score_phase(a)
        return

    rows = load_cascade(a.n_problems, a.seed)
    prompts = [r["problem"] + "\n\n" + INSTRUCTION for r in rows]
    print(f"\n프롬프트 형식 (평문, chat template 없음):\n---\n{prompts[0][:400]}\n---")

    picks = (
        MODELS if a.only is None else [m for m in MODELS if m[0] in a.only.split(",")]
    )
    for tag, path, tp in picks:
        generate(tag + a.tag_suffix, path, a.tp or tp, prompts, rows, a)
    print(f"\n생성 끝. 채점: python {os.path.basename(__file__)} --phase score")


if __name__ == "__main__":
    main()
