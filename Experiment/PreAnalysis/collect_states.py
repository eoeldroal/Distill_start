"""Step 2: roll out the anchor and sample the states Cost(beta) will be measured on.

Generation uses HF transformers (not an inference server) so the exact output token
ids are recorded: a state must be an exact token sequence for both models to be
conditioned on the same thing.

Sampling is full softmax -- no top-p, no top-k, no repetition penalty -- matching the
canonical-sampler decision. Temperature here is a probe setting for collecting states;
Cost(beta) itself is defined per state and does not depend on it.

Outputs: outputs/rollouts.jsonl (prompt_ids, output_ids, text), outputs/states.jsonl
"""
import argparse
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from WorkPlace.ICLR.Experiment.PreAnalysis.common import ANCHOR, OUT, SEED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=os.path.join(OUT, "problems.jsonl"))
    ap.add_argument("--rollouts-per-problem", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-states", type=int, default=4000)
    ap.add_argument("--state-stride", type=int, default=16)
    ap.add_argument("--early-frac", type=float, default=0.15)
    ap.add_argument("--max-state-pos", type=int, default=512)
    ap.add_argument("--limit-problems", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gen-model", default=ANCHOR,
                    help="model that generates the states (default: the anchor, matching "
                         "on-policy distillation at t=0; pass the teacher to bracket t=inf)")
    args = ap.parse_args()

    suffix = f".{args.tag}" if args.tag else ""
    problems = [json.loads(l) for l in open(args.problems)]
    if args.limit_problems:
        problems = problems[: args.limit_problems]

    print(f"generating states with {args.gen_model}")
    tok = AutoTokenizer.from_pretrained(args.gen_model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    # The base model was never trained on the chat format, so let it stop on either
    # the chat turn marker or its own eos.
    stop_ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i >= 0:
            stop_ids.add(i)
    stop_ids = sorted(stop_ids)
    print(f"stop token ids: {stop_ids}")

    jobs = [p for p in problems for _ in range(args.rollouts_per_problem)]
    torch.manual_seed(SEED)

    rollouts = []
    for start in range(0, len(jobs), args.batch_size):
        chunk = jobs[start : start + args.batch_size]
        enc = tok([c["prompt"] for c in chunk], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                top_p=1.0,
                top_k=0,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=stop_ids,
                pad_token_id=tok.pad_token_id,
            )
        plen = enc["input_ids"].shape[1]
        for j, c in enumerate(chunk):
            prompt_ids = enc["input_ids"][j][enc["attention_mask"][j].bool()].tolist()
            gen = out[j][plen:].tolist()
            for k, t in enumerate(gen):
                if t in stop_ids:
                    gen = gen[: k + 1]
                    break
            rollouts.append(
                {
                    "rollout_id": len(rollouts),
                    "problem_id": c["id"],
                    "prompt_ids": prompt_ids,
                    "output_ids": gen,
                    "text": tok.decode(gen, skip_special_tokens=False),
                }
            )
        done = min(start + args.batch_size, len(jobs))
        print(f"  generated {done}/{len(jobs)}", flush=True)

    path_r = os.path.join(OUT, f"rollouts{suffix}.jsonl")
    with open(path_r, "w") as f:
        for r in rollouts:
            f.write(json.dumps(r) + "\n")

    lens = sorted(len(r["output_ids"]) for r in rollouts)
    print(f"wrote {len(rollouts)} rollouts -> {path_r}")
    print(f"output length: median {lens[len(lens)//2]}, min {lens[0]}, max {lens[-1]}")

    # Opening states: one per problem (all rollouts of a problem share it, and it is
    # the model-level exploration unit). Internal states: strided positions inside
    # each rollout, subsampled to the state budget.
    states, seen_opening = [], set()
    for r in rollouts:
        if r["problem_id"] not in seen_opening:
            seen_opening.add(r["problem_id"])
            states.append({"rollout_id": r["rollout_id"], "pos": 0, "kind": "opening"})
    # Early positions are sampled densely: the teacher's thinking-mode opening is a
    # deterministic format ritual, so cost varies sharply over the first few tokens.
    early_pos = [1, 2, 3, 4, 6, 8, 12]
    early = [
        {"rollout_id": r["rollout_id"], "pos": p, "kind": "early"}
        for r in rollouts
        for p in early_pos
        if len(r["output_ids"]) >= p
    ]
    internal = [
        {"rollout_id": r["rollout_id"], "pos": p, "kind": "internal"}
        for r in rollouts
        for p in range(args.state_stride, min(len(r["output_ids"]), args.max_state_pos) + 1, args.state_stride)
    ]
    rng = random.Random(SEED)
    rng.shuffle(early)
    rng.shuffle(internal)
    n_early = min(len(early), int(args.n_states * args.early_frac))
    states.extend(early[:n_early])
    budget = max(0, args.n_states - len(states))
    states.extend(internal[:budget])
    for i, s in enumerate(states):
        s["state_id"] = i

    path_s = os.path.join(OUT, f"states{suffix}.jsonl")
    with open(path_s, "w") as f:
        for s in states:
            f.write(json.dumps(s) + "\n")
    counts = {}
    for s in states:
        counts[s["kind"]] = counts.get(s["kind"], 0) + 1
    print(f"wrote {len(states)} states {counts} -> {path_s}")


if __name__ == "__main__":
    main()
