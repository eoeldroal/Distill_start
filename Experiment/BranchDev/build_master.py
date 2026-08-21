"""Join everything about each trajectory into one record, so later analysis
never has to recompute or re-align anything.

One row per trajectory: which problem, which source wrote it, the blind
approach label where one exists, the raw log-probability under each checkpoint
scored so far, and the head-position log-probs. Adding another checkpoint means
adding one column, not redoing the join.
"""
import glob, json, os

OUT = "outputs"


def load_prefill(tag):
    d = {}
    for l in open(f"{OUT}/prefill_{tag}.jsonl"):
        r = json.loads(l)
        d[(r["problem_id"], r["source_model"], r["sample_k"])] = r
    return d


def main():
    gen = [json.loads(l) for l in open(f"{OUT}/discovery_pilot_v2.jsonl")]
    gen = [r for r in gen if "error" not in r and (r.get("reasoning") or "").strip()]

    # blind approach labels, keyed the way the labelling chunks were built:
    # index within (problem, sorted by model then sample_k)
    labels = {}
    for f in sorted(glob.glob(f"{OUT}/labels_json/p*.json")):
        pid = int(os.path.basename(f)[1:-5])
        rs = sorted([r for r in gen if r["problem_id"] == pid],
                    key=lambda r: (r["model"], r["sample_k"]))
        for rec in json.load(open(f)):
            r = rs[rec["id"]]
            labels[(pid, r["model"], r["sample_k"])] = rec["label"]

    checkpoints = {}
    for path in sorted(glob.glob(f"{OUT}/prefill_*.jsonl")):
        tag = os.path.basename(path)[8:-6]
        checkpoints[tag] = load_prefill(tag)
    print(f"checkpoints found: {list(checkpoints)}")

    out = []
    for r in gen:
        k = (r["problem_id"], r["model"], r["sample_k"])
        rec = {
            "problem_id": r["problem_id"],
            "level": r.get("level"), "type": r.get("type"),
            "answer": r.get("answer"),
            "source_model": r["model"],
            "sample_k": r["sample_k"],
            "approach": labels.get(k),
            "finish_reason": r.get("finish_reason"),
            "n_chars": len(r["reasoning"]),
            "reasoning": r["reasoning"],
            "content": r.get("content"),
            "logp": {}, "logp_tokens": {}, "token_ids": None, "n_tokens": None,
        }
        for tag, d in checkpoints.items():
            if k in d:
                rec["logp"][tag] = d[k]["logp_total"]
                rec["logp_tokens"][tag] = d[k]["logp_tokens"]
                rec["token_ids"] = d[k]["token_ids"]
                rec["n_tokens"] = d[k]["n_tokens"]
        out.append(rec)

    path = f"{OUT}/master_traj.jsonl"
    with open(path, "w") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    nlab = sum(1 for r in out if r["approach"])
    print(f"{len(out)} trajectories -> {path}")
    print(f"  with approach label: {nlab}")
    print(f"  with logp for all {len(checkpoints)} checkpoints: "
          f"{sum(1 for r in out if len(r['logp']) == len(checkpoints))}")
    print(f"  size: {os.path.getsize(path)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
