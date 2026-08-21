"""Concrete states from the two cells that matter: a genuine fork (both models
uncertain) and a mismatch (teacher decisive, anchor spread out)."""
import json
import os

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from WorkPlace.Distill_start.Experiment.PreAnalysis.common import ANCHOR, OUT, TEACHER
from WorkPlace.Distill_start.Experiment.PreAnalysis.examples import show

BETA = 0.4

g = pd.read_parquet(os.path.join(OUT, "quadrants.parquet"))
g = g[g.kind != "opening"]                       # exclude the format ritual
rollouts = {r["rollout_id"]: r for r in
            (json.loads(l) for l in open(os.path.join(OUT, "rollouts.jsonl")))}
states = {s["state_id"]: s for s in
          (json.loads(l) for l in open(os.path.join(OUT, "states.jsonl")))}

fork = g[(g.bT == 2) & (g.bA == 2)].sort_values("mass_bind")
mism = g[(g.bT == 0) & (g.bA == 2)].sort_values("cost")
picks = [
    (int(fork.index[len(fork) // 2]), "GENUINE FORK: teacher uncertain AND anchor uncertain"),
    (int(fork.index[int(len(fork) * 0.9)]), "GENUINE FORK: floor delivering most protection"),
    (int(mism.index[len(mism) // 2]), "MISMATCH: teacher decisive, anchor uncertain"),
]

tok = AutoTokenizer.from_pretrained(TEACHER)
teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16, device_map="cuda").eval()
anchor = AutoModelForCausalLM.from_pretrained(ANCHOR, dtype=torch.bfloat16, device_map="cuda").eval()


def lp(model, ids):
    with torch.no_grad():
        o = model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1)
    return torch.log_softmax(o.logits[:, -1, :].float(), -1)


for sid, title in picks:
    st = states[sid]
    r = rollouts[st["rollout_id"]]
    ids = r["prompt_ids"] + r["output_ids"][: st["pos"]]
    row = g.loc[sid]
    show(tok, lp(teacher, ids), lp(anchor, ids),
         f"{title}  [pos {st['pos']}]", tok.decode(ids[-30:]), float(row["cost"]))
    print(f"top-64 overlap: {int(row['overlap'])}/64;  anchor mass on shared candidates: "
          f"{row['anchor_mass_shared']:.3f};  argmax agree: {bool(row['agree'])}")
