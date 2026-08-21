"""Re-embed the same discovery texts under a different embedder or a different instruction.

Companion to diag_style_axis.py, which judges whatever this produces by the same four tests, so
that a variant is compared against the original on one number rather than on impressions.

Two levers, and only two.

  --instruct    Qwen3-Embedding is instruction aware. Its card documents the exact wrapper
                `Instruct: {task}\nQuery:{text}` and reports that instructions move retrieval
                accuracy by 1 to 5 percent. Note what that claim is and is not: it is about
                accuracy on a retrieval task, not about which axis the similarity runs along.
                The card also expects the instruction on the query side of an asymmetric pair,
                whereas every text here plays the same role, so the instruction goes on all of
                them symmetrically. That is off-label use and the result has to be read as such.

  --model       any locally cached embedder. Nothing is downloaded; a model that is not already
                in the cache fails rather than fetching, which is deliberate.

Both texts and labels are the ones already on disk, so the only thing that changes between runs
is the representation. Nothing here filters, strips, or truncates.
"""
import argparse, json, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")

# Wrapper quoted from the Qwen3-Embedding-8B model card.
QWEN_WRAP = "Instruct: {task}\nQuery:{text}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(OUT, "discovery_pilot_v2.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--instruct", default="",
                    help="task sentence; empty means embed the bare text as the original run did")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.input)]
    rows = [r for r in rows if "error" not in r and (r.get("reasoning") or "").strip()]
    raw = [r["reasoning"] for r in rows]
    if a.instruct:
        texts = [QWEN_WRAP.format(task=a.instruct, text=t) for t in raw]
        print(f"instruction applied to all {len(texts)} texts (symmetric, off-label):")
        print(f"  {a.instruct!r}")
        print(f"  first input begins: {texts[0][:110]!r}")
    else:
        texts = raw
        print(f"no instruction; {len(texts)} bare texts, as in the original run")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")     # fail rather than download
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(a.model, device=a.device,
                            model_kwargs={"torch_dtype": "bfloat16"},
                            trust_remote_code=True)
    print(f"model {a.model}  max_seq_length {m.max_seq_length}")

    tok = m.tokenizer
    lens = sorted(len(tok.encode(t)) for t in texts)
    n = len(lens)
    print(f"token lengths: min {lens[0]}  p50 {lens[n//2]}  p90 {lens[int(n*.9)]}  max {lens[-1]}")
    over = sum(1 for L in lens if L > m.max_seq_length)
    print(f"texts over the model default: {over}/{n}"
          f"  -> {'nothing truncated' if over == 0 else 'SOME TRUNCATED'}")

    X = m.encode(texts, batch_size=a.batch_size, normalize_embeddings=True,
                 show_progress_bar=True, convert_to_numpy=True)
    X = np.asarray(X, dtype=np.float32)
    np.save(os.path.join(OUT, f"emb_{a.tag}.npy"), X)
    with open(os.path.join(OUT, f"emb_{a.tag}_index.jsonl"), "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"i": i, "problem_id": r["problem_id"], "model": r["model"],
                                "sample_k": r["sample_k"], "level": r.get("level"),
                                "type": r.get("type")}, ensure_ascii=False) + "\n")
    # keep the run's own settings beside its output so a tag stays self-describing
    with open(os.path.join(OUT, f"emb_{a.tag}_config.json"), "w") as f:
        json.dump({"model": a.model, "instruct": a.instruct, "n": len(texts),
                   "dims": int(X.shape[1]), "max_seq_length": m.max_seq_length,
                   "wrapper": QWEN_WRAP if a.instruct else None}, f, indent=2)
    print(f"\nembeddings {X.shape} -> outputs/emb_{a.tag}.npy")
    print(f"judge it with:  python diag_style_axis.py --tag {a.tag}")


if __name__ == "__main__":
    main()
