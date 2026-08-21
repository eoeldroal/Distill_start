"""Embed the discovery texts as they are, and report nothing but facts about them.

No boilerplate stripping, no length filter, no truncation setting, no clustering:
every one of those is a knob, and the point of this pass is to look at the data
before any knob has been turned.
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(OUT, "discovery_pilot_v2.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tag", default="qwen3emb8b_raw")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.input)]
    rows = [r for r in rows if "error" not in r and (r.get("reasoning") or "").strip()]
    texts = [r["reasoning"] for r in rows]
    print(f"{len(rows)} texts, verbatim (no preprocessing)")

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(a.model, device=a.device,
                            model_kwargs={"torch_dtype": "bfloat16"})
    print(f"model max_seq_length (default): {m.max_seq_length}")

    tok = m.tokenizer
    lens = sorted(len(tok.encode(t)) for t in texts)
    n = len(lens)
    print(f"token lengths: min {lens[0]}  p50 {lens[n//2]}  p90 {lens[int(n*.9)]}  max {lens[-1]}")
    over = sum(1 for L in lens if L > m.max_seq_length)
    print(f"texts exceeding the model default: {over}/{n}"
          f"  -> {'NOTHING IS TRUNCATED' if over == 0 else 'SOME WOULD BE TRUNCATED'}")

    X = m.encode(texts, batch_size=a.batch_size, normalize_embeddings=True,
                 show_progress_bar=True, convert_to_numpy=True)
    X = np.asarray(X, dtype=np.float32)
    np.save(os.path.join(OUT, f"emb_{a.tag}.npy"), X)
    with open(os.path.join(OUT, f"emb_{a.tag}_index.jsonl"), "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"i": i, "problem_id": r["problem_id"], "model": r["model"],
                                "sample_k": r["sample_k"], "level": r.get("level"),
                                "type": r.get("type")}, ensure_ascii=False) + "\n")
    print(f"\nembeddings {X.shape} -> outputs/emb_{a.tag}.npy")


if __name__ == "__main__":
    main()
