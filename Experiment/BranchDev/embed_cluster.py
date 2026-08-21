"""Embed discovery continuations, cluster them per problem, and draw the result.

Runs in the ICLR conda env (separate from sglang so installs cannot break the
pre-analysis pipeline).

Two things are drawn on the same 2-D projection for every problem: the clusters,
and the source that produced each point. If those two colourings coincide, the
clusters are tracking writing style rather than mathematical approach, and the
branch space would be meaningless. That comparison is the point of the figure.
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(ROOT, "figures")

# Boilerplate openers that every sample from a given source carries. Left in,
# they make same-source texts look alike for reasons that have nothing to do
# with the mathematics.
BOILERPLATE = [
    r"^\s*we need (to )?answer[^.]*\.",
    r"^\s*we need (to )?(find|solve|compute|determine)[^.]*\.",
    r"need reason step by step[^.]*\.",
    r"but final (answer )?(in |within )?boxed[^.]*\.",
    r"^\s*the (problem|user) (asks|is asking)[^.]*\.",
    r"^\s*we are asked[:,]?",
    r"^\s*okay,? let'?s[^.]*\.",
    r"^\s*here'?s a thinking process[^:]*:",
]


def strip_boilerplate(t):
    s = t or ""
    for pat in BOILERPLATE:
        s = re.sub(pat, " ", s, flags=re.I | re.M)
    return re.sub(r"\s+", " ", s).strip()


def load(path, min_chars=40):
    rows = []
    for line in open(path):
        r = json.loads(line)
        if "error" in r:
            continue
        txt = strip_boilerplate(r.get("reasoning"))
        if len(txt) < min_chars:
            continue
        r["text"] = txt
        rows.append(r)
    return rows


def embed(texts, model_name, batch_size, device):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device=device,
                            model_kwargs={"torch_dtype": "bfloat16"})
    m.max_seq_length = 512
    return np.asarray(m.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_numpy=True))


def cluster_one(X, min_cluster_size):
    """HDBSCAN on cosine distance; -1 stays as ambiguous/noise."""
    import hdbscan
    if len(X) < min_cluster_size * 2:
        return np.zeros(len(X), dtype=int)
    D = (1.0 - X @ X.T).astype(np.float64)
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0
    cl = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=min_cluster_size,
                         min_samples=1, cluster_selection_method="eom")
    return cl.fit_predict(D)


def source_entropy(labels, sources):
    """Per-cluster: how mixed are the sources? 1.0 = perfectly even, 0 = single source."""
    out = {}
    for lab in sorted(set(labels)):
        if lab < 0:
            continue
        ss = [s for s, l in zip(sources, labels) if l == lab]
        c = Counter(ss)
        n = len(ss)
        H = -sum((v / n) * np.log(v / n) for v in c.values())
        Hmax = np.log(len(set(sources))) if len(set(sources)) > 1 else 1.0
        out[lab] = (H / Hmax if Hmax > 0 else 0.0, n, len(c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(OUT, "discovery_pilot_v2.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-cluster-size", type=int, default=4)
    ap.add_argument("--tag", default="qwen3emb8b")
    a = ap.parse_args()

    rows = load(a.input)
    print(f"{len(rows)} texts after boilerplate strip", flush=True)
    texts = [r["text"] for r in rows]

    cache = os.path.join(OUT, f"emb_{a.tag}.npy")
    if os.path.exists(cache):
        X = np.load(cache)
        print(f"loaded cached embeddings {X.shape}")
    else:
        X = embed(texts, a.model, a.batch_size, a.device)
        np.save(cache, X)
        print(f"embedded -> {X.shape}, cached at {cache}")

    for i, r in enumerate(rows):
        r["_i"] = i

    by_p = defaultdict(list)
    for r in rows:
        by_p[r["problem_id"]].append(r)

    results = {}
    print(f"\n{'pid':>4}{'n':>5}{'clusters':>10}{'noise%':>8}{'src-entropy':>13}"
          f"{'purity':>9}   sizes")
    for pid in sorted(by_p):
        rs = by_p[pid]
        idx = [r["_i"] for r in rs]
        Xp = X[idx]
        labels = cluster_one(Xp, a.min_cluster_size)
        srcs = [r["model"] for r in rs]
        ent = source_entropy(labels, srcs)
        noise = 100 * sum(1 for l in labels if l < 0) / len(labels)
        ncl = len([l for l in set(labels) if l >= 0])
        mean_ent = np.mean([v[0] for v in ent.values()]) if ent else 0.0
        # purity: largest source share inside a cluster, averaged (1.0 = style-split)
        pur = []
        for lab in sorted(set(labels)):
            if lab < 0:
                continue
            ss = Counter(s for s, l in zip(srcs, labels) if l == lab)
            pur.append(max(ss.values()) / sum(ss.values()))
        mean_pur = np.mean(pur) if pur else 0.0
        sizes = sorted((v[1] for v in ent.values()), reverse=True)
        print(f"{pid:>4}{len(rs):>5}{ncl:>10}{noise:>8.0f}{mean_ent:>13.3f}"
              f"{mean_pur:>9.2f}   {sizes[:8]}")
        results[pid] = {"labels": labels.tolist(), "idx": idx,
                        "entropy": mean_ent, "purity": mean_pur,
                        "n_clusters": ncl, "noise_pct": noise}

    with open(os.path.join(OUT, f"cluster_{a.tag}.json"), "w") as f:
        json.dump({"model": a.model, "min_cluster_size": a.min_cluster_size,
                   "per_problem": {str(k): {kk: vv for kk, vv in v.items()}
                                   for k, v in results.items()}}, f)
    ents = [v["entropy"] for v in results.values()]
    purs = [v["purity"] for v in results.values()]
    print(f"\nmean source-entropy {np.mean(ents):.3f}  (1.0 = clusters fully mixed across sources)")
    print(f"mean source-purity  {np.mean(purs):.3f}  (1.0 = every cluster is one source -> style split)")
    print(f"mean clusters/problem {np.mean([v['n_clusters'] for v in results.values()]):.1f}, "
          f"noise {np.mean([v['noise_pct'] for v in results.values()]):.0f}%")


if __name__ == "__main__":
    main()
