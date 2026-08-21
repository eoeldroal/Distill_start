"""Render the discovery embedding space as publication figures.

One question: when continuations land close together, is it because they share a
mathematical approach or because the same model wrote them? Every panel is drawn
twice from identical coordinates, once coloured by source and once by approach.
Whichever colouring separates is the axis the space is organised along.

Method choice. The claim being made is about distances and class separability,
which is exactly the claim t-SNE and UMAP cannot support: they preserve local
neighbourhoods only, and inter-cluster distances in those projections carry no
meaning (Wattenberg et al. 2016; Jeon et al. 2025 recommend global techniques
such as PCA or MDS for distance-based analyses). PCA is linear, deterministic,
and parameter-free, so it cannot manufacture separation that is not in the data
and cannot be tuned toward a prettier answer.

PCA is computed inside each problem. Across problems the leading variance is the
subject matter, which would bury both axes of interest.

Every panel reports the variance captured by the two drawn axes, and every
figure is accompanied by neighbourhood-preservation numbers (trustworthiness and
continuity) computed against the full 4,096-dimensional space, so the reader can
tell how much of the original geometry survives the projection. The statistical
claim itself is made in the full space, never from the picture.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(ROOT, "figures")

SHORT = {
    "deepseek/deepseek-v4-flash-0731": "DeepSeek V4 Flash",
    "z-ai/glm-5.2": "GLM 5.2",
    "qwen/qwen3.8-27b": "Qwen3.8-27B",
    "minimax/minimax-m3": "MiniMax M3",
    "xiaomi/mimo-v2.5-pro": "MiMo v2.5 Pro",
    "meta/muse-glimmer-30b": "Muse Glimmer",
}
# Okabe-Ito, the de facto standard colour-blind-safe categorical palette
# (Okabe & Ito 2002; Wong, Nature Methods 2011). Black is reserved for text.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]
# Colour alone must never carry meaning: each series also gets a marker shape.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
GREY = "#BFBFBF"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,          # print resolution
    "pdf.fonttype": 42,          # embed TrueType, required by most venues
    "ps.fonttype": 42,
})


def pca2(X):
    """Two leading principal components by SVD, plus per-axis variance share.

    Deterministic: SVD has no random initialisation, so no seed is involved and
    repeated runs return identical coordinates up to axis sign.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T, (S ** 2) / (S ** 2).sum()


def rank_matrix(D):
    """Rank of every point as a neighbour of every other point (self excluded)."""
    n = len(D)
    order = np.argsort(D, axis=1)
    R = np.empty((n, n), dtype=int)
    rows = np.arange(n)[:, None]
    R[rows, order] = np.arange(n)[None, :]
    return R


def trust_cont(Xhi, Ylo, k):
    """Trustworthiness and continuity (Venna & Kaski 2001).

    Trustworthiness penalises points pulled into a neighbourhood by the
    projection that do not belong there in the original space; continuity
    penalises true neighbours pushed out. Both lie in [0, 1], higher is better.
    Together they say how much of the 4,096-dimensional neighbourhood structure
    the two drawn axes actually carry.
    """
    n = len(Xhi)
    if n <= k + 1:
        return float("nan"), float("nan")

    def pdist(A):
        G = A @ A.T
        sq = np.diag(G)
        D = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0.0)
        np.fill_diagonal(D, np.inf)
        return np.sqrt(D)

    Dh, Dl = pdist(Xhi), pdist(Ylo)
    Rh, Rl = rank_matrix(Dh), rank_matrix(Dl)
    nn_h = np.argsort(Dh, axis=1)[:, :k]
    nn_l = np.argsort(Dl, axis=1)[:, :k]
    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    t_pen = sum(Rh[i, j] - k for i in range(n) for j in nn_l[i] if Rh[i, j] >= k)
    c_pen = sum(Rl[i, j] - k for i in range(n) for j in nn_h[i] if Rl[i, j] >= k)
    return 1 - norm * t_pen, 1 - norm * c_pen


def load():
    X = np.load(os.path.join(OUT, "emb_qwen3emb8b_raw.npy"))
    idx = [json.loads(l) for l in open(os.path.join(OUT, "emb_qwen3emb8b_raw_index.jsonl"))]
    lab, meta = {}, {}
    for line in open(os.path.join(OUT, "master_traj.jsonl")):
        d = json.loads(line)
        lab[(d["problem_id"], d["source_model"], d["sample_k"])] = d.get("approach")
        meta[d["problem_id"]] = (d.get("level"), d.get("type"))
    approach = [lab.get((m["problem_id"], m["model"], m["sample_k"])) for m in idx]
    assert len(X) == len(idx) == len(approach), (len(X), len(idx), len(approach))
    return X, idx, approach, meta


def draw(ax, Y, keys, order, title, foot, legend_title):
    """Scatter with one colour AND one marker per key; unlabelled points grey."""
    style = {k: (OKABE_ITO[i % len(OKABE_ITO)], MARKERS[i % len(MARKERS)])
             for i, k in enumerate(order)}
    dead = [j for j, k in enumerate(keys) if k in (None, "unclear", "garbled")]
    if dead:
        ax.scatter(Y[dead, 0], Y[dead, 1], s=22, c=GREY, marker="o", alpha=.5,
                   linewidths=0, zorder=1, label=f"unclear/garbled ({len(dead)})")
    for k in order:
        sel = [j for j, kk in enumerate(keys) if kk == k]
        if not sel:
            continue
        c, m = style[k]
        ax.scatter(Y[sel, 0], Y[sel, 1], s=30, c=c, marker=m, alpha=.9,
                   linewidths=.4, edgecolors="white", zorder=2,
                   label=f"{k} ({len(sel)})")
    ax.set_title(title, pad=6)
    ax.set_xlabel(foot, fontsize=7.6, color="#555")
    # Axes are principal components: unitless, sign-arbitrary, and equally
    # scaled so that a given distance means the same in both directions.
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for s in ax.spines.values():
        s.set_color("#DDD")
    ax.legend(fontsize=6.8, loc="best", frameon=True, framealpha=.92,
              borderpad=.35, labelspacing=.25, handletextpad=.3,
              title=legend_title, title_fontsize=7)


def per_problem(X, idx, approach, meta, pids, out, k=15):
    n = len(pids)
    fig, axes = plt.subplots(n, 2, figsize=(10.5, 4.7 * n))
    axes = np.atleast_2d(axes)
    report = []
    for r, pid in enumerate(pids):
        sel = [i for i, m in enumerate(idx) if m["problem_id"] == pid]
        Y, var = pca2(X[sel])
        tw, ct = trust_cont(X[sel], Y, k)
        report.append((pid, len(sel), var[0], var[1], tw, ct))
        srcs = [SHORT.get(idx[i]["model"], idx[i]["model"]) for i in sel]
        apps = [approach[i] for i in sel]
        foot = (f"PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}% of variance   |   "
                f"trustworthiness {tw:.3f}, continuity {ct:.3f} (k={k})   |   n={len(sel)}")
        lvl, typ = meta.get(pid, ("", ""))
        draw(axes[r, 0], Y, srcs, sorted(set(srcs)),
             f"Problem {pid} ({lvl}, {typ}): by source model", foot, "source")
        live = [a for a in apps if a not in (None, "unclear", "garbled")]
        order = [a for a, _ in sorted(((a, live.count(a)) for a in set(live)),
                                      key=lambda t: -t[1])]
        draw(axes[r, 1], Y, apps, order,
             f"Problem {pid}: by solution approach", foot, "approach")
    fig.suptitle(
        "Discovery continuations in Qwen3-Embedding-8B space, projected by PCA within each problem\n"
        "Identical coordinates in each row; only the colouring differs",
        fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, .985])
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
    return report


def overview(X, idx, approach, out, k=15):
    pids = sorted({m["problem_id"] for m in idx})
    cols, rows = 5, (len(pids) + 4) // 5
    order = sorted(SHORT.values())
    style = {kk: (OKABE_ITO[i % len(OKABE_ITO)], MARKERS[i % len(MARKERS)])
             for i, kk in enumerate(order)}
    fig, axes = plt.subplots(rows, cols, figsize=(2.95 * cols, 3.05 * rows))
    axes = np.atleast_2d(axes)
    for a in axes.flat:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#DDD")
    report = []
    for j, pid in enumerate(pids):
        ax = axes[j // cols, j % cols]
        sel = [i for i, m in enumerate(idx) if m["problem_id"] == pid]
        Y, var = pca2(X[sel])
        tw, ct = trust_cont(X[sel], Y, k)
        live = {approach[i] for i in sel
                if approach[i] not in (None, "unclear", "garbled")}
        report.append((pid, len(sel), var[0], var[1], tw, ct, len(live)))
        srcs = [SHORT.get(idx[i]["model"], idx[i]["model"]) for i in sel]
        for kk in order:
            s2 = [t for t, v in enumerate(srcs) if v == kk]
            if s2:
                c, m = style[kk]
                ax.scatter(Y[s2, 0], Y[s2, 1], s=12, c=c, marker=m,
                           alpha=.9, linewidths=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"p{pid}: {len(live)} approach{'es' if len(live)!=1 else ''}",
                     fontsize=8.2, pad=3)
        ax.set_xlabel(f"PC1+2 {(var[0]+var[1])*100:.0f}%  T {tw:.2f}  C {ct:.2f}",
                      fontsize=6.3, color="#666", labelpad=2)
    for j in range(len(pids), rows * cols):
        axes[j // cols, j % cols].axis("off")
    handles = [plt.Line2D([], [], marker=style[kk][1], ls="", ms=6,
                          color=style[kk][0], label=kk) for kk in order]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.5,
               frameon=False, bbox_to_anchor=(.5, -.008))
    fig.suptitle(
        "All 20 problems, coloured by source model (PCA within each problem)\n"
        "T = trustworthiness, C = continuity against the full 4,096-d space (k=15)",
        fontsize=11.5, y=1.0)
    fig.tight_layout(rect=[0, .015, 1, .985])
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="")
    ap.add_argument("--k", type=int, default=15, help="neighbourhood size for T/C")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    X, idx, approach, meta = load()
    print(f"{len(X)} vectors, {X.shape[1]} dims, "
          f"{len({m['problem_id'] for m in idx})} problems, k={a.k}")

    ov = overview(X, idx, approach, os.path.join(FIG, "emb_overview.png"), a.k)
    pids = ([int(s) for s in a.problems.split(",")] if a.problems else [180, 155, 127])
    per_problem(X, idx, approach, meta, pids, os.path.join(FIG, "emb_detail.png"), a.k)

    # Projection quality table, written next to the figures so the numbers in
    # the captions can be checked against their source.
    path = os.path.join(FIG, "projection_quality.md")
    with open(path, "w") as f:
        f.write("# Projection quality per problem\n\n")
        f.write("PCA within each problem, cosine-normalised Qwen3-Embedding-8B vectors.\n")
        f.write(f"Trustworthiness and continuity computed against the full 4,096-d "
                f"space with k={a.k} (Venna & Kaski 2001), higher is better.\n\n")
        f.write("| problem | n | PC1 | PC2 | PC1+PC2 | trustworthiness | continuity | approaches |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for pid, n, v1, v2, tw, ct, na in ov:
            f.write(f"| {pid} | {n} | {v1*100:.1f}% | {v2*100:.1f}% | "
                    f"{(v1+v2)*100:.1f}% | {tw:.3f} | {ct:.3f} | {na} |\n")
        arr = np.array([[v1 + v2, tw, ct] for _, _, v1, v2, tw, ct, _ in ov])
        f.write(f"| median | | | | {np.median(arr[:,0])*100:.1f}% | "
                f"{np.median(arr[:,1]):.3f} | {np.median(arr[:,2]):.3f} | |\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
