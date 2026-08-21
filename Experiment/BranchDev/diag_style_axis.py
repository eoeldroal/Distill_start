"""Diagnostics on the style-dominates-method result before blaming the embedder.

Section 4.2 of Cal_E pooled every within-problem pair across 20 problems and found that pairs
sharing only the source model sit closer than pairs sharing only the solution method. Before
concluding that the embedding space is organised by author, three alternative explanations have to
be ruled out, and all three are checkable on vectors already stored.

  length      texts are capped at 256 tokens and vendors differ in verbosity, so the "source"
              axis could be a length axis wearing a costume.
  pooling     the 20 problems carry between 1 and 6 approaches each. A pooled mean can invert the
              sign of every individual group (Simpson), so the same contrast is recomputed
              per problem and the problems are counted.
  cell size   same-method-different-source has 42,259 pairs against 2,946 for the mirror cell.
              A difference of means across cells that unequal deserves a paired estimator.

The script then tests whether the author direction can simply be projected out, using the source
labels the group already has, and reports what that does to both contrasts.
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
DEAD = (None, "unclear", "garbled")
RNG = np.random.default_rng(20260820)          # same seed as problem selection


def load(tag="qwen3emb8b_raw"):
    X = np.load(os.path.join(OUT, f"emb_{tag}.npy")).astype(np.float64)
    idx = [json.loads(l) for l in open(os.path.join(OUT, f"emb_{tag}_index.jsonl"))]
    rec = {}
    for line in open(os.path.join(OUT, "master_traj.jsonl")):
        d = json.loads(line)
        rec[(d["problem_id"], d["source_model"], d["sample_k"])] = d
    meta = [rec.get((m["problem_id"], m["model"], m["sample_k"])) for m in idx]
    assert all(m is not None for m in meta)
    pid = np.array([m["problem_id"] for m in idx])
    src = np.array([m["model"] for m in idx])
    app = np.array([d["approach"] for d in meta], dtype=object)
    ntok = np.array([d["n_tokens"] for d in meta])
    # Stored normalised, but in float32, so norms sit within about 0.4% of one. Check that the
    # file is what it claims to be, then renormalise exactly so a dot product is a cosine.
    nrm = np.linalg.norm(X, axis=1)
    assert np.allclose(nrm, 1.0, atol=5e-3), (nrm.min(), nrm.max())
    X /= nrm[:, None]
    return X, pid, src, app, ntok


def cells(X, pid, src, app, sel=None):
    """Mean cosine in the four pair cells, computed within problems only."""
    tot = {k: [0.0, 0] for k in ("ss", "sd", "ds", "dd")}
    for p in np.unique(pid):
        m = (pid == p) if sel is None else ((pid == p) & sel)
        i = np.flatnonzero(m)
        if len(i) < 2:
            continue
        S = X[i] @ X[i].T
        a, s = app[i], src[i]
        live = ~np.isin(a, list(DEAD))
        for u in range(len(i)):
            for v in range(u + 1, len(i)):
                if not (live[u] and live[v]):
                    continue
                k = ("s" if a[u] == a[v] else "d") + ("s" if s[u] == s[v] else "d")
                tot[k][0] += S[u, v]
                tot[k][1] += 1
    return {k: (v[0] / v[1] if v[1] else float("nan"), v[1]) for k, v in tot.items()}


def contrast(c):
    """method-only minus source-only. Positive means the space sees method."""
    return c["sd"][0] - c["ds"][0]


def per_problem(X, pid, src, app):
    """Same contrast inside each problem, so pooling cannot manufacture the sign."""
    rows = []
    for p in np.unique(pid):
        c = cells(X, pid, src, app, sel=(pid == p))
        if c["sd"][1] and c["ds"][1]:
            rows.append((int(p), c["sd"][0], c["sd"][1], c["ds"][0], c["ds"][1],
                         contrast(c)))
    return rows


def paired_by_problem(rows):
    """Sign test over problems: a per-problem estimator immune to cell-size imbalance."""
    d = np.array([r[5] for r in rows])
    neg = int((d < 0).sum())
    return d, neg, len(d)


def length_check(X, pid, src, ntok):
    """Is the author axis a length axis? Two questions, both answered per problem.

    First, do sources differ in length at all. Second, does the first principal component,
    the axis that visibly separates the sources in the figures, track length.
    """
    per_src = {s: ntok[src == s] for s in np.unique(src)}
    corrs = []
    for p in np.unique(pid):
        i = np.flatnonzero(pid == p)
        Z = X[i] - X[i].mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        pc1 = Z @ Vt[0]
        L = ntok[i].astype(float)
        if L.std() > 0 and pc1.std() > 0:
            # sign of PC1 is arbitrary, so only |r| is meaningful
            corrs.append(abs(np.corrcoef(pc1, L)[0, 1]))
    return per_src, np.array(corrs)


def regress_pairs(X, pid, src, app, ntok):
    """Least squares on pair cosines, so length gets a column of its own.

    cos(u,v) ~ 1 + same_method + same_source + |len_u - len_v| + mean_len

    Fitted inside each problem and the coefficients averaged over problems. The two indicator
    coefficients answer the question the cell means answer, but with length held fixed, which
    matters because vendors differ in verbosity by more than a factor of two and a pair from one
    vendor is therefore a length-matched pair by construction. No threshold is chosen anywhere,
    which is the reason for preferring this over splitting the pairs into length bins.
    """
    rows = []
    for p in np.unique(pid):
        i = np.flatnonzero(pid == p)
        live = i[~np.isin(app[i], list(DEAD))]
        if len(live) < 12:
            continue
        B, a, s = X[live], app[live], ntok[live].astype(float)
        sm, ss_ = src[live], None
        S = B @ B.T
        y, Z = [], []
        for u in range(len(live)):
            for v in range(u + 1, len(live)):
                y.append(S[u, v])
                Z.append([1.0,
                          1.0 if a[u] == a[v] else 0.0,
                          1.0 if sm[u] == sm[v] else 0.0,
                          abs(s[u] - s[v]),
                          0.5 * (s[u] + s[v])])
        y, Z = np.asarray(y), np.asarray(Z)
        # scale the two length columns so their coefficients are per 100 tokens
        Z[:, 3] /= 100.0
        Z[:, 4] /= 100.0
        # A problem solved one way makes the same-method column almost all ones, which is
        # collinear with the intercept and gives a coefficient that is arbitrarily large rather
        # than informative. Require both indicators to actually vary before trusting the fit.
        frac = Z[:, 1:3].mean(axis=0)
        if min(frac.min(), 1 - frac.max()) < 0.02:
            continue
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        rows.append((int(p), len(y), *beta))
        del ss_
    return rows


def erase_sources(X, pid, src):
    """Project out the subspace spanned by per-source mean offsets, within each problem.

    With k sources the offsets span at most k-1 dimensions, and removing them makes the source
    means coincide, so no linear probe can read the author off the vector any more. This is the
    difference-of-means special case of concept erasure; it is the weakest and most transparent
    version, which is the point, because a stronger erasure would be harder to argue was not
    also deleting the signal.
    """
    Y = X.copy()
    for p in np.unique(pid):
        i = np.flatnonzero(pid == p)
        B = X[i]
        g = B.mean(0, keepdims=True)
        M = np.stack([B[src[i] == s].mean(0) - g[0] for s in np.unique(src[i])])
        # orthonormal basis of the offset span
        U, S, _ = np.linalg.svd(M, full_matrices=False)
        keep = S > 1e-10
        V = (np.linalg.pinv(np.diag(S[keep])) @ U[:, keep].T @ M) if keep.any() else None
        if V is not None and len(V):
            Q, _ = np.linalg.qr(V.T)
            B = B - (B @ Q) @ Q.T
        n = np.linalg.norm(B, axis=1, keepdims=True)
        Y[i] = B / np.maximum(n, 1e-12)
        del g
    return Y


def probe_source(X, pid, src):
    """How well can a nearest-centroid probe name the author? Leave-one-out, within problem."""
    ok = tot = 0
    for p in np.unique(pid):
        i = np.flatnonzero(pid == p)
        B, s = X[i], src[i]
        for j in range(len(i)):
            cen, lab = [], []
            for u in np.unique(s):
                m = (s == u); m[j] = False
                if m.any():
                    cen.append(B[m].mean(0)); lab.append(u)
            C = np.stack(cen)
            C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
            ok += (lab[int(np.argmax(C @ B[j]))] == s[j]); tot += 1
    return ok / tot, tot


def source_subsets(X, pid, src, app, ntok):
    """The contrast on every three-source subset, ordered by how matched their verbosity is.

    A tempting story is that the author effect is really the verbosity effect, since the vendors
    differ by more than a factor of two in median length. If that story held, the subsets whose
    vendors are closest in length would be the ones where method wins. Enumerating all twenty
    subsets tests the story instead of illustrating it with one of them.
    """
    from itertools import combinations
    med = {s: float(np.median(ntok[src == s])) for s in np.unique(src)}
    rows = []
    for tri in combinations(sorted(med), 3):
        spread = max(med[s] for s in tri) - min(med[s] for s in tri)
        c = cells(X, pid, src, app, sel=np.isin(src, list(tri)))
        rows.append((spread, contrast(c), tri))
    return med, sorted(rows)


def bootstrap_texts(X, pid, src, app, draws=200):
    """Resample texts, not pairs, because texts are the independent units.

    The permutation test in Cal_E shuffles pair-level cosines, but a pair is not an observation:
    the 2,946 source-only pairs are built from far fewer distinct texts, and every text appears in
    many pairs. Resampling whole texts inside each problem respects that dependence and gives an
    interval the pair-level shuffle cannot.
    """
    out = []
    for _ in range(draws):
        keep = []
        for p in np.unique(pid):
            i = np.flatnonzero(pid == p)
            keep.append(RNG.choice(i, size=len(i), replace=True))
        m = np.zeros(len(X), bool)
        m[np.unique(np.concatenate(keep))] = True
        out.append(contrast(cells(X, pid, src, app, sel=m)))
    return np.asarray(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen3emb8b_raw",
                    help="embedding tag under outputs/, so a different embedder can be judged "
                         "by exactly the same four tests")
    a = ap.parse_args()
    X, pid, src, app, ntok = load(a.tag)
    nsrc = len(np.unique(src))
    print(f"embedding: {a.tag}")
    print(f"{len(X)} vectors, {X.shape[1]} dims, {len(np.unique(pid))} problems, "
          f"{nsrc} sources\n")

    print("=" * 74)
    print("1. POOLED CELLS, reproducing Cal_E section 4.2")
    print("=" * 74)
    c0 = cells(X, pid, src, app)
    name = {"ss": "same method + same source", "sd": "same method + diff source",
            "ds": "diff method + same source", "dd": "diff method + diff source"}
    for k in ("ss", "sd", "ds", "dd"):
        print(f"  {name[k]:30s} {c0[k][0]:.4f}   n={c0[k][1]:,}")
    print(f"\n  contrast (method-only minus source-only) = {contrast(c0):+.4f}")
    print("  negative means pairs sharing only the author are the closer ones")

    print("\n" + "=" * 74)
    print("2. IS IT POOLING? same contrast inside each problem")
    print("=" * 74)
    rows = per_problem(X, pid, src, app)
    print(f"  {'prob':>5} {'method-only':>12} {'n':>7} {'source-only':>12} {'n':>6} {'contrast':>10}")
    for p, sd, nsd, ds, nds, d in rows:
        print(f"  {p:>5} {sd:>12.4f} {nsd:>7,} {ds:>12.4f} {nds:>6,} {d:>+10.4f}")
    d, neg, n = paired_by_problem(rows)
    print(f"\n  problems where the author wins: {neg}/{n}")
    print(f"  per-problem contrast: mean {d.mean():+.4f}  median {np.median(d):+.4f}"
          f"  range [{d.min():+.4f}, {d.max():+.4f}]")
    # exact two-sided sign test against a fair coin
    from math import comb
    tail = sum(comb(n, k) for k in range(neg, n + 1)) / 2 ** n
    print(f"  exact sign test, two-sided p = {min(1.0, 2 * tail):.2e}")

    print("\n" + "=" * 74)
    print("3. IS IT LENGTH?")
    print("=" * 74)
    per_src, corrs = length_check(X, pid, src, ntok)
    print(f"  {'source':<34}{'median tokens':>14}{'mean':>9}")
    for s, v in sorted(per_src.items(), key=lambda t: -np.median(t[1])):
        print(f"  {s:<34}{np.median(v):>14.0f}{v.mean():>9.1f}")
    print(f"\n  |correlation| between PC1 and token count, over {len(corrs)} problems:")
    print(f"    median {np.median(corrs):.3f}   mean {corrs.mean():.3f}"
          f"   range [{corrs.min():.3f}, {corrs.max():.3f}]")

    print("\n" + "=" * 74)
    print("4. LENGTH HELD FIXED: least squares on pair cosines")
    print("=" * 74)
    rg = regress_pairs(X, pid, src, app, ntok)
    A = np.array([r[2:] for r in rg])
    print(f"  fitted inside each of {len(rg)} problems, coefficients averaged\n")
    print(f"  {'term':<26}{'mean':>10}{'median':>10}{'sign':>10}")
    terms = ["intercept", "same method", "same source",
             "|len diff| per 100 tok", "mean len per 100 tok"]
    for j, t in enumerate(terms):
        col = A[:, j]
        pos = int((col > 0).sum())
        print(f"  {t:<26}{col.mean():>+10.4f}{np.median(col):>+10.4f}{pos:>7}/{len(col)}")
    sm, ss = A[:, 1], A[:, 2]
    print(f"\n  same method minus same source: mean {(sm-ss).mean():+.4f}"
          f"  median {np.median(sm-ss):+.4f}"
          f"  method wins in {int((sm>ss).sum())}/{len(sm)} problems")

    print("\n" + "=" * 74)
    print("5. CAN THE AUTHOR DIRECTION SIMPLY BE REMOVED?")
    print("=" * 74)
    acc0, tot = probe_source(X, pid, src)
    print(f"  nearest-centroid author probe, before: {acc0:.3f}"
          f"   (chance {1/nsrc:.3f}, n={tot:,})")
    Y = erase_sources(X, pid, src)
    acc1, _ = probe_source(Y, pid, src)
    print(f"  nearest-centroid author probe, after : {acc1:.3f}")
    c1 = cells(Y, pid, src, app)
    print()
    for k in ("ss", "sd", "ds", "dd"):
        print(f"  {name[k]:30s} {c0[k][0]:.4f}  ->  {c1[k][0]:.4f}")
    print(f"\n  contrast {contrast(c0):+.4f}  ->  {contrast(c1):+.4f}")
    rows1 = per_problem(Y, pid, src, app)
    d1, neg1, n1 = paired_by_problem(rows1)
    print(f"  problems where the author still wins: {neg1}/{n1}"
          f"  (median contrast {np.median(d1):+.4f})")
    print("\n  Removing k-1 of 4,096 directions per problem. Whether this is legitimate for the"
          "\n  real experiment is a separate question: there the groups being centred are the"
          "\n  checkpoints under comparison, so centring them also deletes any genuine shift in"
          "\n  their method distribution. See the discussion in Cal_E.")

    print("\n" + "=" * 74)
    print("6. IS THE AUTHOR EFFECT REALLY THE VERBOSITY EFFECT?")
    print("=" * 74)
    med, subs = source_subsets(X, pid, src, app, ntok)
    print("  every three-source subset, ordered by verbosity spread\n")
    print(f"  {'spread':>8}  {'contrast':>9}  sources")
    for spread, d, tri in subs:
        mark = "   <- method wins" if d > 0 else ""
        names = ", ".join(t.split("/")[-1][:14] for t in tri)
        print(f"  {spread:>6.0f}t  {d:>+9.4f}  {names}{mark}")
    won = sum(1 for _, d, _ in subs if d > 0)
    tight = [d for spread, d, _ in subs if spread <= 20]
    print(f"\n  method wins in {won}/{len(subs)} subsets")
    print(f"  among the {len(tight)} tightest subsets (spread <= 20 tokens): "
          f"median contrast {np.median(tight):+.4f}")
    print("  If verbosity were the whole story the tightest subsets would be the ones where"
          "\n  method wins. They are not, so the author effect is not merely a length effect.")

    print("\n" + "=" * 74)
    print("7. HOW PRECISE IS THE CONTRAST? bootstrap over texts, not pairs")
    print("=" * 74)
    bs = bootstrap_texts(X, pid, src, app)
    print(f"  {len(bs)} resamples of whole texts within each problem")
    print(f"    mean {bs.mean():+.4f}   sd {bs.std(ddof=1):.5f}")
    print(f"    95 pct interval [{np.percentile(bs, 2.5):+.4f}, "
          f"{np.percentile(bs, 97.5):+.4f}]")
    print(f"    sign unchanged in {int((bs < 0).sum())}/{len(bs)} resamples")
    print("  A pair-level shuffle treats every pair as an observation, which overstates how much"
          "\n  independent evidence there is, because each text enters many pairs. This interval"
          "\n  is the honest one to quote, and it still excludes zero.")


if __name__ == "__main__":
    main()
