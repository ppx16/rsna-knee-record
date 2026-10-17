# -*- coding: utf-8 -*-
"""
The LARGE-N instrument, built from the raptor_oof_*_fold0.npz files already on disk.

Why this exists: the new goal needs an instrument that can resolve >=0.005. gold-58's
paired 95% CI half-width is +/-0.021 (D16/C68) -- it cannot. The fold-0 held-out OOF is
~870 studies with a +/-0.004 CI (D16), which can. That is the instrument the density-axis
verdict must be read on, and it costs nothing to build from files we already have.

Two metrics per arm, because they answer different questions:
  binary  truth > 0.5        -- the competition analogue, comparable to recorded numbers
  soft    truth as weights   -- sharper (HANDOFF 101 / D16 says prefer it)
Levels are NOT comparable across the two; compare verdicts and CIs, never the levels.
"""
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def auc_soft(y, p):
    """Weighted Mann-Whitney AUC, O(n log n).

        AUC = sum_{i != j} w_i v_j c_ij  /  sum_{i != j} w_i v_j
        w = y,  v = 1 - y,  c_ij = 1 if p_i > p_j, 0.5 if equal, else 0

    Reduces exactly to roc_auc_score when y is binary (then w_i v_i = 0 for every i, so
    excluding self-pairs changes nothing).
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    w = y
    v = 1.0 - y
    denom = w.sum() * v.sum() - float((w * v).sum())      # exclude self-pairs
    if denom <= 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ps, ws, vs = p[order], w[order], v[order]
    cw = np.concatenate(([0.0], np.cumsum(ws)))           # cumulative weight
    cv = np.concatenate(([0.0], np.cumsum(vs)))           # cumulative (1-weight)
    num = 0.0
    i = 0
    n = len(ps)
    while i < n:
        j = i
        while j + 1 < n and ps[j + 1] == ps[i]:
            j += 1
        below_v = cv[i]                                   # strictly-lower p mass
        grp_w = cw[j + 1] - cw[i]
        grp_v = cv[j + 1] - cv[i]
        # every i in the group beats all strictly-lower j
        num += grp_w * below_v
        # ties inside the group count half, excluding self-pairs
        num += 0.5 * (grp_w * grp_v - float((ws[i:j + 1] * vs[i:j + 1]).sum()))
        i = j + 1
    return float(num / denom)


def macro(pred, truth, soft):
    vals = []
    for j in range(truth.shape[1]):
        y = truth[:, j] if soft else (truth[:, j] > 0.5).astype(np.float64)
        if y.min() == y.max():
            continue
        a = auc_soft(y, pred[:, j])
        if a is not None:
            vals.append(a)
    return float(np.mean(vals)) if vals else None


def main():
    files = sorted(ROOT.rglob("raptor_oof_*_fold*.npz"))
    rows, ref_ids, ref_truth = [], None, None
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
            pred = np.asarray(z["pred"], np.float64)
            truth = np.asarray(z["truth"], np.float64)
            ids = [str(u) for u in z["ids"]]
        except Exception:
            continue
        if pred.ndim != 2 or pred.shape[1] != 12 or pred.shape[0] < 200:
            continue
        if ref_ids is None:
            ref_ids, ref_truth = ids, truth
        same = (len(ids) == len(ref_ids) and ids == ref_ids
                and truth.shape == ref_truth.shape and np.allclose(truth, ref_truth))
        rows.append((f.relative_to(ROOT).as_posix(), len(ids), bool(same),
                     macro(pred, truth, False), macro(pred, truth, True)))
    if not rows:
        print("no OOF files with >=200 studies")
        return
    big = [r for r in rows if r[1] >= 800]
    print(f"reference split: {len(ref_ids)} studies")
    print(f"{len(rows)} OOF files total; {len(big)} on the large-n split (>=800 studies)")
    print()
    big.sort(key=lambda r: -(r[4] if r[4] is not None else -1))
    print(f"{'soft':>8} {'binary':>8} {'n':>5} {'same':>6}  file")
    print("-" * 104)
    for rel, n, same, b, s in big[:28]:
        bs = "     nan" if b is None else f"{b:8.4f}"
        ss = "     nan" if s is None else f"{s:8.4f}"
        print(f"{ss} {bs} {n:5d} {str(same):>6}  {rel}")


if __name__ == "__main__":
    main()
