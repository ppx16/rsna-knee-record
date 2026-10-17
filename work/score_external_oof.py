"""Score an EXTERNAL fold-0 OOF prediction with our own instrument.

Question this answers: lixin73 published `oof_mavit_fold0.csv` -- a MaVIT (a different
architecture family) fold-0 out-of-fold prediction. Before spending any GPU on wiring it in
as an extra leg, measure how strong and how DIFFERENT it actually is on OUR fold-0 holdout,
using the same labels our arms were scored against.

Outputs: per-finding AUC, macro AUC (binary at >0.5, the recorded convention), and the
Spearman correlation against our own arms' fold-0 OOF -- because a leg only adds value if it
is both decent AND decorrelated (a near-duplicate adds nothing: FACTS C14/C16).

Usage: python work/score_external_oof.py scratch/mavit/oof_mavit_fold0.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC; ties average. Returns nan if a class is absent."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = float((y > 0.5).sum())
    n0 = float(len(y) - n1)
    return float((r[y > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def find_reference() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Pick a cv arm dir that has labels.parquet, and load available arm OOFs."""
    ref_labels = None
    for p in sorted((ROOT / "downloads" / "cv").glob("*/labels.parquet")):
        df = pd.read_parquet(p)
        if "StudyInstanceUID" in df.columns and all(c in df.columns for c in LABELS):
            ref_labels = df
            print(f"labels: {p.relative_to(ROOT)}  ({len(df)} rows)")
            break
    if ref_labels is None:
        raise SystemExit("no labels.parquet with the 12 findings found under downloads/cv/*")

    arms: dict[str, pd.DataFrame] = {}
    for npz in sorted((ROOT / "downloads" / "cv").glob("*/raptor_oof_*_fold0.npz")):
        try:
            z = np.load(npz, allow_pickle=True)
        except Exception:  # noqa: BLE001
            continue
        keys = set(z.files)
        # our own dumps use "ids"; older/other tools may use "uids"
        idkey = "ids" if "ids" in keys else ("uids" if "uids" in keys else None)
        if idkey is None or "pred" not in keys:
            continue
        ids = np.asarray(z[idkey]).astype(str)
        arms[f"{npz.parent.name}/{npz.stem.replace('raptor_oof_', '')}"] = pd.DataFrame(
            z["pred"], index=ids, columns=LABELS)
    if arms:
        print(f"our fold-0 arm OOFs available: {len(arms)} "
              f"(e.g. {sorted(arms)[:4]})")
    return ref_labels, arms


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    ext_path = Path(argv[1])
    ext = pd.read_csv(ext_path)
    if "StudyInstanceUID" not in ext.columns:
        ext = pd.read_csv(ext_path, index_col=False)
    ext = ext.rename(columns={c: c.strip() for c in ext.columns})
    ext["StudyInstanceUID"] = ext["StudyInstanceUID"].astype(str)
    ext = ext.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    print(f"external: {ext_path.name}  {len(ext)} rows")

    ref, arms = find_reference()
    ref = ref.copy()
    ref["StudyInstanceUID"] = ref["StudyInstanceUID"].astype(str)
    ref = ref.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")

    common = ref.index.intersection(ext.index)
    print(f"matched studies: {len(common)}  (of {len(ref)} reference rows, {len(ext)} external)")
    if len(common) < 100:
        print("!! too few matched studies -- check the split / id format")
        return 1

    y = ref.loc[common, LABELS]
    print("\n=== external model, per-finding AUC on our fold-0 holdout ===")
    per = {}
    for c in LABELS:
        per[c] = auc(y[c].to_numpy(), ext.loc[common, c].to_numpy())
    pos = {c: int((y[c].to_numpy() > 0.5).sum()) for c in LABELS}
    for c in LABELS:
        print(f"  {c:18s} AUC {per[c]:.4f}   positives {pos[c]:4d}")
    vals = [v for v in per.values() if np.isfinite(v)]
    print(f"  {'MACRO':18s} AUC {np.mean(vals):.4f}   ({len(vals)}/12 findings scorable)")

    if arms:
        print("\n=== apples-to-apples: external vs our arms on the SAME studies ===")
        print("  (the external file uses ITS OWN split -- 882 rows vs our 870 -- so every")
        print("   comparison below is restricted to the intersection, never the raw file.)")
        # intersection of ALL our arms and the external model
        common_all = set(common)
        for df in arms.values():
            common_all &= set(df.index.astype(str))
        common_all = sorted(common_all)
        print(f"  studies present in the external file AND in every one of our arms: "
              f"{len(common_all)}")
        if len(common_all) < 100:
            print("  !! intersection too small to compare")
            return 0
        yy = ref.loc[common_all, LABELS]
        vv = [auc(yy[c].to_numpy(), ext.loc[common_all, c].to_numpy()) for c in LABELS]
        ext_macro = float(np.mean([v for v in vv if np.isfinite(v)]))
        print(f"\n  {'external MaVIT':28s} macro {ext_macro:.4f}  (n={len(common_all)})")

        ranked = []
        for name, df in arms.items():
            d = df.copy()
            d.index = d.index.astype(str)
            vv2 = [auc(yy[c].to_numpy(), d.loc[common_all, c].to_numpy()) for c in LABELS]
            vv2 = [v for v in vv2 if np.isfinite(v)]
            ranked.append((float(np.mean(vv2)), name, d))
        ranked.sort(reverse=True)
        for m, name, _ in ranked[:8]:
            print(f"  {name:28s} macro {m:.4f}")
        best_m, best_name, _ = ranked[0]
        print(f"\n  => external {ext_macro:.4f} vs our best {best_name} {best_m:.4f}"
              f"   delta {ext_macro - best_m:+.4f}")

        print("\n=== diversity: Spearman vs our arms (a leg must be decent AND decorrelated)")
        for m, name, d in ranked[:6]:
            rhos = [pd.Series(ext.loc[common_all, c].to_numpy()).corr(
                pd.Series(d.loc[common_all, c].to_numpy()), method="spearman") for c in LABELS]
            print(f"  vs {name:28s} mean rho {np.nanmean(rhos):.3f}")

        print("\n=== would blending add? (rank-mean, external at weight w) ===")
        _, _, d_best = ranked[0]
        for w in (0.2, 0.3, 0.4, 0.5):
            bl = [(1 - w) * d_best.loc[common_all, c].rank(pct=True)
                  + w * ext.loc[common_all, c].rank(pct=True) for c in LABELS]
            mix = pd.DataFrame({c: v.to_numpy() for c, v in zip(LABELS, bl)})
            mv = [auc(yy[c].to_numpy(), mix[c].to_numpy()) for c in LABELS]
            mv = [v for v in mv if np.isfinite(v)]
            print(f"  w_external={w:.1f}: macro {np.mean(mv):.4f}  "
                  f"({np.mean(mv) - best_m:+.4f} vs our best alone)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
