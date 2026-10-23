"""Convention- and per-label robustness of the EffB0 dose conclusion.

TWO QUESTIONS, both about whether the headline claim is real or an artefact of one convention:

  1. SOFT vs BINARY. The +0.00493 at w=0.15 was measured with the SOFT convention (the project's
     pre-registered one). C84 records the two conventions disagreeing in MAGNITUDE on the same
     change (+0.0068 soft vs +0.0129 binary), so a conclusion that exists under only one of them
     is suspect. Both are computed here on the same honest per-fold split.

  2. PER-LABEL. The central selling point of this leg is that it helps MOST WHERE WE ARE WEAKEST --
     Lateral Meniscus, the largest single shortfall in the macro average, at +0.0131. That is the
     reason it was pursued at all, and it has not been checked independently. If the gain is
     spread evenly, or concentrated in labels we are already good at, the leg's rationale weakens
     even if its mean is positive.

Usage:  python work/eff_convention.py
"""
from __future__ import annotations


# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded
# `print("... \u26a0 ...")` raises UnicodeEncodeError at that line and kills the script MID-REPORT, while
# everything printed earlier is already on screen -- so the run looks complete. Self-contained on purpose:
# it imports `sys` itself and depends on nothing else in this tree.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream: leave it alone, prints still degrade
    pass
# ----------------------------------------------------------------------------------------------------
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eff_honest import _col_auc, macro_auc, rankpct  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OOF = ROOT / "downloads" / "oof"
EFF = ROOT / "work" / "eff_assets"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
W_TEST = [0.10, 0.15, 0.20]


def main() -> int:
    ours = np.load(OOF / "oof_equal_prob.npy")
    y = np.load(OOF / "train_y.npy")
    ids = np.asarray([str(x) for x in np.load(OOF / "train_ids.npy", allow_pickle=True).ravel()])
    idx = {u: i for i, u in enumerate(ids)}

    leg = np.full_like(ours, np.nan, dtype=np.float64)
    fold_of = np.full(len(ids), -1, dtype=int)
    for k in range(4):
        p = np.load(EFF / ("oof_effb0mil_sl16_tv7w_f%d.npy" % k))
        pi = pd.read_csv(EFF / ("oof_effb0mil_sl16_tv7w_f%d_ids.csv" % k)).iloc[:, 0].astype(str)
        for r, u in enumerate(pi):
            j = idx.get(u)
            if j is not None:
                leg[j] = p[r]
                fold_of[j] = k
    fold_of[fold_of < 0] = 0

    br = rankpct(ours)
    lr = rankpct(leg)
    lr = np.where(np.isnan(lr), np.nanmedian(lr, axis=0), lr)

    # ---- 1. SOFT vs BINARY, honest per-fold ------------------------------------------
    print("==== soft vs binary, mean gain per fold (4-fold mean) ====")
    print("%-7s %-22s %-22s" % ("w", "SOFT", "BINARY (y>=0.5)"))
    ybin = (y >= 0.5).astype(np.float64)
    for w in W_TEST:
        gs, gb = [], []
        for k in range(4):
            te = fold_of == k
            bl = (1.0 - w) * br[te] + w * lr[te]
            gs.append(macro_auc(y[te], bl) - macro_auc(y[te], br[te]))
            gb.append(macro_auc(ybin[te], bl) - macro_auc(ybin[te], br[te]))
        print("%-7.2f %+.5f (%d/4 +)     %+.5f (%d/4 +)"
              % (w, np.mean(gs), int((np.array(gs) > 0).sum()),
                 np.mean(gb), int((np.array(gb) > 0).sum())))

    # ---- 2. PER-LABEL at w=0.15 ------------------------------------------------------
    w = 0.15
    print("\n==== per-label AUC at w=0.15 (whole covered set, SOFT) ====")
    print("%-18s %9s %9s %9s   %s" % ("label", "parent", "blend", "delta", "note"))
    rows = []
    for j, name in enumerate(LABELS):
        a0 = _col_auc(y[:, j], br[:, j])
        a1 = _col_auc(y[:, j], (1.0 - w) * br[:, j] + w * lr[:, j])
        rows.append((name, a0, a1, a1 - a0))
    for name, a0, a1, d in sorted(rows, key=lambda r: -(r[3])):
        note = ""
        if a0 == min(r[1] for r in rows):
            note = "<-- weakest parent label"
        print("%-18s %9.5f %9.5f %+9.5f   %s" % (name, a0, a1, d, note))
    ds = np.array([r[3] for r in rows])
    print("\nper-label: mean %+.5f  positive %d/12  worst %+.5f  best %+.5f"
          % (ds.mean(), int((ds > 0).sum()), ds.min(), ds.max()))
    weak = sorted(rows, key=lambda r: r[1])[:3]
    strong = sorted(rows, key=lambda r: -r[1])[:3]
    print("weakest-3 parent labels  mean delta %+.5f" % np.mean([r[3] for r in weak]))
    print("strongest-3 parent labels mean delta %+.5f" % np.mean([r[3] for r in strong]))
    print("\n=> the leg's rationale ('helps most where we are weakest') is %s"
          % ("SUPPORTED" if np.mean([r[3] for r in weak]) > np.mean([r[3] for r in strong])
             else "NOT SUPPORTED -- the gain is not concentrated on the weak labels"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
