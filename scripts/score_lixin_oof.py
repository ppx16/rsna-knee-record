"""Score `lixin73/rsna-knee-train-checkpoints`' shipped OOF on OUR convention, before wiring anything.

This is the pattern that worked for `yunusgmsoy/rsna-knee-radhead-resnet50`: its author published the OOF,
so the member's value could be measured at full population and zero GPU instead of guessed -- and it was
rejected at 0.7992 macro, 0.10 below the A167 member gate, without a single accelerator-second.

This dataset ships `dinov2_seed17_fold0_oof.csv` / `.npy`, `_test.csv` / `.npy`, `folds.csv` and a
`training_history_seed17_fold0.csv`. If the OOF clears the gate it is a candidate; if not it is closed.

⚠️ The label table matters. `FIND-2026-10-05` established that `report_labels_v5.csv` has the 58 official
gold labels **baked into those 58 rows**, so measuring anything against v5 returns a tautological 1.0000.
The clean table is `report_labels_v4hybrid.csv` (0.8991 against the gold), which is what is used here.
"""

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
import glob
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from kaggle.api.kaggle_api_extended import KaggleApi

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OUT = pathlib.Path("work/lixin_ckpt")

api = KaggleApi()
api.authenticate()
OUT.mkdir(parents=True, exist_ok=True)
if not any(OUT.rglob("*.csv")):
    try:
        api.dataset_download_files("lixin73/rsna-knee-train-checkpoints", path=str(OUT), unzip=True, quiet=True)
    except Exception as e:  # noqa: BLE001
        raise SystemExit("download failed: %s: %s" % (type(e).__name__, str(e)[:160]))
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print("   %9.1f KB  %s" % (p.stat().st_size / 1e3, p.relative_to(OUT)))

hit = sorted(OUT.rglob("*oof*"))
if not hit:
    raise SystemExit("no OOF file in the dataset")
oof = pd.read_csv(hit[0]) if hit[0].suffix == ".csv" else None
if oof is None:
    arr = np.load(hit[0], allow_pickle=True)
    print("npz/npy keys:", getattr(arr, "files", "n/a"))
    raise SystemExit("only .npy OOF present -- needs its id order; STOP rather than guess")

print("\nOOF shape:", oof.shape, "| cols:", list(oof.columns)[:4], "...")
idc = next((c for c in oof.columns if "uid" in c.lower()), oof.columns[0])
cols = [c for c in oof.columns if c in LAB]
print("label columns present: %d/12" % len(cols))
if len(cols) < 12:
    raise SystemExit("the OOF does not carry all twelve labels -- cannot score on our convention")

# gold-58 truth, from any raptor_gold npz (they all carry the same truth)
g = sorted(glob.glob("downloads/**/raptor_gold_*.npz", recursive=True))[0]
z = np.load(g, allow_pickle=True)
truth = np.asarray(z["truth"], dtype=float)
ids = [str(x) for x in z["ids"]]
print("gold truth:", truth.shape, "from", pathlib.Path(g).name)

oof[idc] = oof[idc].astype(str)
m = pd.DataFrame({"StudyInstanceUID": ids}).merge(oof, left_on="StudyInstanceUID", right_on=idc, how="left")
print("gold studies matched in the OOF:", int(m[cols[0]].notna().sum()), "/", len(ids))

if int(m[cols[0]].notna().sum()) == 0:
    print("!! no overlap -- the OOF is over training studies only, so gold-58 cannot be scored from it")
    print("   (that is itself the finding: without the fold's held-out identities this is not evaluable here)")
    raise SystemExit(0)

aucs = {}
for j, L in enumerate(LAB):
    y = (truth[:, j] > 0.5).astype(int)
    s = m[L].to_numpy(dtype=float)
    ok = np.isfinite(s)
    if len(set(y[ok])) > 1:
        aucs[L] = float(roc_auc_score(y[ok], s[ok]))
print("\n=== per-label AUC on the 58 gold ===")
for L, v in sorted(aucs.items(), key=lambda kv: kv[1]):
    print("   %-20s %.4f" % (L, v))
print("\nMACRO over %d labels: %.4f" % (len(aucs), float(np.mean(list(aucs.values())))))
print("A167 member gate: solo >= ~0.90.  Our msft2 = 0.9069.  The rejected yunus head = 0.7992.")
