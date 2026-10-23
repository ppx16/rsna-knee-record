"""Push the TTA arm ONLY to verify its plumbing. No submission is created by a kernel push.

WHY THIS IS NOT FIRING THE EXPERIMENT. The prereg's gate (`docs/VERIFY-2026-10-06-reader-leg-is-clean.md`)
is about SPENDING A SUBMISSION on the arm -- and a `kernels_push` does not create one; every arm this
session pushed was submitted later, separately, via `competition_submit_code`. What a push does is run the
notebook on the 3-study public preview and produce a log.

So this buys the one thing static checks cannot: proof that Stack++'s three extracted source files
COMPOSE into a working kernel -- `infer.py` importing `knee`/`preprocess` beside it, the env-var TTA modes
being honoured, the checkpoint count guard passing, and the per-label blend actually executing. Finding a
break here costs ~8 GPU-minutes; finding it after the base result lands costs the decision.

⛔ IT WILL NOT SUBMIT. If the preview turns out useful it is still gated on the base leg.
"""
import json
import pathlib
import shutil
import subprocess
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
DST = ROOT / "kernels" / "kernel-0943cfmsft2-rdrtta"
REF = "ppxl16/rsna-knee-0943cfmsft2-rdrtta"

if not (DST / "kernel-metadata.json").is_file():
    raise SystemExit("!! not built -- run scripts/build_reader_arm_tta.py --build first")

meta = json.loads((DST / "kernel-metadata.json").read_text(encoding="utf-8"))
assert meta["id"] == REF, "kernel id drifted: %s" % meta["id"]
assert meta["enable_internet"] is False, "internet must be off"
print("pushing %s (verify-only; no submission will be created)" % REF)

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

api = KaggleApi()
api.authenticate()
for attempt in range(4):
    try:
        api.kernels_push(str(DST))
        print("pushed")
        break
    except Exception as e:  # noqa: BLE001
        print("push attempt %d failed: %s: %s" % (attempt + 1, type(e).__name__, str(e)[:160]))
        time.sleep(15)
else:
    raise SystemExit("push failed on every attempt")

# ⚠️ a push returning without an exception is NOT proof it registered (FACTS A32 records a push that
# reported success while pushing nothing), so ask the server.
time.sleep(20)
for attempt in range(3):
    try:
        st = str(getattr(api.kernels_status(REF), "status", ""))
        print("VERIFIED registered, status:", st)
        break
    except Exception as e:  # noqa: BLE001
        time.sleep(10)
else:
    print("!! could not confirm registration -- treat the push as UNVERIFIED")
