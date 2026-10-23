"""Strongest local preflight: build the FULL member model and load the checkpoint
strict=True, without a GPU and without the competition data.

This is the check that catches interface drift -- a renamed class, a changed
constructor signature, a missing tensor. If it passes, the remaining unknowns are
runtime cost and the blend weight, not wiring.
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
import pathlib
import sys

X = pathlib.Path(r"D:\kaggle\rsna-knee\work\cnx\x")
sys.path.insert(0, str(X))

import torch  # noqa: E402

print("--- importing the member's own modules (as the script does) ---")
try:
    import btkr_cnx_montage448_optional as member
    print("  import OK")
except Exception as e:
    print(f"  IMPORT FAILED: {type(e).__name__}: {e}")
    raise SystemExit(1)

print(f"  LABELS          = {member.LABELS}")
print(f"  BACKBONE        = {member.BACKBONE}")
print(f"  BAND            = {member.BAND}")
print(f"  MONTAGE_SIZE    = {member.MONTAGE_SIZE}")
print(f"  TILES_PER_MONTAGE = {member.TILES_PER_MONTAGE}")
print(f"  len(SLOT_NAMES) = {len(member.SLOT_NAMES)}  {member.SLOT_NAMES}")
print(f"  CENTER_BUDGETS keys = {sorted(member.CENTER_BUDGETS)}")

print("\n--- building MontageSubmissionNet ---")
net = member.MontageSubmissionNet()
n_params = sum(p.numel() for p in net.parameters())
print(f"  built: {type(net).__name__}, params = {n_params:,}")

print("\n--- loading checkpoint strict=True ---")
payload = torch.load(X / "cnx_montage448_f0.pt", map_location="cpu", weights_only=False)
try:
    net.load_state_dict(payload["state_dict"], strict=True)
    print("  *** strict load OK -- wiring is exact ***")
except Exception as e:
    print(f"  strict load FAILED: {type(e).__name__}")
    print(f"  {str(e)[:1500]}")
    raise SystemExit(2)

print("\n--- echoing the script's own load_model() guard ---")
m = member.load_model(X / "cnx_montage448_f0.pt",
                      "3b7f50c7bd16d353b8d79bf6089530a376009b6c3480e1bd97b778aa1f4ecab8")
print(f"  load_model() OK -> {type(m).__name__}, training={m.training}")

print("\n--- serving contract (what the author pinned for this exact weight file) ---")
for k in ("montage_serving_contract", "w0_serving_contract"):
    v = payload.get(k)
    print(f"  {k}: {str(v)[:400]}")
