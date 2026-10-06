"""Fill in the two PENDING ledger rows with their measured public scores."""

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
import csv
import pathlib

LED = pathlib.Path("state/lb_ground_truth.csv")
rows = list(csv.reader(LED.open(encoding="utf-8", newline="")))

SCORES = {"56845973": "0.940", "56848247": "0.942"}
n = 0
for r in rows:
    if len(r) >= 2 and r[1] in SCORES:
        r[3] = SCORES[r[1]]
        r[4] = "COMPLETE"
        n += 1
        print("updated ref %s -> %s COMPLETE" % (r[1], r[3]))

with LED.open("w", encoding="utf-8", newline="") as fh:
    csv.writer(fh).writerows(rows)
print("rows updated:", n)
print()
print("=== last 6 ledger rows ===")
for r in rows[-6:]:
    print("  %s | %s | %s | %s" % (r[0][:19], r[1], r[3] or "-", r[2][:70]))
