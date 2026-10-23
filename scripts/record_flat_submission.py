"""Append the cfmsft2flat submission to the ledger and print the current pending set."""

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

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
api = KaggleApi()
api.authenticate()

subs = api.competition_submissions(COMP)
row = next((s for s in subs if str(getattr(s, "ref", "")) == "56848247"), None)
date = str(getattr(row, "date", "")) if row else ""
print("date of 56848247:", date)

DESC = ("0943cfmsft2flat - cfmsft2 with the outer per-label routing FLATTENED (preset=parent). "
        "Differs from 0943cfmsft2 by exactly ONE line (cell diff) and the run log confirms it at "
        "RUNTIME: per-finding outer w all 0.650 vs cfmsft2's non-uniform map (ACL/Fracture/LatOA "
        "0.750, LatMen 1.000, MedMen 0.800), final SHA 095bcf70 -> c1478aac. Strictly improves the "
        "final pair because the pair's value is the better of the two and cfmsft2 stays in it. "
        "Before-snapshot 2026-10-05T09:14:38Z: rank 523, line 519, margin -4, in-block 13.92%")

led = pathlib.Path("state/lb_ground_truth.csv")
with led.open("a", encoding="utf-8", newline="") as fh:
    csv.writer(fh).writerow([date, "56848247", DESC, "", "PENDING"])
print("appended ledger row")

print()
print("=== all non-COMPLETE submissions ===")
for s in sorted(subs, key=lambda s: str(getattr(s, "date", "")), reverse=True):
    st = str(getattr(s, "status", "")).split(".")[-1]
    if st != "COMPLETE":
        print("  %s  ref=%s  %s  %s" % (getattr(s, "date", ""), getattr(s, "ref", ""), st,
                                        str(getattr(s, "description", ""))[:56]))
print()
print("=== 3 newest COMPLETE ===")
n = 0
for s in sorted(subs, key=lambda s: str(getattr(s, "date", "")), reverse=True):
    if str(getattr(s, "status", "")).split(".")[-1] == "COMPLETE":
        print("  %s  ref=%s  %s" % (getattr(s, "date", ""), getattr(s, "ref", ""),
                                    str(getattr(s, "description", ""))[:56]))
        n += 1
        if n == 3:
            break
