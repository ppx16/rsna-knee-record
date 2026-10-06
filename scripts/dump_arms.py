# -*- coding: utf-8 -*-
"""Print the Raptor ARMS table as the KERNEL will see it.

The arm table lives inside a quoted `_KE_SRC` string, so reading the .ipynb text (or
grepping it) can look right and still be wrong. This parses the module back with the same
`_arms_from_module` the experiment builder uses, so it reports the geometry that will
actually run.

Cross-checked against the author's dataset descriptions, which state the corpus each
checkpoint belongs to:
    maxspan-v5 / -reverse   (336, 64 slots, span .02-.98)  -> "every window position" = 62
    native384dense-v10      (384, 64 slots, span .02-.98)  -> 62
    native384-v8            (384, 44 slots, span .06-.94)  -> "every window position" = 42
    finespacing-v9          (336, 80 slots, span .02-.98)  -> 78
A window count that does not match the volume is the mismatch the author explicitly warns
about: "the 0.932 will not be there".
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
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ens_experiment as E  # noqa: E402

nb_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
    pathlib.Path("kernels/kernel-ens/rsna-knee-ensemble-v1.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
arms = E._arms_from_module(E.extract_module(nb))

print("%-22s %5s %6s %5s %-14s %6s %6s" %
      ("name", "img", "slots", "slices", "span", "k_eval", "w"))
print("-" * 74)
for a in arms:
    slots = a["slots"]
    total = sum(s[2] for s in slots)
    flag = ""
    if total > 0 and a["k_eval"] != total - 2:
        flag = "   <-- k_eval != slots-2"
    print("%-22s %5s %6d %5d %-14s %6d %6s%s" %
          (a["name"], a.get("img"), len(slots), total, str(a["span"]), a["k_eval"], a["w"], flag))
print("\ntotal weight: %.4f" % sum(float(a["w"]) for a in arms))
