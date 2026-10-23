"""Extract the Stack++ reader changes from `romantamrazov/rsna-knee-dinosaur-v5` and judge them.

The prereg for the TTA variant (`docs/VERIFY-2026-10-06-reader-leg-is-clean.md`) names exactly two changes:
  (1) fold x TTA **rank** ensemble before blending, instead of `infer.py`'s probability mean;
  (2) inference slice centres aligned to the training-bin centres, with an extra original-grid TTA pass.

Both are already written by that author. This reads them rather than re-deriving them, and checks the one
thing that decides whether the prereg's claim ("TTA changes no fitted parameter") is true: does the code
only transform inputs and aggregate outputs, or does it consult anything tuned on the leaderboard?
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
import re

SRC = pathlib.Path("work/newpub_1006/rsna-knee-dinosaur-v5")
nb = json.loads(list(SRC.rglob("*.ipynb"))[0].read_text(encoding="utf-8", errors="replace"))
code = [("".join(c.get("source", [])) if isinstance(c.get("source"), list) else (c.get("source") or ""))
        for c in nb["cells"] if c.get("cell_type") == "code"]
allc = "\n\n# ===== cell =====\n\n".join(code)
print("code cells: %d, chars: %d" % (len(code), len(allc)))

print()
print("=== cells that touch the reader / TTA / rank ensemble ===")
for i, c in enumerate(code):
    if re.search(r"own_src|infer\.py|_own|TTA|tta|rank|flip|centre|center", c):
        hits = sorted(set(re.findall(r"[\w\-/]*(?:tta|TTA|flip|rank|centre|center|own)[\w\-/]*", c)))
        if len(hits) > 3:
            print("   cell %2d  %5d chars  terms: %s" % (i, len(c), hits[:12]))

print()
print("=== every mention of tta / flip / rank-ensemble with context ===")
for m in re.finditer(r"(TTA|tta|flip|original.grid|rank ensemble|rank_ensemble)", allc):
    a = max(0, m.start() - 180); b = min(len(allc), m.end() + 220)
    seg = re.sub(r"\s+", " ", allc[a:b])
    print("   ..." + seg[:360])
    print()

print("=" * 96)
print("=== does it consult anything PUBLIC-TUNED? (this decides the prereg's admissibility claim) ===")
for pat, label in [
    (r"public[_ ]?lb|leaderboard|lb_score|tuned", "leaderboard references"),
    (r"OWN_W\s*=\s*[\d.]+|WEIGHT\s*=\s*[\d.]+", "hard-coded weights"),
    (r"np\.argsort|rankdata|\.rank\(", "rank operations"),
    (r"torch\.flip|\[::-1\]|np\.flip", "flips (TTA)"),
    (r"interpolate|cv2\.resize", "resampling"),
]:
    h = re.findall(pat, allc, re.I)
    print("   %-24s x%-4d %s" % (label, len(h), sorted(set(h))[:6]))
