"""Extract Stack++'s reader source (`preprocess.py` / `knee.py` / `infer.py`) from its notebook.

The TTA variant is pre-registered (`docs/VERIFY-2026-10-06-reader-leg-is-clean.md`) and its changes are now
understood (`docs/READ-2026-10-06e-stackpp-reader-changes.md`). What is missing is the CODE: our current
reader arm runs goodpjw2008's three files, and the variant needs Stack++'s versions, which the notebook
writes with `%%writefile`.

This extracts them verbatim. It does NOT judge them again -- that was done -- and it does NOT build or push
anything. Extracting now means the arm can be built the moment the base result lands, instead of starting
from zero then.
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
OUT = pathlib.Path("work/stackpp_src")
OUT.mkdir(parents=True, exist_ok=True)

nb = json.loads(list(SRC.rglob("*.ipynb"))[0].read_text(encoding="utf-8", errors="replace"))
cells = [("".join(c.get("source", [])) if isinstance(c.get("source"), list) else (c.get("source") or ""))
         for c in nb["cells"] if c.get("cell_type") == "code"]

WANT = {"preprocess.py", "knee.py", "infer.py"}
found = {}
for c in cells:
    m = re.match(r"\s*%%writefile\s+(\S+)\s*\n", c)
    if not m:
        continue
    name = pathlib.Path(m.group(1)).name
    if name in WANT:
        body = c[m.end():]
        found[name] = body
        (OUT / name).write_text(body, encoding="utf-8")
        print("extracted %-16s %6d chars -> %s" % (name, len(body), (OUT / name).relative_to(OUT.parent.parent)))

missing = WANT - set(found)
if missing:
    raise SystemExit("!! not found in the notebook: %s" % sorted(missing))

print()
print("=== what makes Stack++'s files different from goodpjw2008's (the base we run today) ===")
base = pathlib.Path("work/reader")
for name in sorted(WANT):
    a = (base / name)
    b = (OUT / name)
    if not a.is_file():
        print("   %-16s base copy not on disk" % name)
        continue
    ta, tb = a.read_text(encoding="utf-8", errors="replace"), b.read_text(encoding="utf-8", errors="replace")
    import difflib
    d = [ln for ln in difflib.unified_diff(ta.splitlines(), tb.splitlines(), lineterm="", n=0)
         if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]
    print("   %-16s base %6d -> variant %6d chars | %d changed lines" % (name, len(ta), len(tb), len(d)))
    for ln in d[:10]:
        print("        " + ln[:130])

print()
print("=== guard: does the variant still reference the competition layout and add NO leaderboard knob? ===")
import re as _re
text = "".join(found.values())
for pat, label in [(r"SERIES_ROOT|_series", "test-series path handling"),
                   (r"TTA_MODES|RSNA_READER_TTA", "TTA modes"),
                   (r"_rank_members", "rank ensemble"),
                   (r"leaderboard|public_lb", "LEADERBOARD REFERENCES (must be 0)"),
                   (r"Synovitis", "Synovitis handling")]:
    print("   %-38s x%d" % (label, len(_re.findall(pat, text, _re.I))))
