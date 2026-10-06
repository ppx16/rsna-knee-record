"""Save the competition pages and pull out the sentences that govern FINAL SUBMISSION SELECTION.

Question under audit: is the selectable pool ALL submissions, or only the most recent N?
`docs/FINAL-SELECTION-*.md` assumes the former and never checks it. `56613303` was submitted
2026-09-27, so if the pool is "the last N" the recommendation is worthless.
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
import re

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
OUT = pathlib.Path("work/comp_pages")
OUT.mkdir(parents=True, exist_ok=True)

api = KaggleApi()
api.authenticate()
for p in api.competition_list_pages(COMP):
    c = getattr(p, "content", "") or ""
    (OUT / ("%s.txt" % p.name.replace(" ", "_"))).write_text(c, encoding="utf-8")
print("saved %d pages to %s" % (len(list(OUT.glob('*.txt'))), OUT))

rules = (OUT / "rules.txt").read_text(encoding="utf-8")
text = re.sub(r"<[^>]+>", " ", rules)
text = re.sub(r"&nbsp;?", " ", text)
text = re.sub(r"\s+", " ", text)

KEYS = ("select", "final submission", "two submission", "2 submission", "most recent",
        "last submission", "chosen", "choose", "retain", "eligible")
sents = re.split(r"(?<=[.!?])\s+", text)
seen = 0
print("\n=== sentences mentioning selection ===")
for s in sents:
    low = s.lower()
    if any(k in low for k in KEYS):
        s = s.strip()
        if 20 < len(s) < 600:
            print("  *", s)
            seen += 1
print("\n(%d sentences)" % seen)
