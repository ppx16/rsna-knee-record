"""Save the competition pages and pull out the sentences that govern FINAL SUBMISSION SELECTION.

Question under audit: is the selectable pool ALL submissions, or only the most recent N?
`docs/FINAL-SELECTION-*.md` assumes the former and never checks it. `56613303` was submitted
2026-09-27, so if the pool is "the last N" the recommendation is worthless.
"""
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
