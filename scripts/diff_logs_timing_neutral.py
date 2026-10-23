"""Did the MCL edit reach the graph?  Forensic log diff, timing-neutralised.

What we know:
  * the notebook that ran HAS `_RAD_EXCLUDE = ("Baker's", "MCL")` -- verified by reading it back;
  * the visible submission.csv is BYTE-IDENTICAL to the parent's;
  * A80 already established that an identical visible submission does NOT prove a change
    failed to apply (3 visible studies => only 3! orderings per label);
  * the _coat_arm_receipt.json differs ONLY in elapsed/load seconds -- zero content fields --
    so it is NOT evidence either way.

So the remaining evidence is the run LOG.  Both logs are compared after removing every
timing/resource token, so a difference can only come from content.
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

A = pathlib.Path(r"D:\kaggle\rsna-knee\work\rcpt_parent\rsna-knee-0943cfmsft2.log")
B = pathlib.Path(r"D:\kaggle\rsna-knee\work\cfmsft2m-out\rsna-knee-0943cfmsft2m.log")

TIME_RX = [
    re.compile(r"\d+\.\d+\s*s\b"),
    re.compile(r"\d+\.\d+"),
    re.compile(r'"time":\s*[0-9.]+'),
    re.compile(r"elapsed[_a-z]*[\"']?\s*[:=]\s*[0-9.]+"),
    re.compile(r"\b\d{4,}\b"),
]


def lines(path):
    out = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip().lstrip(",")
        if not raw:
            continue
        try:
            o = json.loads(raw)
            d = o.get("data", "")
        except Exception:
            d = raw
        out.append(d)
    return out


def norm(s):
    s = s.strip()
    for rx in TIME_RX:
        s = rx.sub("<T>", s)
    s = re.sub(r"\s+", " ", s)
    return s


la, lb = [norm(x) for x in lines(A)], [norm(x) for x in lines(B)]
print(f"stdout lines: parent={len(la)}  this={len(lb)}")

sa, sb = set(la), set(lb)
only_a = [x for x in la if x not in sb]
only_b = [x for x in lb if x not in sa]

print(f"\nlines present in ONE run only: parent={len(only_a)}  this={len(only_b)}")
print("\n--- lines ONLY in the parent (cfmsft2) ---")
for x in only_a[:25]:
    print("   ", x[:200])
print("\n--- lines ONLY in this arm (cfmsft2m) ---")
for x in only_b[:25]:
    print("   ", x[:200])

print("\n--- everything the log says about the Rad exclusion ---")
for label, ls in (("parent", la), ("this", lb)):
    hits = [x for x in ls if re.search(r"exclude|EXCLUDE|rad_alpha|Rad-dual5|rad_e1", x)]
    print(f"  {label}: {len(hits)} line(s)")
    for x in hits[:8]:
        print("      ", x[:190])

print()
if only_a or only_b:
    print("VERDICT: the two runs differ in CONTENT (timing already stripped).")
else:
    print("VERDICT: after removing timings the two runs are line-for-line IDENTICAL.")
    print("         => this log cannot show that the MCL edit reached the graph, and it also")
    print("            cannot show that it did not. The arm is UNVERIFIED, not refuted.")
