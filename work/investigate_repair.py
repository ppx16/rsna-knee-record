"""Two questions before treating Repair-v1 as a lever.

1. `qierroo/rsna-knee-public-blend-drop-repair-v1` carries an extra token `drop-repairv1`. If someone
   measured that DROPPING the member helps, then our stack (which has no Repair-v1 at all) is already
   in the dropped state and ADDING it would be a regression -- the opposite of a lever. This is the
   question that decides the sign, so it is asked before anything is built.

2. Which dataset actually supplies the Repair-v1 checkpoint? A member with no published checkpoint is
   not wireable, however real it looks in the code (the `F_dim=768` lesson: declarations are cheap).
"""
import json
import pathlib
import re

ROOTS = [pathlib.Path("work/newpub_1006"), pathlib.Path("work/above")]


def text_of(p):
    nb = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                      else (c.get("source") or "")) for c in nb["cells"])


# ---- 1. what does the drop-repair notebook DO? -------------------------------------------------
dp = list(pathlib.Path("work/newpub_1006/rsna-knee-public-blend-drop-repair-v1").rglob("*.ipynb"))
if dp:
    t = text_of(dp[0])
    print("=== drop-repair notebook: every line mentioning drop/drop-repair ===")
    for ln in t.splitlines():
        if re.search(r"drop", ln, re.I):
            print("   " + ln.strip()[:190])
    print()
    print("=== its CoAt member list construction ===")
    for m in re.finditer(r"repairv1", t):
        a = max(0, m.start() - 220); b = min(len(t), m.end() + 260)
        seg = t[a:b].replace("\n", " | ")
        print("   ..." + seg[:470])
        print()
        break
    print("=== does it still RUN the reader (vs short-circuit it)? ===")
    for probe in ("_run_repairv1_reader", "use_repairv1", "REPAIR", "repairv1 =", "skip"):
        print("   %-22s occurrences: %d" % (probe, len(re.findall(re.escape(probe), t))))

# ---- 2. which dataset supplies the checkpoint? -------------------------------------------------
print()
print("=" * 92)
print("datasets referenced near a repairv1 / coatnet manifest mention")
for root in ROOTS:
    for p in root.rglob("*.ipynb"):
        try:
            t = text_of(p)
        except Exception:
            continue
        if "repairv1" not in t:
            continue
        ds = sorted(set(re.findall(r"/kaggle/input/(?:datasets/)?[\w\-]+/[\w\-]+", t)))
        coat_ds = [d for d in ds if re.search(r"coat|repair|d4|global|resgat|pjmath|mattia", d, re.I)]
        print("-" * 88)
        print(str(p)[:110])
        for d in coat_ds:
            print("     ", d)
        break
    else:
        continue
    break
