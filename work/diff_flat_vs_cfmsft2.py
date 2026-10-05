"""Prove the claim: kernel-0943cfmsft2flat differs from kernel-0943cfmsft2 on EXACTLY ONE axis.

That claim is the whole reason the arm is worth a submission -- it is what makes the final pair a
clean single-axis hedge. So it gets checked against the artifacts, not asserted in a docstring.

A "difference" here is reported as the set of changed LINES, and each changed line is classified.
Anything that is not the routing preset is a surprise and must be explained.
"""
import difflib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = "rsna-knee-restructured-version-3.ipynb"
A = ROOT / "kernels" / "kernel-0943cfmsft2" / NB
B = ROOT / "kernels" / "kernel-0943cfmsft2flat" / NB


def cells(p):
    nb = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    return ["".join(c["source"]) if "source" in c else "" for c in nb["cells"]]


a, b = cells(A), cells(B)
print("cells: cfmsft2=%d  cfmsft2flat=%d" % (len(a), len(b)))
if len(a) != len(b):
    raise SystemExit("!! cell count differs -- these are not the same notebook")

changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
print("cells that differ: %s" % changed)
print()

total_changed_lines = 0
unexplained = []
for i in changed:
    la, lb = a[i].splitlines(), b[i].splitlines()
    d = [ln for ln in difflib.unified_diff(la, lb, lineterm="", n=0)
         if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]
    total_changed_lines += len(d)
    print("--- cell %d: %d changed line(s) ---" % (i, len(d)))
    for ln in d:
        print("   ", ln[:150])
        body = ln[1:].strip()
        if body and "RSNA_PRESET" not in body:
            unexplained.append((i, body))

print()
print("total changed lines: %d" % total_changed_lines)
print("changes NOT mentioning RSNA_PRESET: %d" % len(unexplained))
for i, body in unexplained:
    print("   !! cell %d: %s" % (i, body[:130]))
print()
if total_changed_lines == len(unexplained) + 0 and not unexplained:
    print("VERDICT: the ONLY difference is the routing preset line. Claim holds.")
elif not unexplained:
    print("VERDICT: every changed line mentions RSNA_PRESET. Claim holds.")
else:
    print("VERDICT: UNEXPLAINED differences present -- claim does NOT hold as stated.")

# and confirm what the preset now resolves to, in both artifacts
for name, cs in (("cfmsft2", a), ("cfmsft2flat", b)):
    hit = [ln.strip() for c in cs for ln in c.splitlines() if "RSNA_PRESET" in ln and "os.environ" in ln]
    print("  %-11s %s" % (name, hit))
