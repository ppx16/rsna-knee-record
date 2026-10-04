"""Are our calibration coefficients THE SAME as the public reference's?

Everything so far only shows the two tables have the same SHAPE and the same seven gated
targets. The reference ships `calibration.json` in the clear, and our notebook carries the
same object as base64+zlib, so the coefficients themselves can be compared one by one.

That converts "our _RAD_CAL has unknown provenance" into either
  * identical  -> it IS the public table, to the last float; or
  * different  -> a real divergence worth investigating.
"""
import base64
import json
import pathlib
import re
import zlib

NB = pathlib.Path(
    r"D:\kaggle\rsna-knee\kernels\kernel-0943cfmsft2m\rsna-knee-restructured-version-3.ipynb")
REF = pathlib.Path(r"D:\kaggle\rsna-knee\work\above\tony\x\calibration.json")

T = "\n".join(
    "".join(c.get("source", "")) if isinstance(c.get("source"), list) else (c.get("source") or "")
    for c in json.loads(NB.read_text(encoding="utf-8"))["cells"])
ours = json.loads(zlib.decompress(base64.b64decode(
    re.search(r"_RAD_CAL_PAYLOAD\s*=\s*'([A-Za-z0-9+/=]+)'", T).group(1))).decode())
ref = json.loads(REF.read_text(encoding="utf-8"))

print(f"ours keys: {sorted(ours.keys())}")
print(f"ref  keys: {sorted(ref.keys())}")
print()

def flat(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, f"{p}{k}."))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(flat(v, f"{p}{i}."))
    else:
        out[p[:-1]] = o
    return out

a, b = flat(ours), flat(ref)
print(f"scalars: ours={len(a)}  ref={len(b)}   common={len(set(a) & set(b))}")

diffs, worst = [], 0.0
for k in sorted(set(a) & set(b)):
    va, vb = a[k], b[k]
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        d = abs(float(va) - float(vb))
        worst = max(worst, d)
        if d != 0:
            diffs.append((k, va, vb, d))
    elif va != vb:
        diffs.append((k, va, vb, float("nan")))

print(f"EXACT-equal scalars: {len(set(a) & set(b)) - len(diffs)}")
print(f"differing scalars  : {len(diffs)}")
print(f"max |difference|   : {worst:.3e}")

if diffs:
    print("\nfirst 15 differences:")
    for k, va, vb, d in diffs[:15]:
        print(f"  {k}\n     ours={va}\n     ref ={vb}\n     |d| ={d}")
else:
    print("\n*** every shared scalar is EXACTLY equal ***")

print()
if not diffs and set(a) == set(b):
    print("VERDICT: our embedded table IS the public reference's calibration table,")
    print("         coefficient for coefficient. The recorded 'unknown provenance' is closed.")
elif not diffs:
    print("VERDICT: all shared scalars identical, but the key sets differ - check the extras.")
    print("  only ours:", sorted(set(a) - set(b))[:20])
    print("  only ref :", sorted(set(b) - set(a))[:20])
else:
    print("VERDICT: the tables DIFFER. Either a different version or a divergence;")
    print("         this is a real finding and must not be glossed.")
