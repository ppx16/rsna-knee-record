"""Locate the two constants the first pass could not extract:
  * `_RAD_CAL` -- recorded in this repo as an "embedded base64/zlib artifact of unknown
    provenance". The reference bundle documents a calibration table with SEVEN GATED targets,
    so confirming our gate list closes that open question against an independent source.
  * the residual-CoatNet hybrid weight (reference: public 0.6 / residual 0.4).
"""
import json
import pathlib
import re

NB = pathlib.Path(
    r"D:\kaggle\rsna-knee\kernels\kernel-0943cfmsft2m\rsna-knee-restructured-version-3.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
T = "\n".join(
    "".join(c.get("source", "")) if isinstance(c.get("source"), list) else (c.get("source") or "")
    for c in nb["cells"])

for pat, label in [
    (r"_RAD_CAL", "_RAD_CAL"),
    (r"rad_cal|RAD_CAL|calibration", "calibration"),
    (r"residual_coat|residual", "residual_coat"),
    (r"0\.40|0\.4\b", "0.4 (hybrid residual?)"),
    (r"gate", "gate"),
]:
    hits = list(re.finditer(pat, T, re.I))
    print("=" * 78)
    print(f"{label}: {len(hits)} hit(s)")
    print("=" * 78)
    for m in hits[:4]:
        a, b = max(0, m.start() - 260), min(len(T), m.end() + 300)
        print("   ...", T[a:b].replace("\n", " | ")[:520])
        print()
