"""Decode our embedded `_RAD_CAL_PAYLOAD` and identify it against the public reference.

This repo recorded `_RAD_CAL` as "an embedded base64/zlib artifact carrying ... unknown
provenance" and wrote "Do not touch it." The reference bundle
(`tonylica/rsna-knee-bend-dinov3-0917-repro-assets`, a SHA-verified repro of
evgendvorkin/rsna-baseline v15) documents a "frozen 12-output, 88-input linear calibration
table; seven gated targets".

Our notebook carries the artifact in the clear (base64+zlib), so the claim can be CHECKED
instead of believed. This decodes it and compares.
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
import base64
import json
import pathlib
import re
import zlib

NB = pathlib.Path(
    r"D:\kaggle\rsna-knee\kernels\kernel-0943cfmsft2m\rsna-knee-restructured-version-3.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
T = "\n".join(
    "".join(c.get("source", "")) if isinstance(c.get("source"), list) else (c.get("source") or "")
    for c in nb["cells"])

m = re.search(r"_RAD_CAL_PAYLOAD\s*=\s*'([A-Za-z0-9+/=]+)'", T)
if not m:
    print("!! _RAD_CAL_PAYLOAD not found")
    raise SystemExit(1)
payload = m.group(1)
print(f"_RAD_CAL_PAYLOAD: {len(payload)} base64 chars")

raw = zlib.decompress(base64.b64decode(payload))
cal = json.loads(raw.decode())

print(f"\ndecoded: {len(raw)} bytes of JSON, top-level keys = {list(cal.keys())}")

gate = cal.get("gate")
print(f"\ngate ({len(gate) if gate else 0} targets): {gate}")

print("\n--- shapes / structure ---")
for k, v in cal.items():
    if isinstance(v, list):
        inner = v[0] if v and isinstance(v[0], list) else None
        print(f"  {k:<18} list of {len(v)}" + (f" x {len(inner)}" if inner else ""))
    elif isinstance(v, dict):
        print(f"  {k:<18} dict with keys {list(v.keys())[:12]}")
    else:
        print(f"  {k:<18} {type(v).__name__} = {str(v)[:80]}")

print()
print("=" * 78)
print("AGAINST THE REFERENCE")
print("=" * 78)
print("  reference says : 'one frozen 12-output, 88-input linear calibration table;")
print("                    seven gated targets'")
n_out = len(cal.get("mean") or [])
n_in = None
coef = cal.get("coef")
if isinstance(coef, list) and coef and isinstance(coef[0], list):
    n_out, n_in = len(coef), len(coef[0])
elif isinstance(coef, list):
    n_out, n_in = 1, len(coef)
print(f"  we decode      : coef {n_out} x {n_in}, gate has {len(gate) if gate else 0} targets")
seven = ["ACL", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Baker's", "Contusion"]
print(f"  seven gated?   : {sorted(gate) == sorted(seven) if gate else False}")
print(f"  our gate list  : {sorted(gate) if gate else None}")
print("=" * 78)
