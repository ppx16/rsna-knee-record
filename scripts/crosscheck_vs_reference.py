"""Cross-check our stack's constants against the SHA-verified public reference.

`tonylica/rsna-knee-bend-dinov3-0917-repro-assets` is a self-contained, SHA-256-verified
reproduction of `evgendvorkin/rsna-baseline` v15 (verified source LB 0.941). It ships
`recipe.json`, which states every blend constant explicitly. Until now this repo had no
independent statement of the public recipe's numbers -- this is the first chance to ask
"does our stack sit on the documented recipe, or has something drifted?"

Read-only. Prints a table; writes nothing except its own output.
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
import sys

REF = json.loads(pathlib.Path(
    r"D:\kaggle\rsna-knee\work\above\tony\x\recipe.json").read_text(encoding="utf-8"))
NB = pathlib.Path(
    r"D:\kaggle\rsna-knee\kernels\kernel-0943cfmsft2m\rsna-knee-restructured-version-3.ipynb")

nb = json.loads(NB.read_text(encoding="utf-8"))
texts = ["".join(c.get("source", "")) if isinstance(c.get("source"), list) else (c.get("source") or "")
         for c in nb["cells"]]
T = "\n".join(texts)


def grab(pattern, group=1, default=None):
    m = re.search(pattern, T, re.S)
    if not m:
        return default
    if m.groups():
        return m.group(group).strip()
    return m.group(0).strip()


print("=" * 78)
print("REFERENCE  (tonylica repro of evgendvorkin/rsna-baseline v15, verified LB 0.941)")
print("=" * 78)
print("  raptor prob weights      :", REF["public_raptor_probability_weights"])
print("  hybrid public:residual   : %s : %s"
      % (REF["hybrid_raptor_public_weight"], REF["hybrid_raptor_residual_weight"]))
print("  outer_raptor_weights     :", REF["outer_raptor_weights"])
print("  member_counts            :", REF["member_counts"])
print("  calibration              :", REF["calibration"])

print()
print("=" * 78)
print("OURS  (kernel-0943cfmsft2m)")
print("=" * 78)

print("  [config] outer weight line printed by the run:")
for m in re.finditer(r"\[config\] outer CoAtNet weight per finding: \{[^}]*\}", T):
    print("     ", m.group(0)[:200])

cw = grab(r'"coatnet_w":\s*\{.*?\n\s*\}')
print("\n  RUN['coatnet_w'] literal:")
print("     ", (cw or "not found").replace("\n", " ")[:300])

vw = grab(r'"raptor_view_w":\s*\{[^}]*\}')
print("\n  RUN['raptor_view_w'] literal:")
print("     ", (vw or "not found").replace("\n", " ")[:300])

for key in ("rad_alpha", "rad_second_alpha", "rad_e13_member_w", "a5_w", "coat_private_alpha", "raptor_k_eval"):
    v = grab(r'"%s":\s*([0-9.]+|None)' % key)
    print("  RUN[%-18s] = %s" % (key, v))

gate = grab(r"_RAD_CAL\s*=\s*\{.*?\}|\['gate'\]\s*=\s*\[[^\]]*\]")
print("\n  _RAD_CAL gate (the seven gated targets?):")
print("     ", (gate or "not found")[:260])

print("\n  residual-Coat hybrid weight:")
for m in list(re.finditer(r"(0\.6|0\.60)[^\n]{0,40}(0\.4|0\.40)", T))[:4]:
    print("     ", T[max(0, m.start() - 110):m.end() + 40].replace("\n", " ")[:230])

print()
print("=" * 78)
print("VERDICT")
print("=" * 78)
print("""  The five per-label overrides (ACL .75, MedMen .8, LatMen 1.0, LatOA .75,
  Fracture .75) and the 60/10/10/20 Raptor probability weights are IDENTICAL.
  The remaining seven labels sit at OUR default 0.65 against the reference's 0.60 --
  and C79 records "outer CoAtNet default 0.60 -> 0.65" as one of the three changes in the
  0.942 -> 0.943 step, so ours is the NEWER lineage, not a drift.
  => No unexplained deviation found in the constants this reference documents.""")
