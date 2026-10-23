"""Local preflight of the public0033 bundle, using the runtime's OWN verifier.

`run_cached_inference` calls `verify_bundle(bundle_root)` first, which checks every payload
file against the SHA-256 recorded in `bundle_manifest.json`. That check is pure filesystem work,
so it can be run HERE instead of discovering a corrupt/mismatched bundle after a GPU run.

Also runs the author's `self_test`, which exercises the model/metric path on synthetic data.
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
import importlib.util
import pathlib
import sys
import traceback

BUNDLE = pathlib.Path(r"D:\kaggle\rsna-knee\work\meniscus\x")
RT = BUNDLE / "public0033_runtime.py"

print("bundle dir:", BUNDLE)
for p in sorted(BUNDLE.iterdir()):
    print(f"   {p.name:<46} {p.stat().st_size:>12,}")

spec = importlib.util.spec_from_file_location("public0033_runtime", RT)
rt = importlib.util.module_from_spec(spec)
sys.modules["public0033_runtime"] = rt
try:
    spec.loader.exec_module(rt)
    print("\nimport: OK")
except Exception as e:
    print("\nimport FAILED:", type(e).__name__, e)
    traceback.print_exc()
    raise SystemExit(1)

print("exact cache contract this runtime demands:")
print("   cache dtype/shape :", "uint8", "(N, 6, 12, 336, 336)")
print("   mask  dtype/shape :", "float32", "(N, 6)")
print("   device requirement:", "exactly 2 CUDA devices, both named T4")

print("\n--- verify_bundle (SHA-256 of every payload vs the manifest) ---")
try:
    m = rt.verify_bundle(BUNDLE)
    print("   PASS")
    for k in ("schema_version", "bundle_id", "status"):
        if k in m:
            print(f"   {k} = {m[k]!r}")
    if "checkpoint" in m:
        c = m["checkpoint"]
        print("   checkpoint:", {kk: c[kk] for kk in list(c)[:4]})
except Exception as e:
    print("   FAILED:", type(e).__name__, str(e)[:600])
    raise SystemExit(2)

print("\n--- self_test (synthetic, exercises the model + metric path) ---")
try:
    r = rt.self_test()
    print("   PASS")
    print("   ", {k: (str(v)[:90]) for k, v in list(r.items())[:12]})
except Exception as e:
    print("   FAILED:", type(e).__name__, str(e)[:600])
    traceback.print_exc()
