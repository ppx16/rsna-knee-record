"""Verify a built/pushed arm carries ALL of its changes, by reading the RESULT.

An edit-based check would pass even if a substitution silently produced the wrong text
(that is exactly the bug `build_0943_addmember.py` hit with an escaped apostrophe).
So this asserts on the assembled notebook.

Usage:  python verify_cfmsft2m.py [arm-dir]
        (default: the locally built arm; pass a pulled dir to verify what Kaggle actually has)
"""
import json
import pathlib
import sys

ARM = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(
    r"D:\kaggle\rsna-knee\kernels\kernel-0943cfmsft2m")
nbs = list(ARM.glob("*.ipynb"))
if len(nbs) != 1:
    print("!! expected exactly 1 notebook in %s, found %d" % (ARM, len(nbs)))
    raise SystemExit(2)
nb_path = nbs[0]
nb = json.loads(nb_path.read_text(encoding="utf-8"))
text = "\n".join(
    "".join(c.get("source", "")) if isinstance(c.get("source"), list) else (c.get("source") or "")
    for c in nb["cells"]
)
meta = json.loads((ARM / "kernel-metadata.json").read_text(encoding="utf-8"))

MCL = '_RAD_EXCLUDE = ("Baker\'s", "MCL")'
FRACTURE_BOTH = '_RAD_EXCLUDE = ("Baker\'s", \'Fracture\')'
BAKERS_ONLY = '_RAD_EXCLUDE = ("Baker\'s",)'

print("verifying: %s  (%s)" % (ARM, nb_path.name))
checks = [
    ("MCL exclusion (this arm's one change)", MCL in text),
    ("cn0 as view index 4", '"name": "cn0-ours"' in text),
    ("msft2 as view index 5", '"name": "msft2-ours"' in text),
    ("gpu_zero runs index 4", "run_single(4, torch.device('cuda:0'))" in text),
    ("gpu_zero runs index 5", "run_single(5, torch.device('cuda:0'))" in text),
    ("Fracture is NOT re-excluded (C81)", FRACTURE_BOTH not in text),
    ("the old Baker's-only line is gone", BAKERS_ONLY not in text),
    ("the four original view weights intact", all(
        k in text for k in ('"maxspan-v5": 0.60', '"native384dense-v10": 0.10',
                            '"maxspan-v5-reverse": 0.10', '"native384-v8": 0.20'))),
    ("the two added view weights intact", all(
        k in text for k in ('"cn0-ours": 0.20', '"msft2-ours": 0.2'))),
    ("cell count still 51", len(nb["cells"]) == 51),
    ("msft2 dataset mounted", "ppxl16/rsna-knee-msft2-ckpt" in (meta.get("dataset_sources") or [])),
    ("cn0 dataset mounted", "ppxl16/rsna-knee-cn0-weights" in (meta.get("dataset_sources") or [])),
]
for name, ok in checks:
    print("  %-44s %s" % (name, "YES" if ok else "*** NO ***"))

print()
print("  id       = %s" % meta.get("id"))
print("  title    = %s" % meta.get("title"))
print("  datasets = %d" % len(meta.get("dataset_sources") or []))
print("  gpu=%s internet=%s kernel_type=%s"
      % (meta.get("enable_gpu"), meta.get("enable_internet"), meta.get("kernel_type")))

bad = [n for n, ok in checks if not ok]
print()
print("VERDICT:", "ALL CHECKS PASS" if not bad else "FAILED: %s" % bad)
raise SystemExit(0 if not bad else 1)
