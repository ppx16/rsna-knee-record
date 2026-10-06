# -*- coding: utf-8 -*-
"""Point the inference kernel at a weights package and a DINOv2 variant - safely.

The inference kernel needs two things set correctly, and both have a wrong version that
fails quietly rather than loudly:

  1. EXACTLY ONE source of weights. `find_weights()` in the notebook walks /kaggle/input
     and returns the first directory holding a usable manifest; it has no ambiguity check.
     Two packages attached therefore leaves the choice to directory order, and the run can
     score the previous ensemble while every log line looks correct - the stale package is
     self-consistent, so even the fingerprint check passes.

     There are two ways to supply the one package:

       --kernel-source <ref>    mount another notebook's OUTPUT (no upload at all).
                                Verified 2026-09-11: a source kernel's output appears at
                                /kaggle/input/notebooks/<owner>/<slug>/, so the training
                                kernel's weights/manifest.json is readable directly. This
                                is the preferred path - publishing ~5.6 GB as a dataset
                                goes through the Clash proxy and is very slow.

       --weights-dataset <ref>  attach a dataset built by package_weights.py. Kept as the
                                fallback; proven on 2026-09-06.

  2. model_sources must carry the DINOv2 checkpoint matching the members' own
     `config.variant`. `infer_from_package` builds each member with `variant=...` from the
     manifest, and `find_dinov2(variant)` falls back to `hits[0]` - the first checkpoint
     it found, whatever it is - when no path contains the variant name. A large-variant
     member with only the small checkpoint attached therefore loads the wrong backbone and
     dies in load_state_dict with a shape mismatch.

    # inspect (default: nothing is written)
    python scripts/prep_inference_kernel.py --kernel-source ppxl16/rsna-knee-train-v1 \\
        --variant large

    # write it
    python scripts/prep_inference_kernel.py --kernel-source ppxl16/rsna-knee-train-v1 \\
        --variant large --apply

On the check this script deliberately does NOT have: there is no "is a second weights
package still attached?" test, because every name-matching entry is removed before the new
one is added, so such a check could never fire - and a guard that cannot fail reads like
protection while providing none. What is real is the refusal below (when a requested
dataset's name gives the pruning nothing to work with) and the assertion that exactly one
weights source remains. Kept datasets and kept kernel sources are printed so a human can
confirm none of them is a package under an unrecognisable name.
"""
from __future__ import annotations


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
import argparse
import json
import shutil
import sys
from pathlib import Path
from kpaths import kdir          # kernels/ layout, see kpaths.py

DEFAULT_META = kdir("kernel-fork") / "kernel-metadata.json"


def looks_like_weights(ref: str) -> bool:
    return "weight" in ref.lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=str(DEFAULT_META))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--kernel-source",
                     help="kernel ref whose OUTPUT the inference kernel should mount")
    src.add_argument("--weights-dataset",
                     help="dataset ref built by package_weights.py (fallback path)")
    ap.add_argument("--variant", default="large",
                    choices=["small", "large", "base", "giant"],
                    help="DINOv2 variant the members were trained on (manifest config)")
    ap.add_argument("--apply", action="store_true",
                    help="write the file; without it this only prints the diff")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    if not meta_path.is_file():
        raise SystemExit(f"no such metadata file: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    using_kernel = args.kernel_source is not None
    if not using_kernel and not looks_like_weights(args.weights_dataset):
        raise SystemExit(
            f"--weights-dataset {args.weights_dataset!r} has no 'weight' in its name, so "
            f"this tool cannot tell it apart from an ordinary dataset and cannot prove it "
            f"removed every previous weights package. Rename the dataset, or edit "
            f"kernel-metadata.json by hand and check dataset_sources yourself.")

    # (1) dataset_sources: keep the non-weights entries, drop every weights-looking one,
    #     and add the requested dataset only on the dataset path.
    ds = list(meta.get("dataset_sources", []))
    ds_kept = [d for d in ds if not looks_like_weights(d)]
    ds_dropped = [d for d in ds if looks_like_weights(d)]
    new_ds = list(ds_kept) + ([] if using_kernel else [args.weights_dataset])

    # (2) kernel_sources: exactly the requested one, or none on the dataset path.
    ks = list(meta.get("kernel_sources", []))
    ks_dropped = [k for k in ks if k != args.kernel_source]
    new_ks = [args.kernel_source] if using_kernel else []

    # (3) The invariant. Exactly one weights source overall.
    n_sources = (1 if using_kernel else 0) + len([d for d in new_ds
                                                  if looks_like_weights(d)])
    if n_sources != 1:
        raise SystemExit(f"internal error: {n_sources} weights sources would be attached; "
                         f"exactly one is required")
    if new_ds.count(args.weights_dataset or "") > 1:
        raise SystemExit("internal error: the requested dataset appears more than once")

    # (4) model_sources: one dinov2 entry, for the requested variant.
    ms = list(meta.get("model_sources", []))
    new_models: list[str] = []
    replaced = False
    for m in ms:
        if "dinov2" in m.lower():
            if not replaced:
                new_models.append(f"metaresearch/dinov2/PyTorch/{args.variant}/1")
                replaced = True
            # further dinov2 entries are dropped: only one backbone is ever needed, and a
            # second one only adds paths for find_dinov2 to fall back onto
        else:
            new_models.append(m)
    if not replaced:
        new_models.append(f"metaresearch/dinov2/PyTorch/{args.variant}/1")

    print(f"meta      : {meta_path}")
    print(f"kernel id : {meta.get('id')}")

    print("\n  dataset_sources")
    for d in ds:
        print(f"    - {d}" + ("   (weights, removed)" if looks_like_weights(d) else ""))
    print("  ->")
    for d in new_ds or ["(none)"]:
        mark = "   <-- the weights dataset" if d == args.weights_dataset else ""
        print(f"    + {d}{mark}")
    if ds_dropped:
        print(f"  removed {len(ds_dropped)} weights dataset(s): {ds_dropped}")
    if ds_kept:
        print(f"  kept {len(ds_kept)} non-weights dataset(s); confirm none of them is a "
              f"weights package under an unrecognisable name: {ds_kept}")

    print("\n  kernel_sources")
    for k in ks or ["(none)"]:
        print(f"    - {k}")
    print("  ->")
    for k in new_ks or ["(none)"]:
        mark = "   <-- mounts the weights via notebook output (NO upload)" \
            if k == args.kernel_source else ""
        print(f"    + {k}{mark}")
    if ks_dropped:
        print(f"  removed {len(ks_dropped)} other kernel source(s): {ks_dropped}")
    if new_ks:
        print("  NOTE: it serves the LATEST version's output, and the mount PATH SHAPE "
              "VARIES - measured 2026-09-11: rsna-knee-tpu-train-v1 appeared at "
              "/kaggle/input/notebooks/ppxl16/<slug>/ but rsna-knee-train-v1 appeared at "
              "/kaggle/input/<slug>/. Do not hardcode either; find_weights() walks "
              "/kaggle/input so it works either way.")
        print("  BEFORE spending GPU on an inference run, confirm with the CPU-only probe "
              "kernel (kernel-probe/, push it with this kernel in kernel_sources): if "
              "find_weights() returns None the notebook TRAINS FROM SCRATCH for ~8h and "
              "ends on a 0.5 submission.")

    print("\n  model_sources")
    for m in ms or ["(none)"]:
        print(f"    - {m}")
    print("  ->")
    for m in new_models or ["(none)"]:
        print(f"    + {m}")

    unchanged = (new_ds == ds and new_ks == ks and new_models == ms)
    if unchanged:
        print("\n  (already in this state - nothing to do)")

    if not args.apply:
        print("\nDRY RUN - nothing written. Re-run with --apply to update the file.")
        return 0

    backup = meta_path.with_suffix(".json.bak")
    shutil.copy(meta_path, backup)
    meta["dataset_sources"] = new_ds
    meta["kernel_sources"] = new_ks
    meta["model_sources"] = new_models
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten. backup at {backup}")
    print("Next: kaggle kernels push -p kernels/kernel-fork   (this saves AND runs; wait for "
          "COMPLETE before competitions submit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
