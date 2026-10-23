"""MECHANICAL PREFLIGHT: check a kernel against every trap this repository has already paid for.

WHY THIS EXISTS. Twice in one night I lost real time to failures that were already written down in FACTS:
  * I mounted knee-raptor-corpus but not knee-raptor-corpus-ext (A56), so the held-out fold silently became
    642 studies instead of 870 while the log still looked healthy.
  * I edited a PULLED kernel-metadata.json in place, inheriting the original author's private BYOD docker
    image, and got a bodyless 403 that took a four-way bisection to explain.
The lesson this repository keeps relearning is that a rule living in prose gets re-broken; a rule living in a
check does not. So the checks live here and run BEFORE anything is built or pushed.

    python work/preflight_pitfalls.py kernels/kernel-tpu-3darm
    python work/preflight_pitfalls.py --all

Exit code is 1 if any BLOCKER fires. Warnings do not fail the run but are printed loudly.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
BLOCKERS, WARNINGS = [], []


def bad(msg, why):
    BLOCKERS.append((msg, why))


def warn(msg, why):
    WARNINGS.append((msg, why))


def check(meta, code_text, name):
    ds = list(meta.get("dataset_sources") or [])
    ms = list(meta.get("model_sources") or [])
    kid = str(meta.get("id") or "")
    title = str(meta.get("title") or "")

    # --- A56: the corpus is TWO datasets. Mounting one silently shrinks the instrument. ------------------
    if any("knee-raptor-corpus" in d and not d.endswith("-ext") for d in ds):
        if not any(d.endswith("knee-raptor-corpus-ext") for d in ds):
            bad("A56 corpus is half-mounted",
                "knee-raptor-corpus is mounted without knee-raptor-corpus-ext; that is 3200 of 4407 studies, "
                "and any held-out fold silently shrinks (measured: 642 instead of 870)")

    # --- A168-3: machine_shape OVERRIDES the enable flags. ------------------------------------------------
    if meta.get("machine_shape") and meta.get("enable_gpu") and not meta.get("enable_tpu"):
        warn("machine_shape present with enable_gpu",
             "A168: a stale machine_shape (e.g. Tpu1VmV38) makes Kaggle count the kernel as a TPU session "
             "even with enable_tpu false. Delete machine_shape unless you really want that accelerator.")

    # --- title must slugify to the id, or SaveKernel returns 409 Conflict. --------------------------------
    if kid and title:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        want = kid.split("/")[-1]
        if slug != want:
            bad("title does not slugify to the id",
                "title %r slugs to %r but the id wants %r -> SaveKernel returns 409 Conflict" % (title, slug, want))

    # --- F51: never inherit a foreign private BYOD docker image. -----------------------------------------
    di = str(meta.get("docker_image") or "")
    if "kaggle-private-byod" in di:
        bad("foreign private BYOD docker image",
            "F51: referencing gcr.io/kaggle-private-byod/... (someone else's private image) returns a "
            "bodyless 403 from kernels_push. Write a FRESH metadata instead of editing a pulled one.")

    # --- GBK: a code_file with non-ASCII bytes cannot be pushed from this console. ------------------------
    cf = meta.get("code_file")
    if cf:
        p = ROOT / "kernels" / name / cf
        if p.is_file():
            nb = p.read_bytes()
            n = sum(1 for c in nb if c > 127)
            if n and cf.endswith(".ipynb"):
                warn("code_file has %d non-ASCII bytes" % n,
                     "kernels_push reads the code_file with the SYSTEM encoding (GBK here); write the "
                     "notebook with ensure_ascii=True (build_reader_arm.py does) or push with PYTHONUTF8=1")

    # --- soft labels: the table is probabilities, not YES/NO. --------------------------------------------
    if code_text and "report_labels" in code_text and "__verdict" in code_text and "float(raw)" not in code_text:
        bad("label parser may only accept YES/NO",
            "report_labels_* carry SOFT probabilities (0.25, 0.975) with no __verdict column; a YES/NO-only "
            "parser turns every cell into nan, the mask goes all-False and an empty tensor reaches the loss "
            "(measured: 'loss nan' on all 12 steps)")

    # --- an arm that evaluates must not be able to report a number it did not earn. ----------------------
    if code_text and ("macro_auc" in code_text or "macro-AUC" in code_text):
        if "n_val" not in code_text and "len(val)" not in code_text:
            warn("reports an AUC without reporting n",
                 "the gate is 'large-n instrument'; an AUC without its n cannot be judged against it")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    if a.all or not a.paths:
        dirs = [d for d in sorted((ROOT / "kernels").iterdir())
                if d.is_dir() and (d / "kernel-metadata.json").is_file()]
    else:
        dirs = [pathlib.Path(p) for p in a.paths]

    for d in dirs:
        mp = d / "kernel-metadata.json"
        if not mp.is_file():
            continue
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            bad("%s: unreadable metadata" % d.name, repr(e))
            continue
        code = ""
        cf = meta.get("code_file")
        if cf and (d / cf).is_file():
            code = (d / cf).read_text(encoding="utf-8", errors="replace")
        before = len(BLOCKERS), len(WARNINGS)
        check(meta, code, d.name)
        fired = (len(BLOCKERS) - before[0], len(WARNINGS) - before[1])
        print("%-44s blockers=%d warnings=%d" % (d.name, fired[0], fired[1]))

    print()
    for m, why in WARNINGS:
        print("WARN  %s\n      %s" % (m, why))
    for m, why in BLOCKERS:
        print("BLOCK %s\n      %s" % (m, why))
    print("\ntotal: %d blocker(s), %d warning(s)" % (len(BLOCKERS), len(WARNINGS)))
    return 1 if BLOCKERS else 0


if __name__ == "__main__":
    sys.exit(main())
