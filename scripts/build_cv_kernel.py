# -*- coding: utf-8 -*-
"""Build the CV-fold inference/training kernels.

WHY A FILE AND NOT AN INLINE COMMAND
    Three separate failures in this project came from building Python source inside a
    PowerShell here-string: the shell ate quote characters and produced either a
    syntax error in the *generated* file (HANDOFF 74.4 #7) or in the builder itself.
    The recorded rule is "write a script file instead".  This is that file.

WHAT IT BUILDS
    `kernel-raptor-train/` is the author's training script plus a prelude that converts
    the label CSV to parquet and injects sys.argv, because a Kaggle *script* kernel
    receives no command-line arguments.  The script already contains a CV fold hook:

        if a.folds > 0:
            fmap = json.load(open(a.fold_file))["folds"]
            held = set(u for u in train_ids if fmap.get(u, -1) == a.fold)
            oof_ids  = [u for u in train_ids if u in held]
            train_ids = [u for u in train_ids if u not in held]

    so `--folds N --fold K --fold_file F` trains on the rest and OOF-predicts fold K.
    This builder produces one kernel per fold, each with the fold flags appended to the
    argv list in the prelude.

Usage:
    python scripts/build_cv_kernel.py --folds 5            # build all five
    python scripts/build_cv_kernel.py --folds 5 --only 0   # just fold 0
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kernel-raptor-train"
FOLD_DATASET = "ppxl16/rsna-knee-cv-folds"
FOLD_FILE_IN_KERNEL = "/kaggle/input/rsna-knee-cv-folds/fold_file.json"


def build(fold: int, folds: int, tag: str, base: pathlib.Path, suffix: str,
          epochs: int = 12, sets: list | None = None,
          kernel_name: str | None = None, edits: list | None = None) -> pathlib.Path:
    code = (base / "knee-mri-training-the-twelve-finding-model.py").read_text(encoding="utf-8")
    meta = json.loads((base / "kernel-metadata.json").read_text(encoding="utf-8"))

    # Source-level edits first (they may touch anything, including the prelude the anchors
    # below rely on).  Each edit is (OLD, NEW) and must match EXACTLY ONCE -- a silent
    # zero-match would produce a "no-op experiment" that reads as "the change did nothing",
    # which is the single most expensive failure mode in this project (HANDOFF 16.1).
    for old, new in (edits or []):
        n = code.count(old)
        if n != 1:
            raise SystemExit("edit %r matches %d times (need exactly 1)" % (old[:60], n))
        code = code.replace(old, new, 1)

    anchor = "'--seed', '42', '--tag', 'v1']"       # the tag in the BASE kernel, not the new one
    if anchor not in code:
        raise SystemExit("argv anchor %r not found - has the base prelude changed?" % anchor)

    # Resolve the fold map by GLOB, exactly the way the prelude already resolves the label
    # CSV.  A dataset does NOT mount at /kaggle/input/<slug>/ - it mounts at
    # /kaggle/input/datasets/<owner>/<slug>/, so a hardcoded path fails with
    # FileNotFoundError (that cost one full fold run on 2026-09-14).  The working label
    # lookup in the same prelude was already using glob; the fold path should have too.
    resolve = ("_fold_cands = glob.glob('/kaggle/input/**/fold_file.json', recursive=True)\n"
               "if not _fold_cands:\n"
               "    raise SystemExit('fold_file.json not found under /kaggle/input')\n"
               "_FOLD_FILE = _fold_cands[0]\n"
               "print('[prelude] fold_file -> %s' % _FOLD_FILE, flush=True)\n")
    if "_FOLD_FILE = _fold_cands[0]" not in code:
        imp = "import pandas as pd\n"
        if imp not in code:
            raise SystemExit("prelude import anchor not found")
        code = code.replace(imp, imp + resolve, 1)

    extra = ("'--seed', '42', '--tag', '%s', '--folds', '%d', '--fold', '%d', "
             "'--fold_file', _FOLD_FILE" % (tag, folds, fold))
    if sets:
        # Appended AFTER the fold flags on purpose: argparse keeps the LAST occurrence of a
        # flag, so this overrides whatever the base prelude already set.  Quote values as
        # strings - the prelude's argv is source text (HANDOFF 74.4 #7).
        extra += "".join(", '%s', '%s'" % (k, v) for k, v in sets)
    extra += "]"
    code = code.replace(anchor, extra, 1)

    if epochs != 12:
        # The prelude carries the base run's hyper-parameters; rewrite the epoch count so a
        # deliberately under-trained arm can act as a LARGE-EFFECT control (HANDOFF 79.3).
        old_ep = "'--res', '224', '--epochs', '12'"
        if old_ep not in code:
            raise SystemExit("epoch anchor %r not found" % old_ep)
        code = code.replace(old_ep, "'--res', '224', '--epochs', '%d'" % epochs, 1)

    ast.parse(code)                      # fail locally, not after a 5-minute Kaggle round-trip

    name = kernel_name or ("kernel-cv-%s-fold%d" % (suffix, fold))
    dst = ROOT / name
    dst.mkdir(exist_ok=True)
    (dst / "knee-mri-training-the-twelve-finding-model.py").write_text(code, encoding="utf-8")

    meta["id"] = "ppxl16/%s" % name.replace("kernel-", "rsna-knee-", 1)
    meta["title"] = name.replace("kernel-", "RSNA knee ", 1).replace("-", " ")
    meta["dataset_sources"] = list(dict.fromkeys(meta["dataset_sources"] + [FOLD_DATASET]))
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dst


BASES = {
    # tag suffix   base kernel dir              what varies
    "A": ("kernel-raptor-train", "public v4hybrid labels"),
    "B": ("kernel-raptor-train-rule", "our rule-parser labels"),
    # G = the LARGE-EFFECT control: same everything, but 1 epoch instead of 12.
    #     Its gold-58 AUC is 0.7272 vs 0.8440 for 12 epochs (HANDOFF 74.1 log) - a 0.117
    #     gap, about 24x this CV's resolution, and it costs ~14 min per fold instead of ~3 h.
    #     Gate 1 must be a CLEAN verdict, so this is the contrast to use (HANDOFF 79.3).
    "G": ("kernel-raptor-train", "1-epoch large-effect control"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--arm", default="A", choices=sorted(BASES),
                    help="A = v4hybrid labels, B = rule labels, G = 1-epoch control")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override; defaults to 12 for A/B and 1 for G")
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--variant", default=None,
                    help="experiment name; kernel becomes kernel-cv-<variant>-fold<K> and the "
                         "run tag becomes <variant><K> (e.g. --variant norm -> cva0-like tag norm0)")
    ap.add_argument("--set", action="append", default=[], metavar="FLAG=VALUE",
                    help="extra training argv appended last, e.g. --set norm=imagenet "
                         "--set res=288.  Repeatable.")
    ap.add_argument("--edit", action="append", default=[], metavar="OLD||NEW",
                    help="literal SOURCE edit, must match exactly once. Repeatable. "
                         "Use for changes that are not CLI flags, e.g. the 2.5D window "
                         "spacing: --edit \"vol[c - 1], vol[c], vol[c + 1]||vol[c - 2], vol[c], vol[c + 2]\"")
    args = ap.parse_args()

    sets = []
    for s in args.set:
        if "=" not in s:
            raise SystemExit("--set needs FLAG=VALUE, got %r" % s)
        k, v = s.split("=", 1)
        sets.append((k if k.startswith("--") else "--" + k, v))
    if sets:
        print("extra argv: %s" % " ".join("%s %s" % kv for kv in sets))

    edits = []
    for e in args.edit:
        if "||" not in e:
            raise SystemExit("--edit needs OLD||NEW, got %r" % e)
        old, new = e.split("||", 1)
        edits.append((old, new))
        print("edit: %r -> %r" % (old[:70], new[:70]))

    base_name, blurb = BASES[args.arm]
    base = ROOT / base_name
    if not base.is_dir():
        raise SystemExit("base kernel %s not found" % base)
    epochs = args.epochs if args.epochs is not None else (1 if args.arm == "G" else 12)
    print("base %-28s (%s)  epochs=%d" % (base_name, blurb, epochs))

    folds = [args.only] if args.only is not None else list(range(args.folds))
    for f in folds:
        if args.variant:
            suffix, tag, kname = args.variant, "%s%d" % (args.variant, f), \
                "kernel-cv-%s-fold%d" % (args.variant, f)
        else:
            suffix, tag, kname = args.arm.lower(), "cv%s%d" % (args.arm.lower(), f), None
        dst = build(f, args.folds, tag, base, suffix, epochs=epochs, sets=sets,
                    kernel_name=kname, edits=edits)
        code = (dst / "knee-mri-training-the-twelve-finding-model.py").read_text(encoding="utf-8")
        i = code.find("'--epochs'")
        j = code.find("'--tag'")
        print("%-24s built  | %s | %s"
              % (dst.name, code[i:i + 26].split("]")[0], code[j:j + 60].split("]")[0] + "]"))
    print("\nast.parse already passed for every kernel above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
