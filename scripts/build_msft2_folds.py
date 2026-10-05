"""Build `kernels/kernel-msft2-f<fold>` — one fold of the FIVE-FOLD msft2 recipe.

WHY. `msft2` is the only thing in this repository with a positive measurement: its SWA checkpoint moved
our rank 708 -> 487 (in-block 38.89 % -> 13.66 %) as a 6th Raptor view. But the run that produced it was
**fold 0 of a five-fold design** -- `--folds 5 --fold 0` -- and folds 1..4 were never trained. This
builder produces them.

WHY IT REUSES build_msft2_kernel.py RATHER THAN REPEATING IT. That module's four patches are the whole
reason msft2's numbers mean anything -- in particular carrying the 8 HEAD tensors out of the checkpoint,
without which the head is re-initialised at random and epoch 0 scores below the parent (the 10.7 GPU-hours
wasted on 2026-09-28). Copying them here would create a second copy that can silently drift; importing
them means `--check` fails loudly if upstream anchors ever move.

⚠️ DEVICE: TPU, per `FACTS` A31 ("GPU 留给提交用，实验走 TPU"). msft2 fold 0 itself ran on TPU
(`kernel-msft2` metadata: enable_tpu=True, enable_gpu=False).
⚠️ CONCURRENCY: TPU allows ONE session and a QUEUED kernel already holds the slot (A28/A136), so these
must be pushed SERIALLY -- one at a time, waiting for each to finish.

Usage:
    python scripts/build_msft2_folds.py --fold 1 --check
    python scripts/build_msft2_folds.py --fold 1 --build
    python scripts/build_msft2_folds.py --fold 1 --build --push
"""
from __future__ import annotations

import argparse
import json
import pathlib
import py_compile
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_msft2_kernel as m2  # noqa: E402  (validated msft2 patches)

SRC = m2.SRC
CODE = m2.CODE

# The exact prelude line that fixes the fold and the tag.
PRELUDE_FOLD_LINE = "            '--seed', '42', '--tag', 'msft0', '--folds', '5', '--fold', '0',\n"

# Fold 0 already exists as `raptor_ft_msft0_swa.pt` inside `ppxl16/rsna-knee-msft2-ckpt`, so re-running
# it would burn ~2.6 TPU-hours to reproduce an artifact we have. Folds 1..4 are what is missing.
ALLOWED = (1, 2, 3, 4)


def build_text(fold: int) -> str:
    src = (SRC / CODE).read_text(encoding="utf-8")
    text = m2.patch(src, "finetune")          # the four validated msft2 patches
    new_line = "            '--seed', '42', '--tag', '%s', '--folds', '5', '--fold', '%d',\n" % (
        tag_for(fold), fold)
    if text.count(PRELUDE_FOLD_LINE) != 1:
        raise SystemExit("ANCHOR FAILURE [prelude-fold-line]: expected exactly 1 match, found %d"
                         % text.count(PRELUDE_FOLD_LINE))
    text = text.replace(PRELUDE_FOLD_LINE, new_line, 1)
    return text


def tag_for(fold: int) -> str:
    return "msft0" if fold == 0 else "msftF%d" % fold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    if a.fold not in ALLOWED:
        raise SystemExit("fold must be one of %s (0 already exists as raptor_ft_msft0_swa.pt)"
                         % (ALLOWED,))

    dst = ROOT / "kernels" / ("kernel-msft2-f%d" % a.fold)
    kid = "ppxl16/rsna-knee-msft2-f%d" % a.fold
    text = build_text(a.fold)

    target = dst / CODE if a.build else ROOT / "work" / ("_msft2_f%d_preview.py" % a.fold)
    if a.build:
        dst.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    py_compile.compile(str(target), doraise=True)

    tag = tag_for(a.fold)
    probes = [
        ("'--tag', '%s', '--folds', '5', '--fold', '%d'," % (tag, a.fold), "the fold/tag argv"),
        ("_HEAD_STATE", "msft2's head-carrying fix"),
        ("[zeroshot] VERDICT", "msft2's zero-shot gate"),
        ("--zero_shot_only", "the gate flag"),
        ("shape_mismatch", "the head shape check"),
        ("head tensors carried", "the head inventory print"),
    ]
    missing = [why for probe, why in probes if probe not in text]
    if missing:
        raise SystemExit("!! patched payload is missing: %s" % missing)
    if "'msft0'" in text:
        print("   note: the literal 'msft0' still appears somewhere (expected only if reused text)")
    print("PATTERN CHECK OK (fold %d, tag %s): compiled, all %d probes present"
          % (a.fold, tag, len(probes)))

    if not a.build:
        print("(--check only: wrote %s)" % target.name)
        return 0

    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = kid
    meta["title"] = "RSNA knee msft2 f%d" % a.fold
    meta["code_file"] = CODE
    meta["enable_tpu"] = True          # A31: experiments on TPU
    meta["enable_gpu"] = False
    meta.pop("machine_shape", None)
    meta["is_private"] = True
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("built %s  (enable_tpu=%s enable_gpu=%s, %d datasets)"
          % (dst.relative_to(ROOT), meta["enable_tpu"], meta["enable_gpu"],
             len(meta.get("dataset_sources", []))))

    if a.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.kernels_push(str(dst))
        print("PUSHED %s" % kid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
