# -*- coding: utf-8 -*-
"""Build the A5-lite kernel payload: the frontier's model + our trainer + the pairing contract.

Same shape as scripts/build_a5_gate.py and for the same reason: a Kaggle *script* kernel uploads
exactly ONE `code_file`, so the model code cannot ride alongside as a second file (that mistake cost
one push already).  `a5_model.py` stays the versioned, re-verifiable source of truth; this inlines it.

WHAT THE KERNEL RUNS, AND WHY IT IS A PAIR
    `--head a5`     the frontier's Net: 6 slot types, slot tokens, `xcodex` pooling, slices as channels
    `--head raptor` our head on the same backbone, same data, same epochs, same optimizer
    Same fold file, same hybrid report labels, same pretrained init.  A single arm would prove
    nothing: `d3f0` already showed that this backbone with our head reaches only 0.853, so the
    question "is their architecture the lever?" is only answerable as a paired comparison.

⚠️ IT IS NOT THE FAITHFUL EXPERIMENT, AND SAYS SO.  A5's own input is 6 slots x 16 slices x 336 px
= 10.8 MB per study (~48 GB corpus), which is not published.  `n_slice` is a constructor argument,
so this trains the same architecture on the 44-slice corpus we already have, at n_slice=6, with the
sixth slot (Axial without fluid, which the corpus does not carry) masked.  What it tests is the HEAD
and the "slices as channels" representation; what it does not test is A5's 16-slice density.

Usage:
    python scripts/build_a5lite.py            # write kernels/kernel-a5lite/kernel_main.py
    python scripts/build_a5lite.py --check     # assert the payload matches its sources
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parent.parent
A5DIR = ROOT / "kernels" / "kernel-a5port"
LITEDIR = ROOT / "kernels" / "kernel-a5lite"
MODEL = A5DIR / "a5_model.py"
TRAIN = LITEDIR / "train_a5lite.py"

# ⚠️ Kaggle runs a script kernel with NO arguments, so argparse would exit(2) on the first required
# flag -- which is why every .py trainer in this repo carries an argv prelude (A49).  The guard means
# an explicit command line still wins.  ONE ARM PER KERNEL DIRECTORY, exactly like the 0943 arms.
ARMS = {
    # tags changed for the augmented pair (A153): a5fa / rfa, so the first pair's OOF (a5f0 / rf0)
    # stays on disk as the degraded-operating-point measurement it is.
    "a5": dict(dir="kernel-a5lite", kid="ppxl16/rsna-knee-a5lite", title="RSNA knee a5lite",
               argv=("--head a5 --fold 0 --folds 5 --epochs 12 --bs 2 --res 336 "
                     "--n-slice 6 --aug 1 --tag a5fa")),
    "raptor": dict(dir="kernel-a5lite-r", kid="ppxl16/rsna-knee-a5lite-r",
                   title="RSNA knee a5lite r",
                   argv=("--head raptor --fold 0 --folds 5 --epochs 12 --bs 2 --res 336 "
                         "--k 12 --aug 1 --tag rfa")),
    # ---- the slice-density ladder (built 2026-09-26, PUSH ONLY IF the augmented pair passes) ----
    # WHY: the corpus stores 44 slices in five groups of 12/10/8/6/8, so `n_slice` was capped at 6 by
    # the SMALLEST group (A152/209) and the frontier's 16-slice density was assumed to need the 48 GB
    # cache.  But `a5_study` windows INSIDE each group, so a rung at n_slice=8 takes 8 DISTINCT slices
    # from the 12/10/8/8 groups and 6+2 repeated from the coronal-no-fluid group -- i.e. only 1 of 5
    # groups is degenerate, against 0 at n_slice=6.  That is a real density increase on the corpus we
    # ALREADY have, and it costs ONE arm: the raptor control's input path never reads `n_slice`
    # (make_input, head=="raptor"), so `rfa` is a valid control for every rung.
    # What it decides: if the A5-vs-raptor gap grows with density, density is the lever and the 48 GB
    # cache is worth building; if the gap is flat in density, the gap is the HEAD and the 48 GB build
    # would buy nothing (209.3).
    # MEASURED distinct/total slices per group (12/10/8/6/8), so the rungs are not interchangeable:
    #   n_slice= 6 -> 6/6  6/6  6/6  6/6  6/6   <- zero padding: the honest ceiling of the corpus
    #   n_slice= 8 -> 8/8  8/8  8/8  6/8  8/8   <- ONE padded group  => a real density rung
    #   n_slice=10 -> 10/10 10/10 8/10 6/10 8/10
    #   n_slice=12 -> 12/12 10/12 8/12 6/12 8/12 <- mostly padding => measures CHANNEL CAPACITY,
    #                                               not density.  A gain here would NOT justify the
    #                                               cache, so only 8 is built.
    "a5n8": dict(dir="kernel-a5lite-n8", kid="ppxl16/rsna-knee-a5lite-n8",
                 title="RSNA knee a5lite n8",
                 argv=("--head a5 --fold 0 --folds 5 --epochs 12 --bs 2 --res 336 "
                       "--n-slice 8 --aug 1 --tag a5n8")),
}
PRELUDE = ('import sys as _sys\n'
           'if not any(_a.startswith("--") for _a in _sys.argv[1:]):\n'
           '    _sys.argv = ["train"] + "%s".split()\n'
           '    print("[prelude] no argv given -> using: %s", flush=True)\n')


def with_prelude(train_text, argv, model_text):
    """Split the trainer's trailing `main()` off, and put it back LAST -- after the model classes.

    ⚠️ Three failures produced this layout, all of them ordering:
      1. the argv prelude cannot go at the very top, because the trainer's
         `from __future__ import annotations` must precede every statement;
      2. it cannot go after the trainer body either, because that body ENDS with `main()`, so
         argparse ran first (`error: the following arguments are required: --head`);
      3. and `main()` cannot run before the model classes are defined -- inlining them after the
         trainer body produced `NameError: name 'ViTSlotToken' is not defined` at build_model.
    Hence: [trainer without its call] + [prelude] + [model] + [call], asserted in that order.
    (`--head raptor` never hit (3) because that branch never touches the A5 classes -- which is why
     the control arm trained happily while this arm kept dying.)
    """
    body = train_text.rstrip("\n")
    call = "main()"
    if not body.endswith(call):
        raise SystemExit("!! the trainer body does not end with %r -- refusing to patch" % call)
    body = body[: -len(call)].rstrip("\n")
    return (body + "\n\n" + PRELUDE % (argv, argv)
            + "\n\n# ---- the frontier's model, verbatim ----\n" + model_text
            + "\n\n" + call + "\n")


def payload(arm):
    """Trainer body FIRST, model body second.

    ⚠️ Order matters for exactly one reason: the trainer's first non-comment statement is
    `from __future__ import annotations`, which Python requires near the top of the file.  Inlining
    the model first pushed it to line 389 and both arms died with `SyntaxError: from __future__
    imports must occur at the beginning of the file`.  The fix is to swap the blocks, NOT to strip
    the line -- deleting lines from a verbatim body is the edit class that has already bitten this
    build twice.  Python resolves `Net`/`ViTSlotToken` when build_model is CALLED, so the model being
    defined lower in the file is fine.
    """
    return ("# GENERATED by scripts/build_a5lite.py -- do not edit\n"
            "#   trainer      : kernels/kernel-a5lite/train_a5lite.py\n"
            "#   model classes: kernels/kernel-a5port/a5_model.py (verbatim from kernel-0943 cell 39)\n"
            + with_prelude(train_body(), ARMS[arm]["argv"], model_body()))


def model_body():
    text = MODEL.read_text(encoding="utf-8")
    i = text.index('"""', text.index('"""') + 3) + 3
    return text[i:].strip("\n")


def train_body():
    text = TRAIN.read_text(encoding="utf-8")
    # the guarded import stays: inlined, it fails harmlessly and is caught on purpose
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    arms = [a.arm] if a.arm else sorted(ARMS)

    for arm in arms:
        spec = ARMS[arm]
        d = ROOT / "kernels" / spec["dir"]
        out = d / "kernel_main.py"
        text = payload(arm)
        probes = ["class Net", "class RaptorClassifier", "def a5_study", "N_SLOT_TYPES",
                  "raptor_oof_", "[prelude] no argv given"]
        for p in probes:
            if p not in text:
                raise SystemExit("!! payload is missing %r" % p)
        # `from __future__` must precede every other statement (a docstring and comments may precede
        # it).  Assert the property, not the layout: that is what the SyntaxError was about.
        head = text.split("def ", 1)[0]
        first_stmt = [l for l in head.split("\n")
                      if l.strip() and not l.strip().startswith("#")]
        if first_stmt and not first_stmt[0].startswith('"""'):
            raise SystemExit("!! first statement is %r" % first_stmt[0][:60])
        if "from __future__ import annotations" not in text.split("class ", 1)[0]:
            raise SystemExit("!! from __future__ is not near the top of the payload")
        # the prelude must run BEFORE the trailing main() call, and main() must run AFTER the model
        # classes are defined.  Assert the ORDER, because that is what all three failures were about.
        if not (text.rindex("[prelude] no argv given") < text.rindex("class Net") < text.rindex("\nmain()")):
            raise SystemExit("!! payload order is wrong: prelude < class Net < main() required")

        if a.check:
            if not out.is_file() or out.read_text(encoding="utf-8") != text:
                raise SystemExit("!! %s is stale -- rerun scripts/build_a5lite.py" % out)
            print("CHECK OK: %s" % out)
            continue

        d.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        meta = {"id": spec["kid"], "title": spec["title"], "code_file": out.name,
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_tpu": False,
                "enable_internet": True,      # needed: the paired arms both start from timm weights
                "keywords": [],
                "dataset_sources": ["dreaddevelopment/knee-raptor-corpus",
                                    "dreaddevelopment/knee-raptor-corpus-ext",
                                    "flight0234/rsna-knee-hybrid-report-labels",
                                    "ppxl16/rsna-knee-cv-folds"],
                "kernel_sources": [], "competition_sources": [], "model_sources": []}
        (d / "kernel-metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote %s (%d chars); id=%s" % (out, len(text), spec["kid"]))
        print("   argv: %s" % spec["argv"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
