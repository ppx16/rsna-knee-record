# -*- coding: utf-8 -*-
"""Extract the frontier's A5 model code out of the submission notebook, verbatim, into the repo.

WHY.  `kernels/kernel-0943` cell 39 contains the complete definition of the A5 model -- the arm this
project kept calling "the recipe we cannot reproduce".  It was never unreadable; it had simply not
been read (FACTS A148).  Until it lives in *our* tree it cannot be versioned, diffed, or trained
against, so this copies the class definitions out byte for byte and stops where the frontier's own
loader begins.

WHAT IT DOES NOT COPY.  The trailing loader block of that cell (which builds five models from
`m_f0..m_f4`) is left behind: `kernels/kernel-a5port/gate.py` reimplements it, so the construction
can be asserted step by step rather than inherited wholesale.

Usage:
    python scripts/extract_a5_model.py            # writes kernels/kernel-a5port/a5_model.py
    python scripts/extract_a5_model.py --check     # re-verify the extracted file against the source
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parent.parent
NB = ROOT / "kernels" / "kernel-0943" / "rsna-knee-restructured-version-3.ipynb"
CELL = 39
DST = ROOT / "kernels" / "kernel-a5port" / "a5_model.py"
MARKER = "models=[]"

HEADER = '''# -*- coding: utf-8 -*-
"""The frontier's A5 model, extracted VERBATIM from the submission notebook.

PROVENANCE.  `kernels/kernel-0943` cell %d, from the top of the cell down to (not including) the
line beginning `%s` -- i.e. every class, copied without edits.  The frontier ships their model code
inside the notebook that produces the submission, so "we cannot reproduce their recipe" was never
quite true: it had not been read yet (FACTS A148/A149).  The loader block is deliberately NOT copied
here; `gate.py` reimplements it so each construction step can be asserted.

WHAT THE CHECKPOINT SAYS THE RECIPE IS -- read out of `m_f0.pt['cfg']`, not guessed:

    backbone  vit_small_patch16_dinov3.lvd1689m
    cond      token        slot-type tokens injected into the ViT
    pool      xcodex       CodexResidualPool: gated residual cross-attention over slot segments
    img       336
    norm      none         [0,1]; NOT imagenet
    n_slice   16           and with stem="native" the loader sets in_chans = n_slice = 16
    stem      native       16 slices enter as 16 CHANNELS -- there is no depth compression here
    n_meta    0, n_sites 109, pe_init "tiled", triplet "none"

⚠️ That last line is the answer to the 0.853-versus-0.909 question.  Our own DINOv3 arm used the SAME
backbone with OUR head (3-channel windows, attention-MIL across windows), so the gap was never the
backbone -- it is the input representation and the pooling.  A142 diagnosed it as architecture plus
recipe; this is the architecture, in code.

⚠️ `stem="native"` with `in_chans=16` is exactly the detail that degrades a port silently: feeding
three-channel windows to a 16-channel patch embed fails loudly, but feeding the right CHANNEL COUNT
from the WRONG slices would not.
"""

'''


def source_body():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    lines = "".join(nb["cells"][CELL]["source"]).split("\n")
    cut = next(i for i, l in enumerate(lines) if l.startswith(MARKER))
    return "\n".join(lines[:cut]), cut


def body_of(path):
    """The extracted file's content without its header, for the check mode."""
    text = path.read_text(encoding="utf-8")
    i = text.index('"""', text.index('"""') + 3) + 3     # end of the module docstring
    return text[i:].lstrip("\n").rstrip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    body, cut = source_body()
    if a.check:
        if not DST.is_file():
            raise SystemExit("!! %s is missing -- run without --check first" % DST)
        have = body_of(DST)
        if have != body:
            raise SystemExit("!! a5_model.py no longer matches notebook cell %d "
                             "(the frontier may have shipped a new version)" % CELL)
        print("CHECK OK: a5_model.py == notebook cell %d, lines 0..%d" % (CELL, cut - 1))
        return 0

    text = (HEADER % (CELL, MARKER)) + body + "\n"
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(text, encoding="utf-8")
    classes = [l.strip()[6:].split("(")[0] for l in body.split("\n") if l.startswith("class ")]
    print("wrote %s" % DST)
    print("  copied notebook lines 0..%d  (%d chars, sha256 %s)"
          % (cut - 1, len(body), hashlib.sha256(body.encode()).hexdigest()[:16]))
    print("  classes: %s" % ", ".join(classes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
