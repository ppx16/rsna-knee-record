"""Inspect a published checkpoint's structure so we can tell whether our stack can consume it.

WHY: `t88take/rsna-knee-t88-weights` -> exp085 claims **LB 0.947 as a SINGLE model** with
**gold58 macro-AUC 0.9062** -- above the A167 member gate (solo >= 0.90), and its manifest says the
5 folds are sha256-verified against the submission that produced 0.947. The author's code dataset
(`t88take/rsna-knee-code`) is 403/private, so the checkpoint is the only description of the model we
can get. Its TENSOR SHAPES are enough to identify the backbone, the input resolution (through the
position-embedding grid) and the head.

⚠️ Loaded with `weights_only=True`. The file comes from a public dataset and a pickle can execute
arbitrary code on load; `weights_only=True` refuses anything that is not plain tensors/metadata, and
if it refuses we treat that as a finding rather than disabling the guard.

Usage:  python work/inspect_ckpt.py work/newckpt/fold0.pt
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
import collections
import pathlib
import sys

import torch


def main() -> int:
    p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "work/newckpt/fold0.pt")
    print("file: %s  %.1f MB" % (p, p.stat().st_size / 1e6))
    try:
        obj = torch.load(p, map_location="cpu", weights_only=True)
    except Exception as e:
        print("!! weights_only=True refused: %s: %s" % (type(e).__name__, str(e)[:300]))
        print("   (this is a finding: the checkpoint carries non-tensor objects)")
        return 2
    print("top-level type: %s" % type(obj).__name__)
    if isinstance(obj, dict):
        print("top-level keys: %s" % list(obj.keys())[:20])
        for k, v in obj.items():
            if not torch.is_tensor(v):
                print("   %-24s %s = %s" % (k, type(v).__name__, str(v)[:160]))
    sd = None
    for k in ("model", "state_dict", "model_state_dict", "weights", "net"):
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            sd = obj[k]
            print("using sub-dict %r (%d tensors)" % (k, len(sd)))
            break
    if sd is None and isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
        sd = obj
        print("using the top level as the state dict (%d tensors)" % len(sd))
    if sd is None:
        print("!! could not find a state dict")
        return 3
    names = list(sd.keys())
    print("\n%d tensors. first 12:" % len(names))
    for n in names[:12]:
        print("   %-64s %s" % (n, tuple(sd[n].shape)))
    print("\nlast 12:")
    for n in names[-12:]:
        print("   %-64s %s" % (n, tuple(sd[n].shape)))
    # group by prefix, to expose the head among the backbone
    pref = collections.Counter(n.split(".")[0] for n in names)
    print("\ntop-level prefixes: %s" % dict(pref))
    # anything that is NOT obviously backbone
    odd = [n for n in names if not any(n.startswith(x) for x in
                                       ("backbone.", "model.", "module.", "student.", "encoder."))]
    print("\nnon-backbone tensors (%d):" % len(odd))
    for n in odd[:40]:
        print("   %-64s %s" % (n, tuple(sd[n].shape)))
    # position embedding -> patch grid -> input resolution for a ViT
    for n in names:
        if "pos_embed" in n:
            sh = tuple(sd[n].shape)
            print("\n%s %s" % (n, sh))
            if len(sh) == 3:
                ntok = sh[1] - 1
                import math
                g = int(math.isqrt(ntok))
                print("   -> %d patch tokens +1 cls = %dx%d grid" % (ntok, g, g))
    total = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    print("\ntotal parameters: %.1f M" % (total / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
