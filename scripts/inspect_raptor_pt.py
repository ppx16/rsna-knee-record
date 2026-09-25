# -*- coding: utf-8 -*-
"""Inspect a Raptor view checkpoint to recover its GEOMETRY.

WHY
    To add a 5th view (`v9 finespacing`) to our ensemble I must reproduce the geometry
    the weights were trained with: `slots` / `span` / `k_eval` / `img` / `res` / `arch`.
    No public notebook carries a v9 `ARMS` entry (I checked all six 0.941 forks -- they are
    byte-identical lists of the same four arms), and `n942` is a different lineage.  Guessing
    would produce an arm that RUNS but predicts wrongly -- the silent-no-op failure class
    (HANDOFF 16.1).  So ask the weights themselves.

    python scripts/inspect_raptor_pt.py downloads/raptor_v9/raptor_ft_coatnet_v9_full.pt
"""
from __future__ import annotations

import pathlib
import sys
import zipfile

import torch

WANT = ("slot", "span", "k_eval", "img", "res", "arch", "name", "file", "norm",
        "crop", "reverse", "mean", "std", "epoch", "gold", "auc", "fingerprint")


def unwrap(p: pathlib.Path) -> pathlib.Path:
    if p.suffix == ".zip" or zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            z.extractall(p.parent)
        hits = sorted(x for x in p.parent.rglob("*.pt"))
        if not hits:
            raise SystemExit("no .pt inside %s" % p)
        return hits[0]
    return p


def show(obj, prefix="", depth=0):
    if depth > 2:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if any(w in kl for w in WANT):
                if isinstance(v, (int, float, str, bool, type(None))):
                    print("  %s%s = %r" % (prefix, k, v))
                elif isinstance(v, (list, tuple)):
                    print("  %s%s = %r" % (prefix, k, list(v)[:24]))
                else:
                    print("  %s%s : %s" % (prefix, k, type(v).__name__))
            if isinstance(v, dict):
                show(v, prefix + str(k) + ".", depth + 1)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    pt = unwrap(pathlib.Path(sys.argv[1]))
    print("checkpoint: %s (%.1f MB)" % (pt.name, pt.stat().st_size / 1e6))
    try:
        ck = torch.load(pt, map_location="cpu", weights_only=True)
        print("loaded with weights_only=True")
    except Exception as e:
        print("weights_only=True failed (%s); retrying weights_only=False" % repr(e)[:90])
        ck = torch.load(pt, map_location="cpu", weights_only=False)

    print("\ntop level: %s" % type(ck).__name__)
    if isinstance(ck, dict):
        for k, v in ck.items():
            extra = ""
            if hasattr(v, "shape"):
                extra = " shape=%s dtype=%s" % (tuple(v.shape), v.dtype)
            elif isinstance(v, dict):
                extra = " (dict, %d entries)" % len(v)
            elif isinstance(v, (int, float, str, bool, type(None))):
                extra = " = %r" % (v,)
            elif isinstance(v, (list, tuple)):
                extra = " = %r" % (list(v)[:14],)
            print("  - %-24s %s%s" % (k, type(v).__name__, extra))
    print("\n=== geometry-ish keys (recursive) ===")
    show(ck)
    if isinstance(ck, dict) and isinstance(ck.get("model"), dict):
        sd = ck["model"]
        print("\nstate_dict: %d tensors, %.1f M params"
              % (len(sd), sum(v.numel() for v in sd.values() if hasattr(v, "numel")) / 1e6))
        ks = list(sd)
        print("  first keys: %s" % ks[:4])
        print("  last keys : %s" % ks[-4:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
