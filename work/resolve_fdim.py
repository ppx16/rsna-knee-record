"""Resolve the F_dim 1024 (raptor-knee-arms) vs 768 (our v5/v8/v10) question.

The puzzle: work/arms_check/raptor_ft_cnv2b336.pt and raptor_ft_coatnet384.pt reportedly carry
F_dim 1024, while our v5/v8/v10 -- same arch family -- load through an F_dim=768 path. If that is
a real mismatch it is a latent silent-failure trap; if it is two different meanings of the same
label, it is a non-issue and the family stays wireable.

So: do not compare the NUMBER. Compare WHERE IT LIVES and what the tensors actually demand.

For each checkpoint print:
  * top-level keys, and whether a cfg/config dict exists
  * every scalar whose key mentions dim/feat/width/hidden/embed/num
  * the attention-pooling head tensors and their shapes  (clsW/clsb/norm/att.*)
  * the backbone's terminal tensor shapes
and then decide, from the head's own input width, what the head actually requires.
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
import pathlib
import re
import sys

import torch

DIMKEY = re.compile(r"dim|feat|width|hidden|embed|num_|channel|out", re.I)
HEADKEY = re.compile(r"^(clsW|clsb|norm\.|att\.)", re.I)

FILES = [
    r"work\arms_check\raptor_ft_coatnet384.pt",
    r"work\arms_check\raptor_ft_cnv2b336.pt",
    r"work\arms_check\raptor_ft_effv2l480.pt",
    r"work\above\tann\x\raptor\v5.pt",
    r"work\above\tann\x\raptor\v8.pt",
    r"work\above\tann\x\raptor\v10.pt",
    r"downloads\rent\raptor_ft_cn0_swa.pt",
    r"work\msft2_dataset\raptor_ft_msft0_swa.pt",
]


def load(p: pathlib.Path):
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(p, map_location="cpu", weights_only=False)


def cfg_of(ck):
    """Find the config dict wherever it lives."""
    if not isinstance(ck, dict):
        return None
    for k in ("cfg", "config", "args", "hparams"):
        if isinstance(ck.get(k), dict):
            return ck[k]
    return None


def upembs(ck):
    """Find an embedding module tensor that reveals the backbone's terminal width."""
    sd = ck.get("model") if isinstance(ck.get("model"), dict) else ck
    if not isinstance(sd, dict):
        return {}
    return {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}


def main() -> int:
    for rel in FILES:
        p = pathlib.Path(rel)
        print("=" * 100)
        if not p.is_file():
            print(f"MISSING {rel}")
            continue
        ck = load(p)
        print(f"{rel}  ({p.stat().st_size/1e6:.1f} MB)  top={type(ck).__name__}")

        if isinstance(ck, dict):
            print("  top keys:", [k for k in ck][:14])
        cfg = cfg_of(ck)
        if cfg:
            print("  --- cfg scalars mentioning dim/feat/width/hidden/embed ---")
            for k, v in sorted(cfg.items()):
                if DIMKEY.search(str(k)) and isinstance(v, (int, float, str, bool, type(None))):
                    print(f"      cfg.{k} = {v!r}")
            for k in ("backbone", "arch", "model_name", "timm"):
                if k in cfg:
                    print(f"      cfg.{k} = {cfg[k]!r}")

        shapes = upembs(ck)
        if shapes:
            head = {k: s for k, s in shapes.items() if HEADKEY.search(k.split(".")[-1]) or HEADKEY.search(k)}
            print(f"  --- state_dict: {len(shapes)} tensors ---")
            print("  --- head tensors (clsW/clsb/norm/att.*) ---")
            for k in sorted(head):
                print(f"      {k:<28} {head[k]}")
            # the backbone's own terminal width: look for the classifier/norm just before head
            tail = [(k, s) for k, s in shapes.items() if k not in head]
            print("  --- last 6 non-head tensors (backbone terminal) ---")
            for k, s in tail[-6:]:
                print(f"      {k:<40} {s}")
        print()

    # Independent check of what timm itself says the arch's feature width is.
    print("=" * 100)
    print("timm's own num_features for the arch strings we and they name:")
    try:
        import timm

        for arch in ("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
                     "coatnet_rmlp_2_rw_384",
                     "convnextv2_base.fcmae_ft_in22k_in1k_384",
                     "tf_efficientnetv2_l.in21k_ft_in1k"):
            try:
                m = timm.create_model(arch, pretrained=False, num_classes=0)
                print(f"   {arch:<48} num_features = {getattr(m, 'num_features', '?')}")
                del m
            except Exception as e:  # noqa: BLE001
                print(f"   {arch:<48} NOT AVAILABLE ({str(e)[:60]})")
    except ImportError:
        print("   timm not importable from this interpreter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
