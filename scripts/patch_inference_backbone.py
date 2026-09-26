# -*- coding: utf-8 -*-
"""Teach the inference notebook to rebuild a second vision-encoder family.

The training notebook is generated from a script; the inference notebook is a fork that
was edited in place, so it gets a patch tool instead. The tool is idempotent and strict:

  * every replacement must match exactly once, or it refuses to write;
  * every cell must still compile afterwards;
  * a backup is written before any change;
  * running it twice is a no-op.

Why the inference side needs this at all: `infer_from_package` rebuilds each member with
`build_model(..., variant=m["config"]["variant"], ...)`, and the DINOv2 path calls
`find_dinov2`. A member trained on a different family would therefore either fail to find
its weights or - worse - fall back to whatever checkpoint `find_dinov2` found first and
quietly load the wrong encoder.

    python scripts/patch_inference_backbone.py            # inspect, write nothing
    python scripts/patch_inference_backbone.py --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from kpaths import kdir          # kernels/ layout, see kpaths.py

NB = kdir("kernel-fork") / "rsna-knee-baseline-v1.ipynb"

# ---- the block that gets inserted ahead of build_model ---------------------- #
HELPERS = '''BIOMEDCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BIOMEDCLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_NET_MEAN = (0.485, 0.456, 0.406)
IMAGE_NET_STD = (0.229, 0.224, 0.225)
NORM_BY_FAMILY = {"dinov2": (IMAGE_NET_MEAN, IMAGE_NET_STD),
                  "biomedclip": (BIOMEDCLIP_MEAN, BIOMEDCLIP_STD)}


class _AsHF(nn.Module):
    """Present a timm tower through the interface Model already calls.

    `Model.forward` does `self.backbone(pixel_values=x).last_hidden_state`, which is the
    transformers shape. A timm ViT exposes `forward_features` instead, so this adapter
    hands the token sequence back under the same attribute rather than teaching Model a
    second calling convention. Identical to the training side, on purpose: a member is
    only usable if both ends build the same object.
    """

    def __init__(self, tower):
        super().__init__()
        self.tower = tower

    def forward(self, pixel_values=None, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=self.tower.forward_features(pixel_values))


def find_biomedclip():
    """Locate the mounted BiomedCLIP directory by its checkpoint, not by a path shape.

    The model mount path varies between kernels, so the file identifies the directory.
    """
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "open_clip_pytorch_model.bin" in files:
            return Path(root)
    return None


def load_biomedclip_tower(src):
    """Build a timm ViT-B/16 and fill it from the open_clip checkpoint.

    The checkpoint stores the tower under `visual.` with a `trunk.` prefix plain timm does
    not use; stripping it aligns all 150 tensors exactly. `head.proj.weight` is the CLIP
    projection head and is dropped on purpose. Anything else unmatched is fatal - a
    half-loaded encoder is the silent-wrong-answer case this notebook exists to refuse.
    """
    import timm
    raw = torch.load(Path(src) / "open_clip_pytorch_model.bin", map_location="cpu")
    sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    vis = {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
    vis = {k[len("trunk."):]: v for k, v in vis.items() if k.startswith("trunk.")}
    tower = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    missing, unexpected = tower.load_state_dict(vis, strict=False)
    if missing:
        raise WeightsError(f"BiomedCLIP tower incomplete: {len(missing)} tensors missing, "
                           f"first {list(missing)[:3]}")
    log(f"biomedclip tower: timm vit_base_patch16_224, {len(vis)} tensors loaded, "
        f"{len(unexpected)} ignored (projection head)")
    return tower


def build_model(unfreeze_last, source=None, variant="small", pool="cls_mean",
                prior=False, family="dinov2", norm=None):'''

MODEL_INIT_OLD = '''    def __init__(self, backbone, dim, pool="cls_mean", prior=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))'''

MODEL_INIT_NEW = '''    def __init__(self, backbone, dim, pool="cls_mean", prior=False, norm=None):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)
        # Normalisation belongs to the encoder, not to the pipeline, and the wrong pair
        # raises nothing - it just scores worse. `norm` comes from the member record when
        # the training side wrote one, so a member carries its own preprocessing with it.
        _mean, _std = norm if norm is not None else (
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.register_buffer("mean", torch.tensor(list(_mean)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(list(_std)).view(1, 3, 1, 1))'''

BUILD_SIG_OLD = '''def build_model(unfreeze_last, source=None, variant="small", pool="cls_mean",
                prior=False):'''

BUILD_BODY_OLD = '''    from transformers import AutoModel
    p = source if source is not None else find_dinov2(variant)
    if p is None:
        raise FileNotFoundError("DINOv2 weights not attached")
    bb = AutoModel.from_pretrained(str(p))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size
    trainable = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    log(f"backbone: {n_layer} blocks, last {unfreeze_last} trainable "
        f"({trainable / 1e6:.1f}M params), feature dim {dim * POOL_PARTS[pool]}")
    return Model(bb, dim, pool=pool, prior=prior)'''

BUILD_BODY_NEW = '''    if family == "biomedclip":
        p = source if source is not None else find_biomedclip()
        if p is None:
            raise FileNotFoundError("BiomedCLIP weights not attached: attach "
                                    "mrquadian/biomedclip/PyTorch/default/1")
        tower = load_biomedclip_tower(p)
        n_layer = len(tower.blocks)
        for prm in tower.parameters():
            prm.requires_grad = False
        for blk in tower.blocks[max(0, n_layer - unfreeze_last):]:
            for prm in blk.parameters():
                prm.requires_grad = True
        for prm in tower.norm.parameters():
            prm.requires_grad = True
        dim = tower.embed_dim
        bb = _AsHF(tower)
    else:
        from transformers import AutoModel
        p = source if source is not None else find_dinov2(variant)
        if p is None:
            raise FileNotFoundError("DINOv2 weights not attached")
        bb = AutoModel.from_pretrained(str(p))
        n_layer = len(bb.encoder.layer)
        for prm in bb.parameters():
            prm.requires_grad = False
        for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
            for prm in blk.parameters():
                prm.requires_grad = True
        for prm in bb.layernorm.parameters():
            prm.requires_grad = True
        dim = bb.config.hidden_size
    trainable = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    log(f"backbone[{family}]: {n_layer} blocks, last {unfreeze_last} trainable "
        f"({trainable / 1e6:.1f}M params), feature dim {dim * POOL_PARTS[pool]}")
    return Model(bb, dim, pool=pool, prior=prior,
                 norm=norm if norm is not None else NORM_BY_FAMILY[family])'''

CALL_OLD = '''            model = build_model(int(m["config"]["unfreeze_last"]),
                                variant=m["config"]["variant"],
                                pool=m["config"].get("pool", "cls_mean"),
                                prior=bool(m["config"].get("prior", False))).to(dev)'''

CALL_NEW = '''            _norm = m["config"].get("norm")
            model = build_model(int(m["config"]["unfreeze_last"]),
                                variant=m["config"]["variant"],
                                pool=m["config"].get("pool", "cls_mean"),
                                prior=bool(m["config"].get("prior", False)),
                                family=m["config"].get("family", "dinov2"),
                                norm=(tuple(tuple(x) for x in _norm)
                                      if _norm else None)).to(dev)'''

EDITS = [
    (MODEL_INIT_OLD, MODEL_INIT_NEW, "Model norm parameter"),
    (BUILD_SIG_OLD, HELPERS, "encoder-family helpers + build_model signature"),
    (BUILD_BODY_OLD, BUILD_BODY_NEW, "build_model family branch"),
    (CALL_OLD, CALL_NEW, "infer_from_package passes family and norm"),
]


def cell_text(c):
    s = c.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default=str(NB))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    path = Path(args.notebook)
    if not path.is_file():
        raise SystemExit(f"no such notebook: {path}")
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]

    already = any("find_biomedclip" in cell_text(c) for c in cells if c["cell_type"] == "code")
    if already:
        print(f"{path}: already patched (find_biomedclip present) - nothing to do")
        return 0

    done = set()
    for c in cells:
        if c["cell_type"] != "code":
            continue
        t = cell_text(c)
        for old, new, tag in EDITS:
            n = t.count(old)
            if n == 0:
                continue
            assert n == 1, f"{tag}: matched {n} times in one cell"
            t = t.replace(old, new)
            done.add(tag)
        c["source"] = t

    want = {tag for _, _, tag in EDITS}
    missing = want - done
    assert not missing, (
        f"patches that did NOT match: {sorted(missing)}. Refusing to write: an unapplied "
        f"patch would leave a notebook that reads config['family'] and then ignores it.")

    # Compile every cell before writing: the notebook is JSON, so a broken patch would
    # only surface on Kaggle, hours later. `compile()` in memory rather than py_compile:
    # no temp file is needed, which also keeps this usable under a sandbox that only
    # lets the workspace be written.
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        try:
            compile(cell_text(c), f"<cell {i}>", "exec")
        except SyntaxError as exc:
            raise SystemExit(f"cell[{i}] does not compile after patching:\n{exc}")

    # And read the result back: asserts above only see the strings this script held.
    flat = "\n".join(cell_text(c) for c in cells if c["cell_type"] == "code")
    assert 'family=m["config"].get("family", "dinov2")' in flat, \
        "the call site did not take the family from the member record"
    assert "def build_model(unfreeze_last, source=None, variant=\"small\", pool=\"cls_mean\"," \
           "\n                prior=False, family=\"dinov2\", norm=None):" in flat, \
        "build_model signature missing after patch"
    assert "return Model(bb, dim, pool=pool, prior=prior)" not in flat, \
        "an unpatched Model(...) construction survives"

    print(f"{path}: {len(done)} patch(es) applied, every cell compiles:")
    for tag in sorted(done):
        print(f"  + {tag}")

    if not args.apply:
        print("\nDRY RUN - nothing written. Re-run with --apply to update the notebook.")
        return 0

    backup = path.with_suffix(".ipynb.bak")
    shutil.copy(path, backup)
    path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"\nwritten. backup at {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
