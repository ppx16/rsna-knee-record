# -*- coding: utf-8 -*-
"""Build the training-version notebook from the baseline (v2: 5-fold ensemble).

Changes vs. the baseline (rsna-knee-baseline-v1.ipynb):
  1. SEED -> env-var driven, so re-running with a different SEED accumulates members.
  2. RUNS -> a single 224 px config (the weights-package format identifies a member by a
     single resolution where the pixel cache and the model input must agree; CACHE_IMG is
     set to 224 to match, which on a T4 lets plan_cache raise slices 3 -> 6).
  3. EPOCHS -> 20 target, N_FOLDS -> 5. The training loop becomes one member per fold, so
     a single run yields a 5-member ensemble. A per-fold time budget keeps the first fold
     from eating the whole run.
  4. N_GROUP_MAX -> 4 (more slice windows; plan_cache still clamps to what RAM allows).
  5. Add save_member() + manifest builder (the exact package find_weights() reads back)
     and call it after each fold.
  6. Backbone -> DINOv2-large (find_dinov2 + build_model default variant). dim/n_layer are
     derived from the checkpoint (bb.config.hidden_size / len(bb.encoder.layer)), so this is
     the only change needed; the vision encoder, not the labels or slices, is the bottleneck.

Run with enable_internet=True and NO weights dataset attached (find_weights() -> None).
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=2026,
                help="value written into the notebook's SEED default. A different seed "
                     "adds a distinct set of members to the same package, because a "
                     "member id hashes fold+seed; re-running the SAME seed produces the "
                     "same ids and is deduplicated away, i.e. a wasted run")
ap.add_argument("--epochs", type=int, default=20,
                help="EPOCHS target per fold. Note this is NOT a time knob for the large "
                     "config: the loop runs min(EPOCHS, what the per-fold share of "
                     "TIME_BUDGET affords), which is ~4-5 for large. What EPOCHS does "
                     "control is the OneCycleLR schedule length, so setting it far above "
                     "what fits leaves the schedule un-annealed at truncation. Set it to "
                     "what actually fits")
ap.add_argument("--variant", default="large", choices=["small", "large"],
                help="DINOv2 backbone. The slice-group config is independent of this, so "
                     "--variant small with the current settings reproduces the 224x6 "
                     "baseline's backbone at 12 slices - which isolates the slice change "
                     "from the backbone change that v4 made at the same time. NOTE: the "
                     "KERNEL METADATA must attach a checkpoint for this variant; "
                     "find_dinov2 falls back to whatever it found first (and build_model "
                     "takes the architecture from that checkpoint), so a mismatch is "
                     "silent at training time")
ap.add_argument("--backbone", default="dinov2", choices=["dinov2", "biomedclip"],
                help="vision encoder family. 'dinov2' is the default and reproduces the "
                     "existing notebook byte for byte; 'biomedclip' applies an extra patch "
                     "set that swaps in a biomedical image-text encoder loaded through the "
                     "preinstalled timm (see HANDOFF 10 for why that is possible at all)")
ARGS = ap.parse_args()

SRC = Path("notebooks/rsna-knee-baseline-v1.ipynb")
DST = Path("notebooks/rsna-knee-train-v1.ipynb")

nb = json.loads(SRC.read_text(encoding="utf-8"))
cells = nb["cells"]


def cell_text(c):
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s


def set_cell_text(c, text):
    c["source"] = text  # single-string source is a valid ipynb form


def replace_once(text, old, new, tag):
    assert text.count(old) == 1, f"{tag}: expected 1 occurrence, found {text.count(old)}"
    return text.replace(old, new)


# ---- config edits (cell 13) ----------------------------------------------- #
RUNS_OLD = '''# Resolution is the axis under test. A feature of width d mm survives resampling only if
# the pixel pitch is at most d/2, and the pitch here is set by the crop above rather than
# by the acquired field of view: CROP_MM / P. At 224 px that is 0.58 mm, above the 0.5 mm
# a 1 mm tear needs; at 336 px it is 0.39 mm and clears it. Both configurations read the
# same cache, so the comparison isolates the resize.
RUNS = [
    {"name": "r224", "img": 224},
    {"name": "r336", "img": 336},
]'''

RUNS_NEW = '''# Resolution. A feature of width d mm survives resampling only if the pixel pitch is at
# most d/2, and the pitch here is set by the crop above rather than by the acquired field
# of view: CROP_MM / P. At 224 px the pitch is 0.58 mm, right at the 0.5 mm a 1 mm tear
# needs; 336 px gives 0.39 mm. The cache is IMG^2 * slices bytes, so the coarser 224 px
# grid buys more slice windows for the same RAM. Measured on this competition: 224 px with
# 6 slices beat 336 px with 3 by 0.023 holdout, so coverage is the cheaper axis - which is
# why this run doubles it instead of buying pixels. The weights package identifies a
# member by one resolution where the pixel cache and the model input must agree, so
# CACHE_IMG is set to 224 below to match this run.
RUNS = [
    {"name": "r224", "img": 224},
]'''

CONFIG_EDITS = [
    ("SEED = 2026",
     "SEED = int(os.environ.get(\"SEED\", \"%d\"))  # per-run seed; re-run with another SEED to grow the ensemble" % ARGS.seed,
     "SEED"),
    ("CACHE_BUDGET_MAX_GB = 24.0",
     "CACHE_BUDGET_MAX_GB = 24.0 # hard ceiling regardless of what the machine reports\n"
     "CACHE_GROUPS_TO_FIT = 4    # slice windows to fit regardless of plan_cache's own\n"
     "                           # arithmetic; 0 leaves the decision to plan_cache",
     "CACHE_GROUPS_TO_FIT"),
    ("EPOCHS = 10",
     "EPOCHS = %d%s# target epochs per fold; the per-fold time budget may cut it short\nN_FOLDS = 5               # report-md5 CV folds; one trained member per fold"
     % (ARGS.epochs, " " * max(1, 17 - len(str(ARGS.epochs)))),
     "EPOCHS/N_FOLDS"),
    ("N_GROUP_MAX = 1",
     "N_GROUP_MAX = 4        # slice windows per study (slices = GROUP * N_GROUP); plan_cache clamps to what RAM allows",
     "N_GROUP_MAX"),
    ("CACHE_IMG = 336",
     "CACHE_IMG = 224        # decode the cache at 224 so cache resolution == 224 model input (see RUNS)",
     "CACHE_IMG"),
    ("def find_dinov2(variant=\"small\"):",
     f"def find_dinov2(variant=\"{ARGS.variant}\"):",
     "find_dinov2 variant"),
    ("variant=\"small\", pool=\"cls_mean\"",
     f"variant=\"{ARGS.variant}\", pool=\"cls_mean\"",
     "build_model variant"),
]

# ---- optional: swap the vision encoder ------------------------------------- #
# Everything below is applied ONLY with --backbone biomedclip, so the DINOv2 path keeps
# producing the byte-identical notebook it produced before (HANDOFF 2.6 keeps a hash for
# exactly that check).
#
# Why this encoder: HANDOFF 8 concludes the real bottleneck is vision capacity, not labels
# (on the 58 gold studies text scores 0.791 against vision's 0.674). BiomedCLIP is
# pretrained on biomedical image-text pairs, so it is a different inductive bias rather
# than just a bigger DINOv2.
#
# Why it is loadable at all (all measured with the CPU-only probe kernel, HANDOFF 10):
#   * Kaggle mounts it at /kaggle/input/models/mrquadian/biomedclip/pytorch/default/1/
#   * its weights are `open_clip_pytorch_model.bin`, an OpenCLIP checkpoint
#   * `open_clip` is NOT in the image and the inference notebook runs with
#     enable_internet=False, so it could never be pip installed at scoring time
#   * but the model's own config says its vision tower is timm `vit_base_patch16_224`,
#     and timm IS installed - so no new dependency is needed
#   * the checkpoint's `visual.*` tensors carry a `trunk.` prefix that plain timm does not
#     use; stripping it aligns all 150 tensors exactly (missing 0, unexpected 0)

# The normalisation constants differ from the ImageNet ones hardcoded in Model, and using
# the wrong pair raises nothing - it just scores worse. Recorded per member so the
# inference side cannot get it wrong.
BACKBONE_OLD = '''def build_model(unfreeze_last, source=None, variant="%s", pool="cls_mean",
                prior=False):''' % ARGS.variant

BACKBONE_NEW = '''BIOMEDCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BIOMEDCLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_NET_MEAN = (0.485, 0.456, 0.406)
IMAGE_NET_STD = (0.229, 0.224, 0.225)
NORM_BY_FAMILY = {"dinov2": (IMAGE_NET_MEAN, IMAGE_NET_STD),
                  "biomedclip": (BIOMEDCLIP_MEAN, BIOMEDCLIP_STD)}


class _AsHF(nn.Module):
    """Present a timm tower through the interface Model already calls.

    `Model.forward` does `self.backbone(pixel_values=x).last_hidden_state`, which is the
    transformers shape. A timm ViT exposes `forward_features` instead, so this adapter
    hands back the token sequence under the same attribute rather than teaching Model a
    second calling convention -- one definition of how a bag reaches the head.
    """

    def __init__(self, tower):
        super().__init__()
        self.tower = tower

    def forward(self, pixel_values=None, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=self.tower.forward_features(pixel_values))


def find_biomedclip():
    """Locate the mounted BiomedCLIP directory by its checkpoint, not by a path shape.

    Kaggle's model mount path varies between kernels (measured twice, differently), so
    the file is what identifies the directory.
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

    `head.proj.weight` is the CLIP projection head and is dropped on purpose: SlotHead
    replaces it. Anything else unmatched is fatal, because a half-loaded encoder is the
    silent-wrong-answer case this notebook keeps trying to avoid.
    """
    import timm
    raw = torch.load(Path(src) / "open_clip_pytorch_model.bin", map_location="cpu")
    sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    vis = {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
    vis = {k[len("trunk."):]: v for k, v in vis.items() if k.startswith("trunk.")}
    tower = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    missing, unexpected = tower.load_state_dict(vis, strict=False)
    if missing:
        raise RuntimeError(f"BiomedCLIP tower incomplete: {len(missing)} tensors missing, "
                           f"first {list(missing)[:3]}")
    log(f"biomedclip tower: timm vit_base_patch16_224, {len(vis)} tensors loaded, "
        f"{len(unexpected)} ignored (projection head)")
    return tower


def build_model(unfreeze_last, source=None, variant="%s", pool="cls_mean",
                prior=False, family="dinov2"):''' % ARGS.variant

BACKBONE_BODY_OLD = '''    from transformers import AutoModel
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

BACKBONE_BODY_NEW = '''    if family == "biomedclip":
        p = source if source is not None else find_biomedclip()
        if p is None:
            raise FileNotFoundError("BiomedCLIP weights not attached")
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
    return Model(bb, dim, pool=pool, prior=prior, norm=NORM_BY_FAMILY[family])'''

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
        # Normalisation belongs to the encoder, not to the pipeline: DINOv2 was fitted
        # with the ImageNet constants and BiomedCLIP with the CLIP ones, and the wrong
        # pair raises nothing - it just scores worse. The default keeps every existing
        # member byte-identical.
        _mean, _std = norm if norm is not None else (
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.register_buffer("mean", torch.tensor(list(_mean)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(list(_std)).view(1, 3, 1, 1))'''

CALL_OLD = "model = build_model(UNFREEZE_LAST).to(dev)"
CALL_NEW = "model = build_model(UNFREEZE_LAST, family=BACKBONE_FAMILY).to(dev)"

# Backbone constants and the member-config labels. These are applied whenever the build is
# NOT the default, because `_member_config` hardcodes variant="large" and
# backbone="facebook/dinov2-large": a `--variant small` run would otherwise write members
# that claim to be large, and the inference side rebuilds the model from exactly those
# fields. (That latent bug predates this work; the small-variant run recommended in 2.5
# would have hit it.)
_PAD = " " * (17 - len(str(ARGS.epochs)))
CONST_OLD = f"EPOCHS = {ARGS.epochs}{_PAD}# target epochs per fold; the per-fold time budget may cut it short"
_REPO = {"dinov2": f"facebook/dinov2-{ARGS.variant}",
         "biomedclip": "mrquadian/biomedclip"}[ARGS.backbone]
_TAG = {"dinov2": ARGS.variant, "biomedclip": "biomedclip-vitb16"}[ARGS.backbone]
CONST_NEW = (f'BACKBONE_FAMILY = "{ARGS.backbone}"\n'
             f'BACKBONE_VARIANT = "{_TAG}"\n'
             f'BACKBONE_REPO = "{_REPO}"\n'
             + CONST_OLD)

CALL_EDITS = [(CONST_OLD, CONST_NEW, "BACKBONE_* constants")]
# NOTE: CALL_OLD is deliberately NOT in CALL_EDITS. The training call lives inside
# NEW_TRAIN, which the cell loop substitutes in at step 6 - AFTER these patches run - so
# patching it through the loop both wasted the edit and, worse, recorded the tag as
# applied while the substitution threw the result away. That is a fail-open shape: the
# assert passed and the generated notebook still called build_model without `family`.
# The call is patched on NEW_TRAIN itself, at the point where NEW_TRAIN is defined.

BACKBONE_EDITS = [
    (MODEL_INIT_OLD, MODEL_INIT_NEW, "Model norm parameter"),
    (BACKBONE_OLD, BACKBONE_NEW, "build_model signature + timm tower"),
    (BACKBONE_BODY_OLD, BACKBONE_BODY_NEW, "build_model family branch"),
]

# Applied to SAVE_CELL before it is inserted, since that cell is not part of the baseline.
LABEL_NEEDED = (ARGS.variant != "large") or (ARGS.backbone != "dinov2")

MEMBER_CFG_OLD = '''        "variant": "large",
        "backbone": "facebook/dinov2-large",'''

MEMBER_CFG_NEW = ('        "family": BACKBONE_FAMILY,\n'
                  '        "variant": BACKBONE_VARIANT,\n'
                  '        "backbone": BACKBONE_REPO,\n')
if ARGS.backbone == "biomedclip":
    # The two families do not share normalisation constants, and the inference side
    # rebuilds preprocessing from the member record, so this has to travel with the member.
    MEMBER_CFG_NEW += ('        "norm": [list(NORM_BY_FAMILY[BACKBONE_FAMILY][0]),\n'
                       '                 list(NORM_BY_FAMILY[BACKBONE_FAMILY][1])],\n')

# ---- plan_cache override --------------------------------------------------- #
# plan_cache sizes the cache against a reserve for the hidden test corpus
# (TEST_SHARE x the training one), which no training kernel ever holds: training stops,
# and the scoring notebook loads members rather than pixels. On a T4 that reserve is what
# clamped the cache to 2 groups of 3 - 6 slices - when the pixels the run actually
# allocates are 4407 * 6 * 12 * 224^2 = 15.9 GB out of 29.8 GB. The slice count is
# therefore set explicitly and checked against what is really free, so the safety net
# that matters (do not overshoot free memory) is kept and only the fictional one is
# dropped.
PLAN_CACHE_OLD = "    groups = max(1, min(N_GROUP_MAX, afford // GROUP))\n"

PLAN_CACHE_NEW = '''    groups = max(1, min(N_GROUP_MAX, afford // GROUP))
    if CACHE_GROUPS_TO_FIT:
        want = max(1, min(N_GROUP_MAX, CACHE_GROUPS_TO_FIT))
        need = n_study * N_SLOT * want * GROUP * IMG * IMG / 1024 ** 3
        assert need <= avail * 0.75, (
            f"CACHE_GROUPS_TO_FIT={want} needs {need:.1f} GB, over 75% of the "
            f"{avail:.1f} GB this machine reports free")
        if want != groups:
            log(f"plan_cache: {groups} -> {want} group(s): the run needs "
                f"{need:.1f} GB for {want * GROUP} slices x {IMG} px, and the reserve "
                f"for a test corpus this kernel never holds is what clamped it")
        groups = want
'''

# ---- save_member + manifest builder cell ----------------------------------- #
SAVE_CELL = '''# --- Save trained members as a reusable weights package --------------------- #
#
# A model is only useful if a later inference-only notebook can load it and trust that
# it computes what it computed here. The package written below is exactly what
# `find_weights` / `infer_from_package` read back: a `manifest.json` describing every
# member (the decode config it was fitted on, its architecture, a fingerprint of its
# weights) plus one `m_<id>.pt` checkpoint per member. After this run, `kaggle kernels
# output` downloads `weights/` and it can be re-uploaded as a private dataset. The member
# id is a hash of decode config + fold + seed, so re-running the same training overwrites
# the same file, while a different SEED (or a later fold) adds a distinct member.

WEIGHTS_OUT = Path("weights")
_PIXEL_KEYS = ("band", "crop_mm", "group", "img", "rules", "slices", "slots")


def _member_config(cfg):
    """The config a member records; matches what `adopt_config_globals` / `build_model`
    read back at inference. `img` must equal the cache resolution the member was fitted
    on, which is why RUNS above fits at 224 and CACHE_IMG is set to 224 to match.
    """
    return {
        "img": int(cfg["img"]),
        "slices": int(CACHE_SLICES),
        "group": int(GROUP),
        "crop_mm": float(CROP_MM),
        "band": [float(x) for x in SLICE_BAND],
        "slots": [s[0] for s in SLOTS],
        "rules": dict(RULES),
        "variant": "large",
        "backbone": "facebook/dinov2-large",
        "unfreeze_last": int(UNFREEZE_LAST),
        "pool": "cls_mean",
        "prior": False,
    }


def save_member(model, dev, cfg, fold=0, holdout=None, annot=None,
                epochs_done=None, seed=None, source_run="train-v1"):
    """Write one trained member and append it to the package manifest."""
    WEIGHTS_OUT.mkdir(exist_ok=True)
    config = _member_config(cfg)
    pixel_group = json.dumps({k: config[k] for k in _PIXEL_KEYS}, sort_keys=True)
    ident = hashlib.md5(f"{pixel_group}|{fold}|{seed}".encode()).hexdigest()[:10]
    fname = f"m_{ident}.pt"

    model.eval()
    fp = fingerprint(model, dev, int(cfg["img"]))  # (2, N_TARGETS) logits

    torch.save({"model": model.state_dict(), "fingerprint": fp}, WEIGHTS_OUT / fname)

    member = {
        "id": ident,
        "file": fname,
        "fold": int(fold),
        "tag": "",
        "config": config,
        "seed": int(seed) if seed is not None else SEED,
        "epochs_done": int(epochs_done) if epochs_done is not None else EPOCHS,
        "holdout": float(holdout) if holdout is not None else None,
        "annot": float(annot) if annot is not None and not np.isnan(annot) else None,
        "fingerprint": np.asarray(fp).tolist(),
        "pixel_group": pixel_group,
        "source_run": source_run,
    }

    man_path = WEIGHTS_OUT / "manifest.json"
    man = {"format": 2, "members": [], "history": [], "targets": TARGETS}
    if man_path.is_file():
        try:
            man = json.loads(man_path.read_text())
        except ValueError:
            man = {"format": 2, "members": [], "history": [], "targets": TARGETS}
    man.setdefault("targets", TARGETS)
    man.setdefault("history", [])
    if all(m["id"] != ident for m in man["members"]):
        man["members"].append(member)
    man_path.write_text(json.dumps(man, indent=2))
    log(f"  saved member {ident} -> {fname} (holdout {holdout:.4f})")
    return member
'''

# ---- 5-fold training section (replaces the single-holdout + RUNS loop) ---- #
NEW_TRAIN = '''    # Grouped on report text: some reports are byte-identical across studies and yield
    # one target vector for all of them, so splitting such a group scores the model on a
    # target whose source it has already trained on.
    import hashlib
    rep = train_df.set_index("StudyInstanceUID")["Report"].fillna("")
    grp = np.array([int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % N_FOLDS
                    for s in st_tr])
    gpos = {s: i for i, s in enumerate(st_tr)}

    # ---- fine-tune: one member per fold ------------------------------------ #
    # The corpus is split on the report text so byte-identical reports never straddle a
    # split, and every study is held out exactly once across the N_FOLDS folds. Each fold
    # trains one member, so one run yields an N_FOLDS-member ensemble; re-running with a
    # different SEED adds another full set to the weights package.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, test_preds = {}, {}

    remaining = TIME_BUDGET - (time.time() - T0)
    fold_budget = max(remaining / N_FOLDS, 60.0)

    for fold in range(N_FOLDS):
        va = np.array([i for i in keep if grp[i] == fold])
        tr = np.array([i for i in keep if grp[i] != fold])
        if len(va) == 0 or len(tr) < BATCH_STUDIES:
            cut = max(1, len(keep) // N_FOLDS)
            va = keep[fold * cut:(fold + 1) * cut]
            tr = np.concatenate([keep[:fold * cut], keep[(fold + 1) * cut:]])
        log(f"fold {fold}: train {len(tr)} / holdout {len(va)} studies")

        # The annotated studies stay in training - they are the highest-quality labels in
        # the corpus and there are too few to discard - so the honest annotation check
        # uses only the ones that fell in this fold's holdout. Evaluating on the rest
        # would score the model against examples it was trained on, at triple weight,
        # with the true answer.
        va_set = set(va.tolist())
        gi = np.array([gpos[s] for s in gold.index if s in gpos and gpos[s] in va_set])
        gold_y = gold.loc[[st_tr[i] for i in gi]].values.astype(int) if len(gi) else None
        yv = (Y[va] > 0.5).astype(int)
        log(f"  annotation check: {len(gi)} of {len(gold)} annotated studies in holdout")

        fold_deadline = time.time() + fold_budget
        for cfg in RUNS:
            pitch = CROP_MM / cfg["img"]
            log(f"=== fold {fold} {cfg['name']}: {cfg['img']} px, {pitch:.3f} mm/pixel, "
                f"{pitch * 14:.2f} mm per patch token ===")
            torch.manual_seed(SEED)
            model = build_model(UNFREEZE_LAST).to(dev)
            opt = torch.optim.AdamW([
                {"params": [p for p in model.backbone.parameters() if p.requires_grad],
                 "lr": LR_BACKBONE},
                {"params": model.head.parameters(), "lr": LR_HEAD},
            ], weight_decay=WEIGHT_DECAY)
            steps = max(EPOCHS * (len(tr) // BATCH_STUDIES), 1)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=[LR_BACKBONE, LR_HEAD], total_steps=steps, pct_start=0.15)
            scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

            best, best_state, best_annot, done_epochs = -1.0, None, float("nan"), 0
            for ep in range(EPOCHS):
                model.train()
                perm = np.random.permutation(tr)
                tot, nstep = 0.0, 0
                for b in range(0, len(perm) - BATCH_STUDIES + 1, BATCH_STUDIES):
                    sel = perm[b:b + BATCH_STUDIES]
                    rows = torch.from_numpy(Ctr[sel]).to(dev)
                    g = int(torch.randint(N_GROUP, (1,)).item())
                    imgs = augment(take_group(rows, g))
                    m = torch.from_numpy(Mtr[sel]).to(dev)
                    y = torch.from_numpy(Y[sel]).to(dev)
                    w = torch.from_numpy(W[sel]).to(dev)
                    with torch.autocast("cuda", enabled=dev.type == "cuda"):
                        loss = (F.binary_cross_entropy_with_logits(
                            model(imgs, m, cfg["img"]), y, reduction="none") * w).mean()
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    tot += loss.item()
                    nstep += 1

                pv = predict(model, Ctr, Mtr, va, dev, cfg["img"])
                d = macro_auc(yv, pv)
                g_auc = float("nan")
                if gold_y is not None and len(gi):
                    g_auc = macro_auc(gold_y, predict(model, Ctr, Mtr, gi, dev, cfg["img"]))
                log(f"  epoch {ep + 1}/{EPOCHS}  loss {tot / max(nstep, 1):.4f}"
                    f"  holdout {d:.4f}  annot(n={len(gi)}) {g_auc:.4f}")

                # Selection reads the holdout alone. The annotation check is reported
                # because it measures something different - agreement with a reading of
                # the images rather than of the reports - but only a handful of annotated
                # studies land in any one holdout, so its sampling error dwarfs the
                # differences between epochs and it cannot arbitrate between them.
                if d > best:
                    best, best_annot = d, g_auc
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                done_epochs = ep + 1
                if time.time() > fold_deadline or time.time() - T0 > TIME_BUDGET:
                    log("  time budget reached")
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
            key = f"fold{fold}_{cfg['name']}"
            results[key] = (best, best_annot)
            test_preds[key] = predict(model, Cte, Mte, np.arange(len(st_te)), dev,
                                      cfg["img"])
            log(f"  {key}: best holdout {best:.4f} (annot {best_annot:.4f})")
            if best_state is not None:
                save_member(model, dev, cfg, fold=fold, holdout=best, annot=best_annot,
                            epochs_done=done_epochs, seed=SEED,
                            source_run=f"train-{cfg['name']}-seed{SEED}")
            del model, opt, sched, scaler, best_state
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
'''

# The training call is patched HERE, on NEW_TRAIN itself, and not through the cell loop:
# the loop substitutes NEW_TRAIN into the cell at step 6, after the per-cell patches have
# already run, so a patch applied there would be discarded - and its tag would still read
# as "applied". Patching the source of the substitution is the only place it survives.
NEW_TRAIN_PATCHED = NEW_TRAIN
_train_call_patched = False
if LABEL_NEEDED:
    assert NEW_TRAIN.count(CALL_OLD) == 1, (
        "the training call is not where the label patch expects it in NEW_TRAIN; without "
        "it the notebook would define BACKBONE_FAMILY and then never pass it to the model")
    NEW_TRAIN_PATCHED = NEW_TRAIN.replace(CALL_OLD, CALL_NEW)
    _train_call_patched = True

# --------------------------------------------------------------------------- #
applied = {"runs": False, "save": False, "train": False, "cache": False, "config": 0}

for c in cells:
    if c["cell_type"] != "code":
        continue
    t = cell_text(c)

    # config edits live in the cell that defines them
    if "RUNS = [" in t and RUNS_OLD in t and not applied["runs"]:
        t = t.replace(RUNS_OLD, RUNS_NEW)
        applied["runs"] = True
    for old, new, tag in CONFIG_EDITS:
        if old in t and t.count(old) == 1:
            t = t.replace(old, new)
            applied["config"] += 1

    # The backbone swap is applied strictly: each patch must land exactly once, because a
    # silent skip here would produce a notebook that still says biomedclip in its config
    # while running DINOv2 - the worst possible failure mode for an experiment.
    if ARGS.backbone == "biomedclip":
        for old, new, tag in BACKBONE_EDITS:
            n = t.count(old)
            if n == 0:
                continue
            assert n == 1, f"{tag}: matched {n} times in one cell"
            t = t.replace(old, new)
            applied.setdefault("backbone", set()).add(tag)

    if LABEL_NEEDED:
        for old, new, tag in CALL_EDITS:
            n = t.count(old)
            if n == 0:
                continue
            assert n == 1, f"{tag}: matched {n} times in one cell"
            t = t.replace(old, new)
            applied.setdefault("label", set()).add(tag)

    if PLAN_CACHE_OLD in t:
        assert t.count(PLAN_CACHE_OLD) == 1, "plan_cache matched more than once"
        t = t.replace(PLAN_CACHE_OLD, PLAN_CACHE_NEW)
        applied["cache"] = True

    # the training section replacement lives in main()
    if "def main():" in t and "    # Grouped on report text:" in t:
        start = t.index("    # Grouped on report text:")
        end = t.index('    log("---- summary ----")')
        t = t[:start] + NEW_TRAIN_PATCHED + t[end:]
        applied["train"] = True

    set_cell_text(c, t)

# Insert the save_member cell immediately before the "try: main()" cell.
insert_at = None
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        t = cell_text(c)
        if t.strip().startswith("try:") and "main()" in t:
            insert_at = i
            break

assert insert_at is not None, "could not find the try: main() cell"
save_src = SAVE_CELL
if LABEL_NEEDED:
    assert save_src.count(MEMBER_CFG_OLD) == 1, \
        "member-config labels not found in SAVE_CELL (the cell changed?)"
    save_src = save_src.replace(MEMBER_CFG_OLD, MEMBER_CFG_NEW)
    applied.setdefault("label", set()).add("member config labels")
new_cell = {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": save_src}
cells.insert(insert_at, new_cell)
applied["save"] = True

# ---- verify ---------------------------------------------------------------- #
assert applied["runs"], "RUNS not replaced"
assert applied["train"], "training section not replaced"
assert applied["save"], "save_member cell not inserted"
assert applied["cache"], "plan_cache override not applied"
assert applied["config"] == 7, f"config edits: expected 7, applied {applied['config']}"

if ARGS.backbone == "biomedclip":
    got = applied.get("backbone", set())
    want = {tag for _, _, tag in BACKBONE_EDITS}
    missing = want - got
    assert not missing, (
        f"backbone patches that did NOT match the baseline: {sorted(missing)}. "
        f"This is fatal rather than skippable: an unapplied patch would leave a notebook "
        f"whose config says biomedclip while the code still builds DINOv2.")
    print(f"ok: backbone=biomedclip, {len(got)} tower patch(es) applied")
else:
    assert "backbone" not in applied, "dinov2 build must not touch the tower patches"

if LABEL_NEEDED:
    got = applied.get("label", set())
    want = {tag for _, _, tag in CALL_EDITS}
    missing = want - got
    assert not missing, (
        f"labelling patches that did NOT match: {sorted(missing)}. Fatal: the member "
        f"record is what the inference side rebuilds from, so a stale label means a "
        f"member that says large while holding small weights (or vice versa).")
    assert _train_call_patched, (
        "the training call was never patched on NEW_TRAIN, so family=BACKBONE_FAMILY is "
        "not reaching build_model and the run would silently build the wrong encoder")
    print(f"ok: labels updated for variant={ARGS.variant} backbone={ARGS.backbone}")
else:
    assert "label" not in applied, "the default build must stay byte-identical"
    assert not _train_call_patched, "the default build must not patch NEW_TRAIN"

# The final guard: read the generated notebook back and confirm the two things that the
# asserts above CANNOT see, because they only inspect the generator's own strings.
if LABEL_NEEDED:
    _gen = "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in cells if c["cell_type"] == "code")
    assert CALL_OLD not in _gen, (
        "the generated notebook still calls build_model WITHOUT family= - the patch was "
        "discarded somewhere in the substitution order")
    assert CALL_NEW in _gen, "the generated notebook does not call build_model with family="

out = json.dumps(nb, ensure_ascii=False, indent=1)
assert '"name": "r336"' not in out, "r336 still present"
assert "N_FOLDS" in out and "for fold in range(N_FOLDS):" in out
assert "def save_member(" in out
assert "save_member(model, dev, cfg, fold=fold" in out
assert "grp[i] == fold" in out and "grp[i] == 0" not in out
assert "CACHE_GROUPS_TO_FIT = 4" in out, "slice count not pinned"
assert "CACHE_BUDGET_MAX_GB = 24.0 # hard ceiling" in out, "bad split at CACHE_BUDGET_MAX_GB"
assert 'CACHE_IMG = 224' in out, "cache resolution not 224"
assert "N_GROUP_MAX = 4" in out, "N_GROUP_MAX not raised"

# compile-check every code cell: a bad escape in an injected cell (a \n inside a
# triple-quoted source string turning into a real newline) should fail here, not on the
# GPU after the decode pass has already run.
for _i, _c in enumerate(cells):
    if _c["cell_type"] != "code":
        continue
    _t = cell_text(_c)
    try:
        compile(_t, f"<cell {_i}>", "exec")
    except SyntaxError as _e:
        raise SystemExit(f"generated notebook has SyntaxError in cell {_i} line "
                         f"{_e.lineno}: {_e.msg}\n  {_t.splitlines()[_e.lineno - 1]}")

DST.write_text(out, encoding="utf-8")
print(f"wrote {DST} ({len(cells)} cells)")
print("ok: r224 x12 slices, 5-fold loop, save_member per fold")
