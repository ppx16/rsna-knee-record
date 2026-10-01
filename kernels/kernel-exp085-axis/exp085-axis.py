"""CPU gate for exp085: does the reconstruction work, and WHICH forward order is right?

WHAT THIS DECIDES, IN ONE RUN
    `t88take/rsna-knee-t88-weights` -> exp085 claims **LB 0.947 as a single model** with **gold58
    macro-AUC 0.9062**, sha256-verified against the submission that produced it. Its code is private
    (403), so the model was reconstructed from the checkpoint itself. `work/exp085_model.py` proves the
    MODULE TREE exactly -- `load_state_dict(strict=True)` passes over all 245 tensors with 0 missing,
    0 unexpected and 0 shape mismatches -- but **the ORDER of operations in the head is a hypothesis**
    read off shapes, not off code.

    Three readings are implemented (`A`, `B`, `C`). This kernel runs all of them on the real gold set
    and prints gold58 macro-AUC for each, crossed with two crop scales so the two unknowns (forward
    ORDER vs slice PREPROCESSING) can be told apart:

        a forward-order error reads ~0.5 (logits scrambled);
        a preprocessing error reads high-but-not-0.906 (the head still works, the input is off);
        the correct pair reads ~0.9062.

NO ACCELERATOR, NO QUOTA
    `enable_gpu: false, enable_tpu: false`. The weekly GPU quota is exhausted account-wide (30.45 h of
    30.00, refreshing 2026-10-03) and the single TPU slot is held by the queued msft2 fine-tune, so a
    CPU kernel is the ONLY way to get this answer today. 58 studies x 12 slices is 696 images per
    configuration; DINOv2-base at 224 on CPU is a few minutes per grid cell, not hours.

SLICE POLICY, taken from the checkpoint's OWN config (not invented)
    n_slice_cached 12 | n_group 4 | slice_select band_groups | group_pick per_study
    slice_sampling center | slice_depth_range (0.2, 0.8) | crop_mm 130.0 | img 224
    The corpus is already a preprocessed (44, 336, 336) volume per study, so the 12 slices are the
    evenly spaced indices inside the central 20-80 % of depth, and the crop is a centre square.
    ⚠️ `crop_mm` cannot be converted to pixels without knowing the corpus spacing, which is exactly
    why TWO crop scales are run instead of one guessed value.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import glob  # noqa: E402
import math  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import cv2  # noqa: E402
import pandas as pd  # noqa: E402

print("[env] torch %s | cuda %s" % (torch.__version__, torch.cuda.is_available()), flush=True)

# ── deps that the CPU image may not ship ────────────────────────────────────────────────────────
try:
    import transformers  # noqa: F401
except Exception as _e:
    print("[env] installing transformers (%s)" % repr(_e)[:80], flush=True)
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "transformers"], check=True)
import transformers  # noqa: E402
print("[env] transformers %s" % transformers.__version__, flush=True)

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


# ═══════════════════════════════ model (reconstruction) ═════════════════════════════════════════
class Backbone(nn.Module):
    def __init__(self, name, img):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model
        assert "dinov2-base" in name, name
        # ⚠️ image_size 518, NOT the input size 224: the checkpoint's position_embeddings are
        # (1, 1370, 768) = 1 + 37**2, HF's default grid, and 224 works by interpolation. Building at
        # 224 gives (1, 257, 768) and the load fails with a size mismatch (measured locally).
        cfg = Dinov2Config(image_size=518, patch_size=14, hidden_size=768, num_hidden_layers=12,
                           num_attention_heads=12, intermediate_size=3072)
        self.net = Dinov2Model(cfg)

    def forward(self, x):
        return self.net(x).last_hidden_state


class SlicePool(nn.Module):
    def __init__(self, feat=1536, attn=256, n_targets=12):
        super().__init__()
        self.feat, self.attn = feat, attn
        self.norm = nn.LayerNorm(feat)
        self.depth_mlp = nn.Sequential(nn.Linear(1, attn), nn.SiLU(), nn.Linear(attn, feat))
        self.key = nn.Linear(feat, attn)
        self.value = nn.Linear(feat, feat)
        self.query = nn.Parameter(torch.zeros(n_targets, attn))

    def forward(self, x, depth):
        x = self.norm(x) + self.depth_mlp(depth)
        k, v = self.key(x), self.value(x)
        a = torch.softmax(self.query.unsqueeze(0) @ k.transpose(-1, -2) / math.sqrt(self.attn), dim=-1)
        return a @ v, a


class Head(nn.Module):
    def __init__(self, feat=1536, attn=256, n_targets=12, n_slots=6):
        super().__init__()
        self.attn = attn
        self.proj = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, attn))
        self.out = nn.Linear(attn, n_targets)
        self.query = nn.Parameter(torch.zeros(n_targets, attn))
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, attn))
        self.slot_prior = nn.Parameter(torch.zeros(n_targets, n_slots))

    def forward(self, pooled, order):
        h = self.proj(pooled)
        if order == "A":
            return (h * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
        if order == "B":
            w = torch.softmax(h @ self.slot_emb.t() / math.sqrt(self.attn) + self.slot_prior, dim=-1)
            mixed = w @ self.slot_emb
            return (mixed * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
        if order == "C":
            lg = (h * self.query.unsqueeze(0)).sum(-1, keepdim=True) \
                 + h @ self.slot_emb.t() / math.sqrt(self.attn) + self.slot_prior
            w = torch.softmax(lg, dim=-1)
            return self.out(w @ self.slot_emb).mean(dim=1)
        raise SystemExit("unknown order %r" % order)


class Exp085(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))
        self.backbone = Backbone(cfg.get("backbone_name", "facebook/dinov2-base"), int(cfg.get("img", 224)))
        feat = int(cfg.get("slice_feature_dim", 1536))
        attn = int(cfg.get("slice_attention_dim", 256))
        self.slice_pool = SlicePool(feat, attn)
        self.head = Head(feat, attn)

    def forward(self, imgs, depth, order):
        b, s = imgs.shape[:2]
        tok = self.backbone(imgs.flatten(0, 1))
        # slice_feature_dim 1536 = 2 x 768 -> CLS concatenated with the mean patch token
        feat = torch.cat([tok[:, 0], tok[:, 1:].mean(1)], -1).view(b, s, -1)
        pooled, attn = self.slice_pool(feat, depth)
        return self.head(pooled, order)


def find(*names, root="/kaggle/input"):
    for n in names:
        g = glob.glob(root + "/**/" + n, recursive=True)
        if g:
            return g[0]
    return None


# ═══════════════════════════════ data ═══════════════════════════════════════════════════════════
class TwoPartVols:
    """The two published corpus parts as one (N, 44, 336, 336) array."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, i):
        return self.a[i] if i < self.a.shape[0] else self.b[i - self.a.shape[0]]


def main():
    dev = "cpu"
    ckpt = find("exp085/fold0.pt", "fold0.pt")
    if not ckpt:
        raise SystemExit("!! exp085 fold0.pt not mounted")
    print("[data] ckpt %s (%.0f MB)" % (ckpt, os.path.getsize(ckpt) / 1e6), flush=True)
    obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = {k: v for k, v in obj.items() if k != "state_dict"}
    sd = obj["state_dict"]

    # ⚠️ The corpus parts are NOT both called "part*": the two published datasets ship
    # `all_vols.npy` + `all_ids.npy` and `extra_vols.npy` + `extra_ids.npy`, and they are .npy
    # (15.9 GB + 6.0 GB), not .npz. Matching on "part" silently found nothing.
    va, vb = find("all_vols.npy"), find("extra_vols.npy")
    ia, ib = find("all_ids.npy"), find("extra_ids.npy")
    if not (va and vb and ia and ib):
        print("[data] layout under /kaggle/input:", flush=True)
        for p in sorted(glob.glob("/kaggle/input/*"))[:40]:
            print("   ", p, flush=True)
        raise SystemExit("!! corpus parts not found: need all_vols.npy+all_ids.npy and "
                         "extra_vols.npy+extra_ids.npy")
    vols = TwoPartVols(np.load(va, mmap_mode="r"), np.load(vb, mmap_mode="r"))
    ids = np.concatenate([np.load(ia, allow_pickle=True).astype(str),
                          np.load(ib, allow_pickle=True).astype(str)])
    id2row = {u: i for i, u in enumerate(ids)}
    _s = vols[0]
    print("[data] volumes %s dtype=%s range=[%.3f, %.3f] | ids %d"
          % (vols.shape, _s.dtype, float(np.asarray(_s).min()), float(np.asarray(_s).max()),
             len(ids)), flush=True)

    tcsv = find("train.csv")
    tr = pd.read_csv(tcsv)
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold = tr[tr[LAB].notna().all(axis=1)]
    gold = gold[gold["StudyInstanceUID"].isin(id2row)]
    print("[data] GOLD set = %d studies" % len(gold), flush=True)
    if len(gold) < 20:
        raise SystemExit("!! gold set too small (%d) -- check the mounted labels" % len(gold))

    model = Exp085(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[model] MISSING=%d UNEXPECTED=%d  %s" % (len(missing), len(unexpected),
                                                    (missing[:4], unexpected[:4])), flush=True)
    model.load_state_dict(sd, strict=True)          # must pass; the architecture is proven locally
    print("[model] strict=True load PASS", flush=True)
    model.eval()

    # ── slice POLICIES, all read out of the checkpoint's own config ────────────────────────────
    # The gate run fixed the head (order A, 0.7997 vs 0.68/0.66) and left a 0.106 gap to the author's
    # 0.9062, which can only be the INPUT. This run varies only that. `slice_select band_groups` +
    # `n_group 4` + `group_pick per_study` + `slice_sampling center` says the central depth range is
    # split into 4 bands with 3 slices chosen per band; the gate used the crudest reading (12 slices
    # spread evenly over the whole range) and is kept here as the CONTROL.
    lo, hi = cfg.get("slice_depth_range", (0.2, 0.8))
    n_slice = int(cfg.get("n_slice_cached", 12))
    n_group = int(cfg.get("n_group", 4))
    img = int(cfg.get("img", 224))
    D = vols.shape[1]
    A0, A1 = int(round(lo * D)), int(round(hi * D)) - 1          # inclusive central range
    per = n_slice // n_group                                     # 3 slices per band
    BANDS = []
    for b in range(n_group):
        s = A0 + (len(range(A0, A1 + 1)) * b) // n_group
        e = A0 + (len(range(A0, A1 + 1)) * (b + 1)) // n_group   # exclusive
        BANDS.append((s, e - 1))

    def clip(i, s, e):
        return int(min(max(i, s), e))

    # ── AGGREGATION MODES. Run 2 eliminated the slice policy and the crop scale and left the one
    # structural difference the config names and we had never implemented:
    #     n_group 4 | group_pick per_study | eval_group_agg mean
    # Every experiment so far computed ONE forward over all 12 slices and attended across them.
    # `eval_group_agg: mean` is an aggregation over FOUR GROUPS: the literal reading is that the model
    # runs once per group of 3 slices, giving 12 logits per group, and the prediction is the MEAN of
    # the 4 groups' logits. `slice_pool.depth_mlp` may likewise take the WITHIN-GROUP depth.
    idx = [int(round(v)) for v in np.linspace(A0, A1, n_slice)]
    GROUPS = [[idx[b * per + k] for k in range(per)] for b in range(n_group)]
    print("[data] central %d..%d, %d per group, linspace slices %s" % (A0, A1, per, idx), flush=True)
    for b, g in enumerate(GROUPS):
        print("   group %d %s" % (b, g), flush=True)
    ALL = sorted(set(idx))

    # ── INPUT-PIPELINE VARIANTS. Run 3 settled every knob the checkpoint names -- head order (A),
    # eval_group_agg (mean, with max/median ruled out), n_group/group_pick, slice_depth_range, 224 px,
    # CLS concat mean-patch -- and eliminated the other two (slice_select, crop_mm). It still left
    # 0.087 to the author's 0.9062, so the difference has to be in the CORPUS PREPROCESSING: the thing
    # a checkpoint cannot state and a config field only names.
    #
    #   base              CONTROL: grp_mean_local at crop 272 -- must reproduce run 3's 0.8191
    #   mirror            the MIRRORED slice set [D-1-k]. If the corpus's depth axis runs the other
    #                     way, every slice we feed is the wrong anatomy and the depth encoding is
    #                     inverted; this is the largest single-effect hypothesis left.
    #   pct_clip          per-volume 1st-99th percentile window instead of a flat /255. A windowing
    #                     convention is exactly what "the corpus was cached by someone else" would
    #                     change, and it is invisible in the tensor shapes.
    #   pct_clip_mirror   both.
    #
    # ⚠️ `rev_idx` (the same 12 slices in reverse order) is deliberately NOT run: slice attention is
    # permutation-invariant and `depth_of` uses min/max, so it would be a no-op by construction.
    VARIANTS = [("base", False, False), ("mirror", True, False),
                ("pct_clip", False, True), ("pct_clip_mirror", True, True)]
    MIRROR = [D - 1 - k for k in idx]
    ALL = sorted(set(idx) | set(MIRROR))
    print("[data] slices %s | mirrored %s | union %d (embedded once per intensity)" %
          (idx, MIRROR, len(ALL)), flush=True)
    CROPS = [272]
    Y = gold[LAB].values.astype(np.float32)
    P = {(c, v[0]): [] for c in CROPS for v in VARIANTS}

    def depth_of(sel, local):
        a = np.array(sel, dtype=np.float32)
        if local:
            s, e = float(min(sel)), float(max(sel))
            return ((a - s) / max(1e-6, (e - s)))[:, None]
        return ((a - lo * D) / max(1e-6, (hi - lo) * D))[:, None]

    def prep(raw, pct):
        v = np.asarray(raw, dtype=np.float32)
        if pct:
            a, b = float(np.percentile(v, 1.0)), float(np.percentile(v, 99.0))
            v = (np.clip(v, a, b) - a) / max(1e-6, b - a)
        else:
            v = v / 255.0
        return v

    with torch.no_grad():
        for gi, (_, row) in enumerate(gold.iterrows()):
            raw = vols[id2row[row["StudyInstanceUID"]]]
            for name, do_mirror, do_pct in VARIANTS:
                sel = MIRROR if do_mirror else idx
                groups = [[sel[b * per + k] for k in range(per)] for b in range(n_group)]
                v = prep(raw, do_pct)
                for c in CROPS:
                    o = (v.shape[-1] - c) // 2
                    ims = np.stack([cv2.resize(v[k, o:o + c, o:o + c], (img, img),
                                               interpolation=cv2.INTER_LINEAR) for k in ALL])
                    x = torch.from_numpy(ims).unsqueeze(1).repeat(1, 3, 1, 1)
                    tok = model.backbone(((x - model.mean) / model.std))
                    feat = torch.cat([tok[:, 0], tok[:, 1:].mean(1)], -1)      # (N, 1536)
                    pos = {k: i for i, k in enumerate(ALL)}
                    outs = []
                    for g in groups:
                        f = feat[[pos[k] for k in g]].unsqueeze(0)
                        d = torch.from_numpy(depth_of(g, True)).unsqueeze(0)
                        outs.append(model.head(model.slice_pool(f, d)[0], "A")[0])
                    P[(c, name)].append(torch.stack(outs, 0).mean(0).numpy())
            if gi % 10 == 0:
                print("   [%2d/%d]" % (gi, len(gold)), flush=True)

    from sklearn.metrics import roc_auc_score

    def auc_of(pr):
        per = [roc_auc_score(Y[:, j], pr[:, j]) for j in range(len(LAB))
               if len(np.unique(Y[:, j])) > 1]
        return float(np.mean(per)), per

    print("\n" + "=" * 96)
    print("%-7s %s   (author exp085 gold58: 0.9062;  A167 member gate: solo >= 0.90)"
          % ("crop", "  ".join("%-18s" % v[0] for v in VARIANTS)))
    print("-" * 96)
    best = None
    for c in CROPS:
        cells = []
        for v in VARIANTS:
            a, _ = auc_of(np.asarray(P[(c, v[0])]))
            cells.append("%-18.4f" % a)
            if best is None or a > best[0]:
                best = (a, c, v[0])
        print("%-7d %s" % (c, "  ".join(cells)), flush=True)
    print("-" * 96)
    print("BEST: crop=%d variant=%s  macro-AUC=%.4f" % (best[1], best[2], best[0]))
    print("base MUST reproduce run 3's grp_mean_local at crop 272 = 0.8191.")
    print("A big jump on `mirror` means the corpus depth axis is the other way round.")
    print("A big jump on `pct_clip` means the intensity/window convention is the gap.")
    print("No jump on any of them means exp085 needs the author's own preprocessing.")
    json.dump({"variants": [v[0] for v in VARIANTS], "slices": idx, "mirror": MIRROR,
               "results": [{"crop": c, "variant": v[0],
                            "macro_auc": auc_of(np.asarray(P[(c, v[0])]))[0],
                            "per_label": auc_of(np.asarray(P[(c, v[0])]))[1]}
                           for c in CROPS for v in VARIANTS]},
              open("/kaggle/working/exp085_axis.json", "w"), indent=1)
    print("wrote /kaggle/working/exp085_axis.json")


if __name__ == "__main__":
    main()
