# -*- coding: utf-8 -*-
"""A5-lite: train the frontier's A5 architecture on OUR corpus, against our own head, same recipe.

WHY THIS EXISTS (and why it is not the faithful experiment)
    FACTS A149 established that the frontier's A5 model can be rebuilt from their notebook and loads
    their weights strictly.  The remaining question is whether that ARCHITECTURE is what buys their
    0.909 versus our 0.877 -- but answering it faithfully needs their input cache: 6 slots x 16
    slices x 336 px = 10.8 MB per study, about 48 GB for the corpus, which does not exist publicly
    and would take most of a day to build and upload.

    So this runs the cheap decisive test FIRST.  `n_slice` is a CONSTRUCTOR argument (the patch embed
    takes `in_chans = n_slice`), so the same architecture can be trained on the corpus we already
    have, at a smaller slice count.  What that tests is the HEAD -- slot-type tokens, per-segment
    attention, and the `xcodex` gated residual cross-attention pooling -- plus the "slices as
    channels" representation.  What it does NOT test is A5's 16-slice density.

    ⚠️ And it runs BOTH heads in one script, on identical data, folds, epochs, batch size and
    learning rates.  A single arm would prove nothing: `d3f0` already showed that the same backbone
    with our head sits at 0.853, so the comparison has to be paired to mean anything.

THE SLOT MAPPING, AND WHERE IT IS HONEST
    The published corpus is 44 slices per study in five groups -- the frontier's own `_SLOTS44`:
        Sagittal/fluid 12, Sagittal/no-fluid 10, Coronal/fluid 8, Coronal/no-fluid 6, Axial 8
    A5 wants six slots: the same five planes/contrasts, plus Axial WITHOUT fluid, which the corpus
    does not carry.  So slots 0..3 map one-to-one, slot 4 is the corpus's axial group, and slot 5 is
    emitted with visibility 0 -- the mask the model already takes for exactly this case.
    Each real slot contributes `n_slice` slices taken at even spacing from its group.

Usage (built and pushed by scripts/build_a5lite.py):
    --head {a5,raptor}  --fold N --folds 5 --epochs 12 --res 336 --n-slice 6 --bs 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

try:                                    # the builder inlines the model code and strips this import
    from a5_model import Net, ViTSlotToken, N_SLOT_TYPES, MASK_IDX
except ImportError:
    pass

# ---------------------------------------------------------------- corpus / labels / folds
CORPUS_PARTS = [("all_vols.npy", "all_ids.npy", "all_masks.npy"),
                ("extra_vols.npy", "extra_ids.npy", "extra_masks.npy")]
# the frontier's _SLOTS44 groups, in the order the corpus stores them
SLOT_GROUPS = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
               ("Coronal", 0, 6), ("Axial", -1, 8)]

LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
       'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def _find(names):
    for root, _dirs, files in os.walk("/kaggle/input"):
        for n in names:
            if n in files:
                return os.path.join(root, n)
    raise FileNotFoundError("none of %s under /kaggle/input" % (names,))


def open_corpus():
    vols, ids, msks = [], [], []
    for vn, inn, mn in CORPUS_PARTS:
        try:
            vols.append(np.load(_find([vn]), mmap_mode="r"))
            ids.append(np.load(_find([inn]), allow_pickle=True))
            msks.append(np.load(_find([mn])))
        except FileNotFoundError:
            print("[a5lite] corpus part missing:", vn, flush=True)
    if not vols:
        raise SystemExit("no corpus parts attached")
    V = np.concatenate([np.asarray(v) for v in vols], 0)
    I = np.concatenate([np.asarray(i).astype(str) for i in ids], 0)
    M = np.concatenate([np.asarray(m) for m in msks], 0)
    print("[a5lite] corpus: %s vols, %s masks, D=%d" % (V.shape, M.shape, V.shape[1]), flush=True)
    return V, list(I), M


def read_labels():
    """The same hybrid report labels our arms were trained on.

    ⚠️ Found by COLUMNS, not by filename.  The first version carried a list of candidate names and
    died on `FileNotFoundError: none of ['labels.parquet', 'hybrid_report_labels.csv',
    'report_labels_v2.csv']` -- the dataset actually ships `report_labels_v4hybrid.csv`.  A name list
    will always be one rename away from breaking; the 12 label columns are the stable contract.
    """
    import pandas as pd
    cands = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith((".csv", ".parquet")):
                cands.append(os.path.join(root, f))
    best = None
    for p in sorted(cands):
        try:
            df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
        except Exception:                                              # noqa: BLE001
            continue
        hit = [t for t in LAB if t in df.columns]
        if len(hit) > (len(best[1]) if best else 0):
            best = (p, hit, df)
    if best is None or len(best[1]) < 12:
        raise SystemExit("no label file with all 12 findings; saw %s"
                         % [(os.path.basename(p), n) for p, n, _ in
                            [(c, 0, None) for c in cands]][:6])
    p, hit, df = best
    print("[a5lite] labels from %s (%d/%d findings present, %d rows)"
          % (os.path.basename(p), len(hit), len(LAB), len(df)), flush=True)
    y = {}
    for _, r in df.iterrows():
        uid = str(r["StudyInstanceUID"])
        vals = []
        for t in LAB:
            v = r.get(t, np.nan)
            if isinstance(v, str):
                v = {"YES": 1.0, "NO": 0.0}.get(v.strip().upper(), np.nan)
            vals.append(float(v) if v == v else np.nan)
        y[uid] = vals
    have = sum(1 for v in y.values() if any(x == x for x in v))
    print("[a5lite] labels: %d studies, %d with at least one verdict" % (len(y), have), flush=True)
    return y


def _as_fold_map(d, depth=0):
    """Accept both `{uid: fold}` and the wrapped `{"folds": {uid: fold, ...}}`.

    ⚠️ The first version assumed the flat shape and died with
    `TypeError: int() argument must be ... not 'dict'` -- `ppxl16/rsna-knee-cv-folds` ships
    `{"folds": {...}}`.  (I had "checked" this locally and read the INNER dict, because my probe
    printed `d[first_key]` rather than the value type distribution -- which is why the fix below is
    a recursive search plus a printed receipt rather than another guess.)
    """
    if not isinstance(d, dict) or depth > 3:
        return None
    if d and all(isinstance(v, (int, float)) for v in d.values()):
        return {str(k): int(v) for k, v in d.items()}, "flat"
    for k, v in d.items():
        got = _as_fold_map(v, depth + 1)
        if got:
            return got[0], "under %r" % k
    return None


def read_folds():
    p = _find(["fold_file.json", "v52_oof.csv"])
    if p.endswith(".json"):
        got = _as_fold_map(json.loads(open(p).read()))
        if not got:
            raise SystemExit("could not find a uid->fold map in %s" % p)
        fold, how = got
    else:
        import pandas as pd
        df = pd.read_csv(p)
        fold, how = {str(u): int(f) for u, f in
                     zip(df["StudyInstanceUID"], df["fold"])}, "v52_oof.csv"
    print("[a5lite] folds from %s (%s): %d studies, fold counts %s"
          % (os.path.basename(p), how, len(fold),
             {k: sum(1 for v in fold.values() if v == k) for k in sorted(set(fold.values()))}),
          flush=True)
    return fold


# ---------------------------------------------------------------- the A5-lite input
def a5_study(vol, n_slice, res, rng=None):
    """(D,H,W) uint8 -> (N_SLOT, n_slice, res, res) float32 in [0,1], plus a visibility mask.

    Slices are taken inside each corpus group.  With `rng` (training) the window inside the group is
    shifted at random -- the SAME KIND of augmentation the raptor arm gets from random window
    centres, so the pair stays symmetric.  Slot 5 (Axial without fluid) does not exist in the corpus
    and is returned masked -- see the module docstring.
    """
    D, H, W = vol.shape
    out = np.zeros((N_SLOT_TYPES, n_slice, res, res), np.float32)
    vis = np.zeros(N_SLOT_TYPES, np.float32)
    off = 0
    for s, (_plane, _fs, count) in enumerate(SLOT_GROUPS):
        if rng is not None and count > n_slice:
            start = int(rng.integers(0, count - n_slice + 1))
            idx = [off + start + i for i in range(n_slice)]
        else:
            idx = [off + int(round(t)) for t in np.linspace(0, count - 1, n_slice)]
        idx = [min(max(i, 0), D - 1) for i in idx]
        block = vol[idx]                                        # (n_slice, H, W) uint8
        if (H, W) != (res, res):
            t = torch.from_numpy(block).unsqueeze(1).float()     # (n,H,1,W)
            t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)
            block = t.squeeze(1).numpy()
        out[s] = block.astype(np.float32) / 255.0
        vis[s] = 1.0
        off += count
    return out, vis


def collate_a5(studies, res, n_slice):
    """-> (T,C,H,W), slot ids, study index, B, visibility mask -- the contract Net.forward wants."""
    ims, slots, sidx, vms = [], [], [], []
    for b, (arr, vis) in enumerate(studies):
        for s in range(N_SLOT_TYPES):
            ims.append(arr[s]); slots.append(s + 1); sidx.append(b)
        vms.append(vis)
    return (torch.from_numpy(np.stack(ims)), torch.tensor(slots, dtype=torch.long),
            torch.tensor(sidx, dtype=torch.long), len(studies), torch.from_numpy(np.stack(vms)))


# ---------------------------------------------------------------- our head, for the control arm
class RaptorClassifier(nn.Module):
    """Copied from our trainer (`kernel-raptor-train`) so the control arm is the same object ours is.

    It takes (B, K, 3, H, W) windows and pools with per-finding attention over K.
    """

    def __init__(self, backbone, F_dim=384, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def forward(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(B, K, -1)
        h = self.norm(f)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb, None


def build_model(head, backbone_name, res, n_slice, n_slot_for_raptor):
    """Both arms get the SAME backbone and the SAME pretrained init -- only the head differs."""
    if head == "a5":
        enc = timm.create_model(backbone_name, pretrained=True, num_classes=0,
                                in_chans=n_slice, img_size=res)
        enc = ViTSlotToken(enc, N_SLOT_TYPES)
        return Net(enc, "token", 0, "xcodex", stem="native", n_slice=n_slice)
    enc = timm.create_model(backbone_name, pretrained=True, num_classes=0,
                            in_chans=3, img_size=res, global_pool="token", dynamic_img_size=True)
    return RaptorClassifier(enc, F_dim=enc.num_features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["a5", "raptor"], required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--n-slice", type=int, default=6)
    ap.add_argument("--k", type=int, default=12, help="windows per study for the raptor control arm")
    ap.add_argument("--bb-lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--backbone", default="vit_small_patch16_dinov3.lvd1689m")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--aug", type=int, default=1,
                    help="1 (default): both arms get a random sampling offset plus the real "
                         "trainer's medical-safe intensity jitter. 0: the degraded setting of the "
                         "first paired run (FACTS A153), kept so that comparison stays reproducible.")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tag = a.tag or ("%s_f%d" % (a.head, a.fold))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[a5lite] head=%s fold=%d res=%d n_slice=%d epochs=%d bs=%d dev=%s"
          % (a.head, a.fold, a.res, a.n_slice, a.epochs, a.bs, dev), flush=True)

    V, ids, M = open_corpus()
    labels = read_labels()
    fold = read_folds()
    pos = {u: i for i, u in enumerate(ids)}

    def usable(u):
        return u in pos and u in labels and u in fold and any(x == x for x in labels[u])

    tr = [u for u in ids if usable(u) and fold[u] != a.fold]
    va = [u for u in ids if usable(u) and fold[u] == a.fold]
    if a.limit:
        tr, va = tr[:a.limit], va[:max(64, a.limit // 4)]
    print("[a5lite] train %d  held-out %d" % (len(tr), len(va)), flush=True)

    model = build_model(a.head, a.backbone, a.res, a.n_slice, a.k).to(dev)
    bb = [p for n, p in model.named_parameters() if n.startswith(("enc.", "backbone."))]
    hd = [p for n, p in model.named_parameters() if not n.startswith(("enc.", "backbone."))]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.bb_lr}, {"params": hd, "lr": a.head_lr}],
                            weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.bb_lr * 4, a.head_lr], total_steps=max(1, a.epochs * max(1, len(tr) // a.bs)))
    lossf = nn.BCEWithLogitsLoss()

    def batches(seq):
        for i in range(0, len(seq), a.bs):
            yield seq[i:i + a.bs]

    def make_input(uids, rng=None):
        """`rng` present => TRAINING => both arms get the same KIND of augmentation.

        Why this exists (FACTS A153): the first paired run had NO augmentation on either arm, so the
        control scored 0.7366 while the real trainer's version of the same head scores 0.8530.  The
        architecture's +0.0575 was therefore measured at a degraded operating point, and this project
        has twice measured such gains shrinking as the base strengthens (A109, A112).  So both arms
        now get (i) a random sampling offset and (ii) the real trainer's "medical-safe" intensity
        jitter -- NO flips, because laterality is signal.  Identical treatment, so the pair stays
        architecture-only.
        """
        if a.head == "a5":
            studies = [a5_study(np.asarray(V[pos[u]]), a.n_slice, a.res, rng) for u in uids]
            out = collate_a5(studies, a.res, a.n_slice)
            if rng is not None:
                out = (jitter(out[0], rng),) + out[1:]
            return out
        wins = []
        for u in uids:
            vol = np.asarray(V[pos[u]]); D = vol.shape[0]
            if rng is not None:
                cs = np.sort(rng.choice(np.arange(1, D - 1), size=a.k, replace=False))
            else:
                cs = np.linspace(1, D - 2, a.k).round().astype(int)
            tri = np.stack([np.stack([vol[c - 1], vol[c], vol[c + 1]], 0) for c in cs], 0)
            t = torch.from_numpy(tri).float() / 255.0
            if (t.shape[-2], t.shape[-1]) != (a.res, a.res):
                t = F.interpolate(t.flatten(0, 1), size=(a.res, a.res),
                                  mode="bilinear", align_corners=False).view(t.shape[0], 3, a.res, a.res)
            wins.append(t)
        x = torch.stack(wins, 0)
        if rng is not None:
            x = jitter(x, rng)
        return x, None, None, len(uids), None

    def jitter(x, rng):
        """The real trainer's light medical-safe aug: mild multiplicative + additive intensity jitter.

        Applied identically to BOTH arms, so it cannot favour either one.
        """
        gain = torch.from_numpy(rng.uniform(0.9, 1.1, size=(x.shape[0],) + (1,) * (x.dim() - 1))).float()
        bias = torch.from_numpy(rng.normal(0.0, 0.02, size=(x.shape[0],) + (1,) * (x.dim() - 1))).float()
        return torch.clamp(x * gain + bias, 0.0, 1.0)

    def forward_logits(inp):
        if a.head == "a5":
            im, slot, sidx, B, vm = inp
            out = model(im.to(dev), slot.to(dev), torch.zeros(B, 0, device=dev), sidx.to(dev), B)
            return out[0] if isinstance(out, (tuple, list)) else out
        x = inp[0].to(dev)
        out = model(x)
        return out[0] if isinstance(out, (tuple, list)) else out

    def target(uids):
        return torch.tensor([[labels[u][j] for j in range(12)] for u in uids], dtype=torch.float32)

    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        rng = np.random.default_rng(1000 + ep)
        order = list(rng.permutation(len(tr)))
        for chunk in batches(order):
            uids = [tr[i] for i in chunk]
            y = target(uids).to(dev)
            m = ~torch.isnan(y)
            if not bool(m.any()):
                continue
            opt.zero_grad(set_to_none=True)
            logits = forward_logits(make_input(uids, rng if a.aug else None))
            loss = lossf(logits[m], y[m].nan_to_num(0.0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += float(loss); nb += 1
        print("[a5lite] ep%d loss %.4f  (%.1fs, %d steps)" % (ep, tot / max(nb, 1),
                                                              time.time() - t0, nb), flush=True)

    model.eval()
    P = np.full((len(va), 12), np.nan, np.float32)
    with torch.no_grad():
        for i, chunk in enumerate(batches(list(range(len(va))))):
            uids = [va[j] for j in chunk]
            out = torch.sigmoid(forward_logits(make_input(uids)))
            P[i * a.bs:i * a.bs + len(chunk)] = out.float().cpu().numpy()
    out = "/kaggle/working/raptor_oof_%s_fold%d.npz" % (tag, a.fold)
    np.savez_compressed(out, ids=np.array(va, dtype=object), pred=P,
                        fold=np.full(len(va), a.fold, np.int8), nfolds=a.folds)
    print("[a5lite] wrote %s  (%d studies)" % (out, len(va)), flush=True)


main()
