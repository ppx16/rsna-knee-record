# -*- coding: utf-8 -*-
"""3D volumetric ARM (CUDA sibling of the TPU kernel) -- same model, same gate, rented 4090.

Generated from kernels/kernel-tpu-3darm/kernel_main.py by work/_make_cuda_arm.py; the ONLY
differences are the device and the optimizer step. Everything else is deliberately identical.

WHY (FACTS A160). The member gate's base is A5(DINOv3) + public20(DINOv2) + Rad(ResNet50): it contains NO
CoAtNet/Raptor, so cn0's +0.0028 was largely 'filling in a whole missing family' rather than 'this member is
strong'. The corrected criterion is therefore not a STRONGER arm but an arm whose inductive bias differs from
{DINOv2, DINOv3, ResNet50, CoAtNet}. Every family in the stack is 2D slots, and A163 showed that more
same-family members end in redundancy. A real 3D volume is the unexplored axis.

GATE (pre-registered, docs/PREREG-2026-10-06q-3d-smoke.md section 4): solo macro-AUC must be >= 0.90 measured
on a LARGE-n instrument -- gold-58's CI is +/-0.021 and cannot judge that line -- against msft2's 0.8847 on the
same instrument. This kernel therefore trains on folds != held-out and reports macro-AUC on the held-out fold,
writing the OOF so the gate can be applied without rerunning anything.

TWO THINGS ALREADY MEASURED, AND THIS FILE KEEPS THEM:
  * Conv3d + BN3d + adaptive pool + BCE compile and step on PJRT (smoke v2, PASS, state/smoke3d_result.json).
  * The label table is SOFT (0.25 / 0.975 ...) with no __verdict column. The original parser turned every cell
    into nan, the mask went all-False, an empty tensor reached BCEWithLogitsLoss, and all 12 steps printed
    `loss nan`. The parser below is the fixed one; the same fix printed loss 0.6923 -> 0.6325 on the smoke.

IT IS DELIBERATELY NOT CLEVER. No augmentation, no scheduler, no EMA. The question is whether a real volume
carries signal, and anything else added now would make a negative answer uninterpretable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
for _v, _p in (("HF_HOME", "/kaggle/working/.hf"),
               ("HUGGINGFACE_HUB_CACHE", "/kaggle/working/.hf/hub"),
               ("TORCH_HOME", "/kaggle/working/.torch"),
               ("XDG_CACHE_HOME", "/kaggle/working/.cache")):
    os.environ[_v] = _p
    os.makedirs(_p, exist_ok=True)

import numpy as np
import pathlib                                            # noqa: E402
import torch                                                  # noqa: E402
import torch.nn as nn                                         # noqa: E402

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def log(*a):
    print(*a, flush=True)


def find_one(pattern, what):
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        raise SystemExit('!! %s not found (%s)' % (what, pattern))
    return sorted(hits, key=len)[0]


def load_labels(path):
    """soft floats, the case that actually occurs here -- see the module docstring."""
    import csv
    out = {}
    with open(path, newline='', encoding='utf-8-sig', errors='replace') as f:
        for r in csv.DictReader(f):
            uid = r.get('StudyInstanceUID') or r.get('study_uid') or r.get('id')
            if not uid:
                continue
            y = []
            for t in TARGETS:
                raw = (r.get(t + '__verdict') or r.get(t) or '').strip()
                if not raw:
                    y.append(float('nan'))
                    continue
                up = raw.upper()
                if up in ('YES', 'TRUE'):
                    y.append(1.0)
                elif up in ('NO', 'FALSE'):
                    y.append(0.0)
                else:
                    try:
                        y.append(float(raw))
                    except ValueError:
                        y.append(float('nan'))
            out[uid] = np.array(y, dtype=np.float32)
    return out


def load_folds(path):
    d = json.loads(open(path, encoding='utf-8').read())
    folds = d.get('folds') if isinstance(d, dict) else None
    if not isinstance(folds, dict):
        raise SystemExit('!! fold file has no {"folds": {uid: fold}} mapping')
    return {k: int(v) for k, v in folds.items()}


class Vol3D(nn.Module):
    def __init__(self, n=12, width=48):
        super().__init__()
        w = width
        self.f = nn.Sequential(
            nn.Conv3d(1, w, 3, stride=(1, 2, 2), padding=1), nn.BatchNorm3d(w), nn.ReLU(inplace=True),
            nn.Conv3d(w, w * 2, 3, stride=(2, 2, 2), padding=1), nn.BatchNorm3d(w * 2), nn.ReLU(inplace=True),
            nn.Conv3d(w * 2, w * 4, 3, stride=(2, 2, 2), padding=1), nn.BatchNorm3d(w * 4), nn.ReLU(inplace=True),
            nn.Conv3d(w * 4, w * 4, 3, stride=(2, 2, 2), padding=1), nn.BatchNorm3d(w * 4), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(w * 4, n))

    def forward(self, x):
        return self.head(self.f(x))


def build_x(vol_of, uids, D, HW):
    xs = []
    for u in uids:
        v = np.asarray(vol_of(u))
        n = v.shape[0]
        idx = np.linspace(0, n - 1, D).round().astype(int)
        sub = v[idx]
        s = max(1, sub.shape[-1] // HW)
        sub = sub[:, ::s, ::s]
        t = torch.from_numpy(np.ascontiguousarray(sub)).float().unsqueeze(0)
        t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(D, HW, HW),
                                            mode='trilinear', align_corners=False)[0]
        xs.append(t / 255.0)
    return torch.stack(xs)


def macro_auc(y, p):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for j in range(y.shape[1]):
        col = y[:, j]
        m = np.isfinite(col)
        if m.sum() < 10:
            continue
        # BINARISE: the table carries SOFT probabilities and roc_auc_score refuses continuous y_true.
        # Convention stated so the bar cannot drift silently: positive when the label exceeds 0.5.
        b = (col[m] > 0.5).astype(np.int8)
        if b.min() == b.max():
            continue
        aucs.append(roc_auc_score(b, p[m, j]))
    return (float(np.mean(aucs)) if aucs else float('nan')), aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--held-fold', type=int, default=0)
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--depth', type=int, default=32)
    ap.add_argument('--hw', type=int, default=128)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval-every', type=int, default=300)
    ap.add_argument('--max-train', type=int, default=0, help='0 = all studies outside the held fold')
    a = ap.parse_args()

    # ---- CUDA (this copy) instead of PJRT (the Kaggle/TPU sibling) ----
    if not torch.cuda.is_available():
        raise SystemExit('!! no CUDA device -- this copy is for the rented 4090')
    dev = torch.device('cuda')
    log('[3darm] cuda %s ; %s' % (torch.version.cuda, torch.cuda.get_device_name(0)))

    vpath = find_one('/kaggle/input/**/all_vols.npy', 'corpus volumes')
    ipath = find_one('/kaggle/input/**/all_ids.npy', 'corpus ids')
    epath = None
    eipath = None
    try:
        epath = find_one('/kaggle/input/**/extra_vols.npy', 'corpus-ext volumes')
        eipath = find_one('/kaggle/input/**/extra_ids.npy', 'corpus-ext ids')
    except SystemExit:
        log('[3darm] NOTE: corpus-ext not mounted -- running on part 1 only (3200 studies)')
    lpath = find_one('/kaggle/input/**/report_labels_v4hybrid.csv', 'soft labels')
    fpath = find_one('/kaggle/input/**/fold_file.json', 'fold definitions')
    log('[3darm] corpus %s | labels %s | folds %s'
        % (os.path.basename(vpath), os.path.basename(lpath), os.path.basename(fpath)))

    # ⚠️ TWO PARTS. FACTS A56: the public corpus is knee-raptor-corpus (3200) + knee-raptor-corpus-ext
    # (1207) = the full 4407. Reading only the first silently shrinks every fold, and nothing fails --
    # mounting a dataset is not reading it.
    PARTS = []
    for _vp, _ip in ((vpath, ipath), (epath, eipath)):
        if not _vp:
            continue
        _v = np.load(_vp, mmap_mode='r')
        _ids = ([str(u) for u in np.load(_ip, allow_pickle=True)] if _ip
                else [str(i) for i in range(_v.shape[0])])
        PARTS.append((_v, {u: i for i, u in enumerate(_ids)}))
        log('[3darm] part %s: %s studies' % (pathlib.Path(_vp).name, len(_ids)))
    ids = [u for _v, _m in PARTS for u in _m]

    def _vol_of(u):
        for _v, _m in PARTS:
            if u in _m:
                return _v[_m[u]]
        raise KeyError(u)
    lab = load_labels(lpath)
    folds = load_folds(fpath)
    log('[3darm] studies %d across %d part(s) | labels %d | folds %d'
        % (len(ids), len(PARTS), len(lab), len(folds)))

    # only studies that have BOTH an id in the corpus and a fold assignment and at least one label
    usable = [u for u in ids if u in folds and u in lab and np.isfinite(lab[u]).any()]
    val = [u for u in usable if folds[u] == a.held_fold]
    tr = [u for u in usable if folds[u] != a.held_fold]
    if a.max_train:
        tr = tr[:a.max_train]
    log('[3darm] train %d | held-out fold %d: %d' % (len(tr), a.held_fold, len(val)))
    if len(val) < 50:
        raise SystemExit('!! held-out fold too small (%d) -- folds and corpus ids may not share a format' % len(val))

    model = Vol3D(12, a.width).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    log('[3darm] Conv3d width=%d, %.2fM params, input (1,%d,%d,%d)'
        % (a.width, nparam / 1e6, a.depth, a.hw, a.hw))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)

    def predict(uids):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(uids), a.bs):
                chunk = uids[i:i + a.bs]
                X = build_x(_vol_of, chunk, a.depth, a.hw).to(dev)
                outs.append(torch.sigmoid(model(X)).cpu().numpy())
        model.train()
        return np.concatenate(outs, 0) if outs else np.zeros((0, 12), np.float32)

    Yval = np.stack([lab[u] for u in val]).astype(np.float32)
    best = (-1.0, None)
    model.train()
    t0 = time.time()
    for step in range(1, a.steps + 1):
        uids = [tr[int(i)] for i in rng.integers(0, len(tr), size=a.bs)]
        X = build_x(_vol_of, uids, a.depth, a.hw).to(dev)
        Y = torch.from_numpy(np.stack([lab[u] for u in uids]).astype(np.float32)).to(dev)
        m = ~torch.isnan(Y)
        if int(m.sum()) == 0:
            continue
        opt.zero_grad(set_to_none=True)
        out = model(X)
        loss = lossf(out[m], torch.nan_to_num(Y[m]))
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            log('[3darm] step %4d/%d  loss %.4f  %.1fs' % (step, a.steps, float(loss), time.time() - t0))
        if step % a.eval_every == 0 or step == a.steps:
            P = predict(val)
            auc, per = macro_auc(Yval, P)
            log('[3darm] HELD-OUT fold %d macro-AUC %.4f  (n=%d)' % (a.held_fold, auc, len(val)))
            if auc > best[0]:
                best = (auc, P.copy())
                np.save('/kaggle/working/oof_3darm_fold%d.npy' % a.held_fold, P)
                np.save('/kaggle/working/oof_3darm_fold%d_ids.npy' % a.held_fold, np.array(val))
                torch.save(model.state_dict(), '/kaggle/working/ckpt_3darm_fold%d.pt' % a.held_fold)

    auc, per = macro_auc(Yval, best[1] if best[1] is not None else predict(val))
    res = {'held_fold': a.held_fold, 'macro_auc': round(float(auc), 4),
           'n_val': len(val), 'n_train': len(tr), 'params_m': round(nparam / 1e6, 2),
           'width': a.width, 'depth': a.depth, 'hw': a.hw, 'steps': a.steps,
           'seconds': round(time.time() - t0, 1),
           'reference_msft2_same_convention': 0.8847,
           'gate': 'solo >= 0.90 on this LARGE-n instrument (gold-58 cannot judge it: CI +/-0.021)'}
    json.dump(res, open('/kaggle/working/arm3d.json', 'w'), indent=2)
    log('[3darm] FINAL held-out macro-AUC %.4f (best %.4f)' % (auc, best[0]))
    log('[3darm] %s' % ('GATE PASSED -- beats the 0.90 member bar' if auc >= 0.90 else
                        'below the 0.90 bar -- record it honestly, do NOT wire it as a member'))
    log('[3darm] reference: msft2 = 0.8847 on this same instrument')
    return 0


if __name__ == '__main__':
    sys.exit(main())
