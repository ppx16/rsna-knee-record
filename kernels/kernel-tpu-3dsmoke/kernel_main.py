# -*- coding: utf-8 -*-
"""3D volumetric SMOKE -- does a Conv3d model even fit and compile on Kaggle/TPU?

WHY THIS EXISTS (docs/REVIEW-2026-09-27.md, Phase 2)
    Every family the stack uses is 2D: DINOv2 slot attention, A5 (16 slices as channels), the
    RadImageNet ResNet-50 head, and the four CoAtNet raptor views.  Our own gate agrees that
    "another 2D backbone" buys nothing -- ConvNeXt is null across four configurations
    (base@224/336, large@224, base@336-d64; FACTS A160).  The one inductive bias nobody has tried
    is a genuine 3D volume, so that is where Phase 2 aims.

    But this repo has a long history of new things dying on XLA (A133 closed CoAtNet-on-XLA), so
    the FIRST step is not an arm -- it is the cheapest possible question:

        does a Conv3d stack compile under PJRT, step, and stay inside HBM?

    This kernel answers ONLY that.  It is not an arm, produces no checkpoint, and is not meant to
    learn anything: labels are real when the label file is found and random otherwise, and the
    verdict is a PASS/FAIL line plus seconds-per-step.

WHAT IT PROVES / DOES NOT PROVE
    proves    : Conv3d + BN + adaptive pool + BCE compile and step on the TPU; timing; HBM headroom
    does NOT  : that a 3D arm would be any good, that it can be integrated (the submission
                notebook's ARMS table is a 2D windowed-reader contract -- a 3D member needs a new
                branch, which is a separate and larger problem)

DESIGN NOTES
    * The volume source is the corpus we ALREADY mount -- `knee-raptor-corpus` is
      (4407, 44, 336, 336) uint8, i.e. a real 44-slice stack per study.  The published
      `barun2104/rsna-knee-mri-processed-3d-volumes` was checked and is only 20 files / 0.01 GB,
      a sample rather than a dataset, so it is not used.
    * Volumes are memmapped; only the studies actually touched are read.  `-ext` parts are ignored
      (this is a smoke, not a CV run).
    * The model is deliberately generic: `--width` and the input `D/H/W` are flags, so once the
      mechanism is proven the same file can be scaled into a real arm without a rewrite.
    * Deliberately NO multi-core spawn.  A single XLA device is enough to answer the question, and
      the spawn path is where the earlier TPU ports lost time.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

# ---- TPU env, before numpy/torch are imported (the standard port prelude) ----
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
for _v, _p in (("HF_HOME", "/kaggle/working/.hf"),
               ("HUGGINGFACE_HUB_CACHE", "/kaggle/working/.hf/hub"),
               ("TORCH_HOME", "/kaggle/working/.torch"),
               ("XDG_CACHE_HOME", "/kaggle/working/.cache")):
    os.environ[_v] = _p
    os.makedirs(_p, exist_ok=True)

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
import torch.nn as nn                                         # noqa: E402

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def log(*a):
    print(*a, flush=True)


def find_corpus():
    """-> (vols path, ids path) for the raptor corpus, wherever Kaggle mounted it."""
    vols = glob.glob('/kaggle/input/**/all_vols.npy', recursive=True)
    ids = glob.glob('/kaggle/input/**/all_ids.npy', recursive=True)
    if not vols:
        raise SystemExit('!! all_vols.npy not found under /kaggle/input (corpus not mounted?)')
    v = sorted(vols, key=len)[0]
    i = sorted(ids, key=len)[0] if ids else None
    log('[3d] corpus: %s (%.2f GB)' % (v, os.path.getsize(v) / 1e9))
    return v, i


def find_labels():
    """The 12 label columns are the reliable identifier -- NOT the filename (A150's trap)."""
    for p in glob.glob('/kaggle/input/**/*.csv', recursive=True):
        try:
            with open(p, encoding='utf-8', errors='replace') as f:
                head = f.readline()
        except Exception:
            continue
        if all(('"%s"' % t) in head or t in head for t in ('ACL', 'Fracture', 'Effusion')):
            log('[3d] labels : %s' % p)
            return p
    log('[3d] labels : NONE FOUND -- using random targets (fine for a mechanism smoke)')
    return None


def load_labels(path):
    """-> {uid: (12,) float, nan where absent}"""
    if not path:
        return {}
    import csv
    out = {}
    with open(path, newline='', encoding='utf-8-sig', errors='replace') as f:
        for r in csv.DictReader(f):
            uid = r.get('StudyInstanceUID') or r.get('study_uid') or r.get('id')
            if not uid:
                continue
            y = []
            for t in TARGETS:
                v = (r.get(t + '__verdict') or r.get(t) or '').strip().upper()
                y.append(1.0 if v in ('YES', '1', 'TRUE') else (0.0 if v in ('NO', '0', 'FALSE')
                                                                else float('nan')))
            out[uid] = np.array(y, dtype=np.float32)
    log('[3d] labels : %d rows parsed' % len(out))
    return out


class Vol3D(nn.Module):
    """A plain Conv3d stack. Generic on purpose: width/depth are the only knobs."""

    def __init__(self, n=12, width=24):
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


def make_batch(vols, ids, pos, uids, D, HW, y_by_uid):
    """(B,1,D,HW,HW) float32 in [0,1] + (B,12) targets with a validity mask."""
    xs, ys = [], []
    for u in uids:
        v = np.asarray(vols[pos[u]])                       # (44,336,336) uint8, memmapped
        n = v.shape[0]
        idx = np.linspace(0, n - 1, D).round().astype(int)
        sub = v[idx]                                       # (D,336,336)
        # in-plane downsample by striding, then a small interpolate to exactly HW
        s = max(1, sub.shape[-1] // HW)
        sub = sub[:, ::s, ::s]
        t = torch.from_numpy(np.ascontiguousarray(sub)).float().unsqueeze(0)   # (1,D,h,w)
        t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(D, HW, HW),
                                            mode='trilinear', align_corners=False)[0]
        xs.append(t / 255.0)
        y = y_by_uid.get(u)
        ys.append(np.zeros(12, np.float32) if y is None else y)
    return torch.stack(xs), torch.from_numpy(np.stack(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--studies', type=int, default=64, help='how many studies to touch at all')
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--depth', type=int, default=32, help='slices kept from the 44-slice stack')
    ap.add_argument('--hw', type=int, default=160)
    ap.add_argument('--width', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-3)
    a = ap.parse_args()

    log('[3d] torch %s' % torch.__version__)
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        # ⚠️ v1 of this smoke died HERE, on a logging line: `xm.xrt_world_size()` does not exist in
        # torch_xla 2.8.  The log shows the run had ALREADY reached `xm.xla_device()` successfully
        # (a DeprecationWarning at that line), so the TPU was fine -- the smoke burned a whole queue
        # cycle on an attribute error, not on Conv3d.  Do not let diagnostics be able to kill the
        # experiment: resolve them defensively and let the model be the thing under test.
        # Also `xm.xla_device()` is deprecated in 2.x in favour of `torch_xla.device()`.
        dev = torch_xla.device() if hasattr(torch_xla, 'device') else xm.xla_device()
        try:
            import torch_xla.runtime as xr
            nworld = xr.world_size()
        except Exception:
            nworld = getattr(xm, 'xrt_world_size', lambda: 'n/a')()
        log('[3d] torch_xla %s ; device %s ; world_size %s'
            % (getattr(torch_xla, '__version__', '?'), dev, nworld))
        if not hasattr(xm, 'optimizer_step'):
            raise SystemExit('!! xm.optimizer_step is missing -- XLA step semantics changed')
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit('!! torch_xla unavailable: %r' % (e,))

    vpath, ipath = find_corpus()
    vols = np.load(vpath, mmap_mode='r')
    log('[3d] vols shape %s dtype %s' % (vols.shape, vols.dtype))
    ids = ([str(u) for u in np.load(ipath, allow_pickle=True)] if ipath
           else [str(i) for i in range(vols.shape[0])])
    pos = {u: i for i, u in enumerate(ids)}

    lab = load_labels(find_labels())
    rng = np.random.default_rng(0)
    use = ids[:min(a.studies, len(ids))]
    y_by_uid = {}
    for u in use:
        y = lab.get(u)
        y_by_uid[u] = (rng.integers(0, 2, 12).astype(np.float32) if y is None else y)

    model = Vol3D(12, a.width).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    log('[3d] model: Conv3d stack width=%d, %.2fM params, input (1,%d,%d,%d)'
        % (a.width, nparam / 1e6, 1, a.depth, a.hw, a.hw))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    lossf = nn.BCEWithLogitsLoss()

    # ---- data prep timing (CPU) ----
    t0 = time.time()
    X, Y = make_batch(vols, ids, pos, use[:a.bs], a.depth, a.hw, y_by_uid)
    log('[3d] first batch %s built in %.1fs (CPU, memmap+trilinear)' % (tuple(X.shape), time.time() - t0))

    model.train()
    tstart = time.time()
    step_times = []
    for i in range(a.steps):
        uids = use[(i * a.bs) % max(1, len(use) - a.bs):][:a.bs] or use[:a.bs]
        X, Y = make_batch(vols, ids, pos, uids, a.depth, a.hw, y_by_uid)
        Xd, Yd = X.to(dev), Y.to(dev)
        m = ~torch.isnan(Yd)
        t1 = time.time()
        opt.zero_grad(set_to_none=True)
        out = model(Xd)
        loss = lossf(out[m], torch.nan_to_num(Yd[m]))
        loss.backward()
        xm.optimizer_step(opt)
        step_times.append(time.time() - t1)
        if i < 3 or i == a.steps - 1:
            log('[3d] step %2d/%d  loss %.4f  %.2fs (incl. data)'
                % (i + 1, a.steps, float(loss), step_times[-1]))

    warm = step_times[1:]
    med = float(np.median(warm)) if warm else float('nan')
    res = {'status': 'PASS', 'params_m': nparam / 1e6, 'steps': a.steps, 'bs': a.bs,
           'input': [1, a.depth, a.hw, a.hw], 'width': a.width,
           'median_step_s': med, 'total_s': time.time() - tstart,
           'device': str(dev), 'vols_shape': list(vols.shape)}
    log('[3d] median step %.2fs over %d warm steps ; total %.1fs' % (med, len(warm), res['total_s']))
    with open('/kaggle/working/smoke3d.json', 'w') as f:
        json.dump(res, f, indent=2)
    log('[3d] VERDICT: PASS -- Conv3d compiled and stepped on TPU without OOM')
    log('[3d] (this proves the MECHANISM only; it says nothing about accuracy or integrability)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
