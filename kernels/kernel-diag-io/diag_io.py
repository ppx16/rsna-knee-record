"""CPU-only diagnostic: is a training epoch I/O-bound or CPU-bound on the raptor corpus?

WHY (HANDOFF 82.3)
    The CV fold kernels take 1548 s/epoch on 3479 studies (0.445 s/study) while the identical
    production script took 864 s/epoch on 4349 studies (0.199 s/study) in another session.
    Same script, same machine_shape, no DataParallel anywhere -- 2.26x unexplained.  That
    factor is worth ~3 GPU-hours per experiment, so it is worth measuring.

    The dataset class reads `vol = self.vols[row]` for EVERY item -- the whole (44,336,336)
    uint8 volume, ~4.9 MB -- and then keeps only 12 windows of 3 slices.  So the hypothesis is
    that the epoch is dominated by memmap reads from the dataset mount, and the 2.26x is
    session-to-session mount throughput.  Rival hypothesis: CPU cost of the bilinear resizes.

    This kernel decides between them WITHOUT using a GPU slot.

OUTPUT
    * volume shape/dtype, corpus totals
    * single-thread and multi-thread sequential-read MB/s over the volume rows
    * simulated __getitem__ cost (read volume + 12 window extracts + resize to 224)
    * valid-slice-count distribution, overall AND split by the CV fold assignment, which also
      tests whether fold 0 is systematically made of longer series (the other candidate
      explanation for the 2.26x)
"""
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def find(name):
    hits = sorted(glob.glob("/kaggle/input/**/" + name, recursive=True))
    return hits[0] if hits else None


def mb(x):
    return x / 1e6


def timed(fn, reps=1):
    t = time.time()
    out = None
    for _ in range(reps):
        out = fn()
    return (time.time() - t) / reps, out


def main():
    t_start = time.time()
    print("=== files ===", flush=True)
    paths = {n: find(n) for n in ("all_vols.npy", "all_masks.npy", "all_ids.npy",
                                  "extra_vols.npy", "extra_masks.npy", "fold_file.json")}
    for k, v in paths.items():
        print("  %-16s %s" % (k, v), flush=True)
    if not paths["all_vols.npy"]:
        raise SystemExit("corpus not mounted")

    t, vols = timed(lambda: np.load(paths["all_vols.npy"], mmap_mode="r"))
    print("\n=== memmap open: %.3f s ===" % t, flush=True)
    print("  all_vols  shape %s dtype %s  = %.2f GB" % (vols.shape, vols.dtype, mb(vols.nbytes) / 1000))
    N, D, H, W = vols.shape
    row_bytes = D * H * W
    print("  per-study row = %.2f MB ; %d studies" % (mb(row_bytes), N), flush=True)

    masks = np.load(paths["all_masks.npy"], mmap_mode="r") if paths["all_masks.npy"] else None
    if masks is not None:
        print("  all_masks shape %s dtype %s" % (masks.shape, masks.dtype), flush=True)

    # ---------------- raw sequential read throughput ----------------
    print("\n=== raw row reads (warmup 3, then 30 rows x 3 rounds) ===", flush=True)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, N, 30)
    for i in idx[:3]:
        _ = vols[i].sum()
    for label, nthread in (("1 thread", 1), ("4 threads", 4), ("8 threads", 8)):
        def task():
            if nthread == 1:
                s = 0
                for i in idx:
                    s += int(vols[i][0, 0, 0])
                return s
            with ThreadPoolExecutor(nthread) as ex:
                return sum(ex.map(lambda i: int(vols[i][0, 0, 0]), idx))
        t, _ = timed(task, reps=3)
        print("  %-10s %6.2f s for %d rows -> %7.1f MB/s  (%.4f s/study)"
              % (label, t, len(idx), mb(row_bytes * len(idx)) / t, t / len(idx)), flush=True)

    # ---------------- simulated __getitem__ ----------------
    print("\n=== simulated __getitem__ (read volume + 12 windows of 3 slices -> resize 224) ===",
          flush=True)
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:                                    # pragma: no cover
        print("  torch unavailable: %s" % e, flush=True)
        torch = None

    def item(i, res=224, k=12):
        v = vols[i]
        m = masks[i] if masks is not None else np.ones(D, np.uint8)
        valid = np.where(m > 0)[0]
        if len(valid) < 3:
            valid = np.arange(min(3, D))
        cs = np.linspace(valid.min() + 1, valid.max() - 1, k).round().astype(int)
        tiles = []
        for c in cs:
            c = max(1, min(int(c), D - 2))
            tri = np.stack([v[c - 1], v[c], v[c + 1]], 0).astype(np.float32) / 255.0
            if torch is not None:
                tt = torch.from_numpy(np.ascontiguousarray(tri))
                if tt.shape[-1] != res:
                    tt = F.interpolate(tt[None], size=(res, res), mode="bilinear",
                                       align_corners=False)[0]
                tiles.append(tt.numpy())
            else:
                tiles.append(tri[:, :res, :res])
        return np.stack(tiles)

    if torch is not None:
        for label, nthread in (("1 thread", 1), ("4 threads", 4)):
            def task():
                if nthread == 1:
                    return [item(i) for i in idx]
                with ThreadPoolExecutor(nthread) as ex:
                    return list(ex.map(item, idx))
            t, _ = timed(task, reps=2)
            print("  %-10s %.4f s/item  -> %.1f items/s  (%.0f s/epoch at 3479 studies)"
                  % (label, t / len(idx), len(idx) / t, t / len(idx) * 3479), flush=True)

    # ---------------- fold composition ----------------
    if paths["fold_file.json"] and masks is not None:
        fmap = json.load(open(paths["fold_file.json"]))["folds"]
        ids = np.load(paths["all_ids.npy"], allow_pickle=True) if paths["all_ids.npy"] else None
        print("\n=== fold composition (valid slices per study) ===", flush=True)
        if ids is not None:
            ids = np.array([str(x) for x in ids])
            nvalid = (np.asarray(masks) > 0).sum(1)
            print("  n_ids %d  n_masks %d" % (len(ids), len(nvalid)), flush=True)
            fold = np.array([fmap.get(u, -1) for u in ids], dtype=int)
            for name, sel in (("ALL", np.ones(len(ids), bool)),
                              ("fold0 train", fold >= 0),
                              ("gold (no fold)", fold < 0)):
                if sel.sum() == 0:
                    continue
                v = nvalid[sel]
                print("  %-14s n=%5d  valid-slices mean %.1f  p10 %d  median %d  p90 %d  max %d"
                      % (name, sel.sum(), v.mean(), np.percentile(v, 10), np.median(v),
                         np.percentile(v, 90), v.max()), flush=True)
            for f in sorted(set(fold.tolist())):
                if f < 0:
                    continue
                sel = fold == f
                v = nvalid[sel]
                print("  fold %-2d        n=%5d  mean %.1f  median %d  max %d"
                      % (f, sel.sum(), v.mean(), np.median(v), v.max()), flush=True)
        else:
            print("  all_ids.npy not found - cannot join folds", flush=True)

    print("\nDONE in %.1f s" % (time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
