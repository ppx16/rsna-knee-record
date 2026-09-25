# -*- coding: utf-8 -*-
"""Audit the 64-slot dense corpus against the published 44-slot one, slice by slice.

WHY.  C85 concluded "the dense corpus does not help CoAtNet".  That conclusion is only valid if the
dense corpus is actually a *good* denser version of the same thing.  If it is subtly defective -- more
duplicated slices, lower occupancy, different intensity -- then the null result is an artifact of the
corpus, not a fact about density.  So audit the artifact before believing the negative.

Three concrete defect hypotheses, all checkable from the files on disk:

  H1 DUPLICATES.  pick_indices() is `rint(linspace(floor(lo*(n-1)), floor(hi*(n-1)), k))`, so a group
     can yield at most `hi-lo+1` DISTINCT slice indices.  64 slots need ~0.96*(n-1)+1 >= 64, i.e.
     n >= 67 slices; 44 slots need n >= 63.  For studies with 63..66 slices the 44-slot corpus is fully
     distinct while the 64-slot one must repeat slices -- i.e. the "denser" corpus would carry fewer
     real slices than it appears to.
  H2 OCCUPANCY.  If the denser sampling reaches into the ends of the stack where a group is empty, the
     64-slot corpus could fill fewer slots (measured mask density 0.933-0.949 -- but the 44-slot value
     was never recorded, so there is nothing to compare it against).
  H3 PREPROCESSING.  Different intensity distribution would mean the crop/resize path differs.

Both corpora are on this machine in the same row order, so every check is a direct per-slice compare.

    python scripts/audit_corpus64.py --rows 40
"""
import argparse
import hashlib
import pathlib
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R44 = pathlib.Path(r"D:\kaggle\corpus\data\knee-raptor-corpus")
R44E = pathlib.Path(r"D:\kaggle\corpus\data\knee-raptor-corpus-ext")
R64 = pathlib.Path(r"D:\kaggle\rsna-knee\corpus64\shards")

# group layout of each corpus, so a "group" can be compared with its counterpart
G44 = [("Sag", 1, 12), ("Sag", 0, 10), ("Cor", 1, 8), ("Cor", 0, 6), ("Ax", -1, 8)]
G64 = [("Sag", 1, 18), ("Sag", 0, 14), ("Cor", 1, 12), ("Cor", 0, 8), ("Ax", -1, 12)]


def img_hash(a):
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=8).digest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=40)
    a = ap.parse_args()

    m44 = np.load(R44 / "all_masks.npy")
    m44e = np.load(R44E / "extra_masks.npy")
    M44 = np.concatenate([m44, m44e], axis=0)
    v44 = np.load(R44 / "all_vols.npy", mmap_mode="r")
    v44e = np.load(R44E / "extra_vols.npy", mmap_mode="r")
    n44 = v44.shape[0]

    ids64 = np.concatenate([np.load(R64 / ("ids_s%d.npy" % i), allow_pickle=True).astype(str)
                            for i in range(8)])
    V64 = np.concatenate([np.load(R64 / ("vols_s%d.npy" % i), mmap_mode="r") for i in range(8)],
                         axis=0)
    M64 = np.concatenate([np.load(R64 / ("masks_s%d.npy" % i)) for i in range(8)], axis=0)

    ids44 = np.concatenate([np.load(R44 / "all_ids.npy", allow_pickle=True).astype(str),
                            np.load(R44E / "extra_ids.npy", allow_pickle=True).astype(str)])
    print("shapes: 44-slot %s %s | 64-slot %s %s" % (v44.shape, v44e.shape, V64.shape, M64.shape))
    print("ids identical row-for-row: %s" % (list(ids44) == list(ids64)))

    rows = np.linspace(0, len(ids64) - 1, a.rows).astype(int)
    agg = {44: {"slots": 0, "filled": 0, "distinct": 0, "dups": 0, "sum": 0.0, "nz": 0},
           64: {"slots": 0, "filled": 0, "distinct": 0, "dups": 0, "sum": 0.0, "nz": 0}}
    per_group = {44: {}, 64: {}}
    worst = []

    for r in rows:
        row44 = np.asarray(v44[r] if r < n44 else v44e[r - n44])
        row64 = np.asarray(V64[r])
        mk44, mk64 = M44[r], M64[r]
        off44 = off64 = 0
        dup_pairs = 0
        for (g, (_, _, k44), (_, _, k64)) in zip(range(5), G44, G64):
            for tag, row, mk, off, k in ((44, row44, mk44, off44, k44), (64, row64, mk64, off64, k64)):
                blk = row[off:off + k]
                hashes = [img_hash(blk[j]) for j in range(k)]
                nd = len(set(hashes))
                filled = int((mk[off:off + k] > 0).sum())
                s = agg[tag]
                s["slots"] += k
                s["filled"] += filled
                s["distinct"] += nd
                s["dups"] += k - nd
                nz = blk[blk > 0]
                s["sum"] += float(nz.sum()) if nz.size else 0.0
                s["nz"] += int(nz.size)
                gk = per_group[tag].setdefault(g, {"distinct": 0, "slots": 0, "filled": 0})
                gk["distinct"] += nd
                gk["slots"] += k
                gk["filled"] += filled
            off44 += k44
            off64 += k64
        if agg[64]["dups"] - agg[44]["dups"] > 0:
            pass

    print()
    print("%-8s %8s %8s %10s %8s %10s" % ("corpus", "slots", "filled", "distinct", "dups", "mean(nz)"))
    for tag in (44, 64):
        s = agg[tag]
        print("%-8s %8d %8d %10d %8d %10.2f" % (
            "%d-slot" % tag, s["slots"], s["filled"], s["distinct"], s["dups"],
            (s["sum"] / s["nz"]) if s["nz"] else 0.0))
    print()
    print("%-6s %-28s %-28s" % ("group", "44-slot distinct/filled", "64-slot distinct/filled"))
    for g in range(5):
        p44, p64 = per_group[44][g], per_group[64][g]
        print("%-6d %-28s %-28s" % (
            g, "%d/%d" % (p44["distinct"], p44["filled"]), "%d/%d" % (p64["distinct"], p64["filled"])))

    print()
    d64 = agg[64]["dups"] / max(agg[64]["slots"], 1)
    d44 = agg[44]["dups"] / max(agg[44]["slots"], 1)
    print("duplicate-slot RATE: 44-slot %.1f%%  64-slot %.1f%%" % (100 * d44, 100 * d64))
    o44 = agg[44]["filled"] / max(agg[44]["slots"], 1)
    o64 = agg[64]["filled"] / max(agg[64]["slots"], 1)
    print("occupancy rate     : 44-slot %.1f%%  64-slot %.1f%%" % (100 * o44, 100 * o64))
    verdict = []
    if d64 > d44 + 0.02:
        verdict.append("!! 64 槽重复率明显更高 -- 密语料的'更密'可能是假的")
    if o64 < o44 - 0.02:
        verdict.append("!! 64 槽占用率明显更低")
    print("AUDIT:", " / ".join(verdict) if verdict else "no duplicate/occupancy defect detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
