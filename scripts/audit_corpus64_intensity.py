# -*- coding: utf-8 -*-
"""Is the 64-slot corpus's lower intensity a legitimate span effect, or a preprocessing mismatch?

The audit found mean non-zero intensity 102.2 (published 44-slot) versus 58.1 (our 64-slot) -- a 43%
gap.  Two explanations, and they have opposite consequences:

  (a) SPAN EFFECT -- span 0.02-0.98 reaches the ends of the stack, and MRI stacks are low-signal at
      their ends, so the mean legitimately drops.  Then the corpus is fine.
  (b) PIPELINE MISMATCH -- our prep() scales intensities differently from the published builder.  Then
      the C85 and 154 comparisons are confounded: they compare "our corpus" against "their corpus",
      not "64 slots" against "44 slots".

How to tell them apart: compare intensity at the SAME RELATIVE POSITION within each group.  A span
effect is a gradient -- the middle should match and only the extremes should fall off.  A pipeline
mismatch shifts the whole distribution, middle included.

⚠️ Why the original verify44 could not have caught (b): scripts/.../build_corpus.py compares rows with
corr_rows(), which mean-centres AND normalises each row first, so it is INVARIANT to any global scale
factor.  A verify built on correlation alone can never detect a brightness/scale difference.

    python scripts/audit_corpus64_intensity.py --rows 60
"""
import argparse
import pathlib
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R44 = pathlib.Path(r"D:\kaggle\corpus\data\knee-raptor-corpus")
R44E = pathlib.Path(r"D:\kaggle\corpus\data\knee-raptor-corpus-ext")
R64 = pathlib.Path(r"D:\kaggle\rsna-knee\corpus64\shards")
G44 = [("Sag", 1, 12), ("Sag", 0, 10), ("Cor", 1, 8), ("Cor", 0, 6), ("Ax", -1, 8)]
G64 = [("Sag", 1, 18), ("Sag", 0, 14), ("Cor", 1, 12), ("Cor", 0, 8), ("Ax", -1, 12)]
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]


def stats(block_slots):
    """mean of non-zero pixels, and fraction of pixels that are non-zero"""
    vals, nz = [], []
    for s in block_slots:
        n = s[s > 0]
        vals.append(float(n.mean()) if n.size else 0.0)
        nz.append(float((s > 0).mean()))
    return np.array(vals), np.array(nz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60)
    a = ap.parse_args()

    v44 = np.load(R44 / "all_vols.npy", mmap_mode="r")
    v44e = np.load(R44E / "extra_vols.npy", mmap_mode="r")
    n44 = v44.shape[0]
    V64 = np.concatenate([np.load(R64 / ("vols_s%d.npy" % i), mmap_mode="r") for i in range(8)],
                         axis=0)
    rows = np.linspace(0, V64.shape[0] - 1, a.rows).astype(int)

    # acc[tag][group][frac] = list of per-slot mean intensities
    acc = {44: {}, 64: {}}
    for r in rows:
        row44 = np.asarray(v44[r] if r < n44 else v44e[r - n44])
        row64 = np.asarray(V64[r])
        o44 = o64 = 0
        for g in range(5):
            k44, k64 = G44[g][2], G64[g][2]
            for tag, row, off, k in ((44, row44, o44, k44), (64, row64, o64, k64)):
                blk = row[off:off + k]
                for f in FRACS:
                    j = int(round(f * (k - 1)))
                    n = blk[j][blk[j] > 0]
                    acc[tag].setdefault(g, {}).setdefault(f, []).append(
                        float(n.mean()) if n.size else 0.0)
            o44 += k44
            o64 += k64

    print("mean non-zero intensity at the SAME relative position in each group")
    print("(span effect => middle matches, extremes fall off; pipeline mismatch => everything shifts)")
    print()
    print("%-18s %10s %10s %10s %10s %10s" % ("group/pos", *["%.2f" % f for f in FRACS]))
    for g in range(5):
        nm = "%s f=%d" % (G44[g][0], G44[g][1])
        for tag in (44, 64):
            v = [np.mean(acc[tag][g][f]) for f in FRACS]
            print("%-18s %10.1f %10.1f %10.1f %10.1f %10.1f   %s" % (
                nm if tag == 44 else "", *v, "%d-slot" % tag))
        m44 = np.mean(acc[44][g][0.5]); m64 = np.mean(acc[64][g][0.5])
        print("%-18s    mid ratio 64/44 = %.3f" % ("", (m64 / m44) if m44 else float("nan")))
    print()
    mids44 = [np.mean(acc[44][g][0.5]) for g in range(5)]
    mids64 = [np.mean(acc[64][g][0.5]) for g in range(5)]
    ratio = np.mean([b / a2 for a2, b in zip(mids44, mids64) if a2])
    ends44 = [np.mean(acc[44][g][f]) for g in range(5) for f in (0.0, 1.0)]
    ends64 = [np.mean(acc[64][g][f]) for g in range(5) for f in (0.0, 1.0)]
    ratio_end = np.mean([b / a2 for a2, b in zip(ends44, ends64) if a2])
    print("MIDDLE slots  64/44 intensity ratio: %.3f" % ratio)
    print("EXTREME slots 64/44 intensity ratio: %.3f" % ratio_end)
    print()
    if ratio < 0.85:
        print("=> a PIPELINE MISMATCH: even the middle of each group is much darker, so this is not a")
        print("   span effect.  The C85/154 comparisons are confounded by preprocessing, and the")
        print("   original verify44 could not have caught it (correlation is scale-invariant).")
    else:
        print("=> consistent with a SPAN EFFECT: the middle matches; the drop is concentrated at the")
        print("   stack ends, which is where MRI signal is weakest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
