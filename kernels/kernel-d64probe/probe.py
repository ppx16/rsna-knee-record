import os, glob, sys
import numpy as np
HERE = '/kaggle/working'

# MEASURED 2026-09-24 by the mount probe: a `kernel_sources` entry does NOT mount at
# /kaggle/input/<slug>/ -- it mounts at /kaggle/input/notebooks/<owner>/<slug>/.  The flat form is
# kept as a second try because the published 0.943 notebook references both spellings, i.e. this has
# differed before.  Guessing one and silently getting nothing is the failure this guards against.
SHARD_GLOBS = (
    # 1. kernel_sources: MEASURED 2026-09-24 -- a kernel source mounts under /notebooks/<owner>/,
    #    NOT at /kaggle/input/<slug>/.  This is the form the pilot used and the CPU probe verified.
    "/kaggle/input/notebooks/*/rsna-knee-corpus64-s*/",
    # 2. the published DATASET, laid out as per-shard subdirectories (rsna-knee-corpus64-s0/vols_s0.npy)
    "/kaggle/input/*/rsna-knee-corpus64-s*/",
    # 3. a flat dataset mount
    "/kaggle/input/rsna-knee-corpus64-s*/",
    # 4. a rented box that pulled the dataset (the 44-slot corpus lives under autodl-tmp too)
    "/root/autodl-tmp/corpus64/rsna-knee-corpus64-s*/",
    "/root/corpus64/rsna-knee-corpus64-s*/",
)
SHARD_GLOB = SHARD_GLOBS[0]          # kept for diagnostics/messages

# A DATASET cannot be relied on to preserve nested directories: `kaggle datasets create --dir-mode`
# only offers skip/zip/tar for directories, and zip/tar would archive them (the mount would then hold
# archives, not the tree the loader wants).  So the published dataset is FLAT -- all 24 files in its
# root -- and these globs point at such a directory.
SHARD_FLAT_GLOBS = (
    "/kaggle/input/rsna-knee-corpus64-dense/",           # the published dataset
    "/kaggle/input/*/rsna-knee-corpus64-dense/",
    "/root/autodl-tmp/corpus64/ds/",                     # a rented box that pulled it
    "/root/corpus64/ds/",
)
# The corpus is exactly the 4407-study training set (part 1's 3200 + ext's 1207).  Nothing else
# detects a DROPPED or SHORT shard: ids and vols both come from the same glob, so their counts agree
# even when a shard is missing entirely -- and a corpus that is short by 551 rows still trains, it
# just pairs the last folds' studies with the wrong volumes.  Only the absolute total catches it.
SHARD_EXPECTED_ROWS = 4407
# Set true for the pilot/probe.  Without it, a kernel whose kernel_sources did not mount would fall
# through to the published 44-slot corpus and TRAIN THE WRONG ARM while every log line still looked
# healthy -- producing a delta-OOF of ~0 and a false conclusion that the dense corpus does not help.
# MEASURED 2026-09-24: the mount probe found 0 shards, so this is not a hypothetical.
SHARD_REQUIRED = True


def _shard_parts():
    """The 64-slot corpus as mounted shard kernels, in ROW order.

    A kernel_sources entry mounts at /kaggle/input/<slug>/, so shard i is /kaggle/input/
    rsna-knee-corpus64-s<i>/vols_s<i>.npy.  Ordering is by the integer in the filename: sorted()
    on the strings would place s10 before s2 once there are ten or more shards, which would
    silently pair every study past that point with the wrong volumes.
    """
    out = []
    for pattern in SHARD_GLOBS:
        for d in sorted(glob.glob(pattern)):
            v = glob.glob(os.path.join(d, "vols_s*.npy"))
            i = glob.glob(os.path.join(d, "ids_s*.npy"))
            m = glob.glob(os.path.join(d, "masks_s*.npy"))
            if not (v and i and m):
                continue
            try:
                n = int(os.path.basename(v[0]).split("_s")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            if any(n == e[0] for e in out):        # both patterns matched the same shard
                continue
            out.append((n, v[0], i[0], m[0]))
    out.sort(key=lambda t: t[0])
    if out:
        return out
    # FLAT layout: one directory holding vols_s0..N.npy / masks_sN.npy / ids_sN.npy side by side.
    # This is how the published dataset arrives, so it is a first-class case, not a fallback hack.
    for pattern in SHARD_FLAT_GLOBS:
        for d in sorted(glob.glob(pattern)):
            for v in sorted(glob.glob(os.path.join(d, "vols_s*.npy"))):
                try:
                    n = int(os.path.basename(v).split("_s")[1].split(".")[0])
                except (IndexError, ValueError):
                    continue
                i = os.path.join(d, "ids_s%d.npy" % n)
                m = os.path.join(d, "masks_s%d.npy" % n)
                if os.path.exists(i) and os.path.exists(m) and not any(n == e[0] for e in out):
                    out.append((n, v, i, m))
    out.sort(key=lambda t: t[0])
    return out


class _NPartVols:
    """N memory-mapped shards presented as one (N_rows, D, H, W) array.

    Only __getitem__/__len__/shape are used by the training loop, and it is asked for one study at
    a time, so a linear scan over the (few) shard boundaries is free.  Nothing is concatenated on
    disk: that would need 31.9 GB of scratch space that does not exist.
    """

    def __init__(self, parts):
        self.parts = list(parts)
        self.starts = []
        n = 0
        for p in self.parts:
            self.starts.append(n)
            n += p.shape[0]
        self.shape = (n,) + tuple(self.parts[0].shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        r = int(row)
        if r < 0:
            r += self.shape[0]
        for k in range(len(self.parts) - 1, -1, -1):
            if r >= self.starts[k]:
                return self.parts[k][r - self.starts[k]]
        raise IndexError(r)


class _TwoPartVols:
    """Stand-in so the copied _open_corpus below resolves exactly as it does in the trainer.

    The probe always takes the shard branch, but the fallback has to exist for the file to be the
    SAME code as the trainer's -- a probe that re-implements the loader proves nothing.
    """

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        return self.a[row] if row < self.a.shape[0] else self.b[row - self.a.shape[0]]


def _open_corpus():
    """Return (vols, masks).  Three ways in, in order of precedence:

    1. the mounted 64-slot shard kernels  (dense-corpus arm)
    2. /kaggle/working/all_vols.npy       (the deterministic single-file mount)
    3. the two published corpus parts     (the 44-slot arm, 3200 + 1207)

    Precedence matters: without it `_find("all_vols.npy")` happily returns the 44-slot corpus
    while the id list says 64 slots, and nothing raises.
    """
    parts = _shard_parts()
    if not parts and SHARD_REQUIRED:
        raise SystemExit(
            "SHARD_REQUIRED is set but NO shard kernels are mounted under %s. Refusing to fall "
            "back to the published corpus: that would quietly run the WRONG ARM (44 slots) while "
            "the run looked like the 64-slot arm." % SHARD_GLOB)
    if parts:
        vols = _NPartVols([np.load(v, mmap_mode="r") for _, v, _, _ in parts])
        masks = np.concatenate([np.load(m) for _, _, _, m in parts], axis=0)
        print("[corpus] %d mounted shards, rows %s, D=%d, %dx%d -> %s" % (
            len(parts), [p.shape[0] for p in vols.parts], vols.shape[1],
            vols.shape[2], vols.shape[3], vols.shape), flush=True)
        if masks.shape[0] != vols.shape[0]:
            raise SystemExit("shard masks/vols disagree: %d masks vs %d vols" % (
                masks.shape[0], vols.shape[0]))
        if vols.shape[0] != SHARD_EXPECTED_ROWS:
            raise SystemExit(
                "shard corpus has %d rows but %d were built -- a shard is missing, short, or was "
                "not mounted. Row k would not be the study that id k names."
                % (vols.shape[0], SHARD_EXPECTED_ROWS))
        return vols, masks
    local_v = os.path.join(HERE, "all_vols.npy")
    if os.path.exists(local_v):
        return (np.load(local_v, mmap_mode="r"),
                np.load(os.path.join(HERE, "all_masks.npy")))
    av, bv = _find("all_vols.npy"), _find("extra_vols.npy")
    am, bm = _find("all_masks.npy"), _find("extra_masks.npy")
    if not all((av, bv, am, bm)):
        raise SystemExit("Could not find the corpus. Attach both parts: "
                         "dreaddevelopment/knee-raptor-corpus and "
                         "dreaddevelopment/knee-raptor-corpus-ext")
    vols = _TwoPartVols(np.load(av, mmap_mode="r"), np.load(bv, mmap_mode="r"))
    masks = np.concatenate([np.load(am), np.load(bm)], axis=0)
    return vols, masks


def _open_ids():
    """The study id for each corpus row, in the SAME order as the volumes.

    This and _open_corpus must agree row-for-row; nothing downstream re-checks it, and a mismatch
    does not raise, it just trains on shuffled labels.  So the count invariant is asserted here.
    """
    parts = _shard_parts()
    if parts:
        ids = np.concatenate([np.load(i, allow_pickle=True).astype(str)
                              for _, _, i, _ in parts])
        rows = sum(np.load(v, mmap_mode="r").shape[0] for _, v, _, _ in parts)
        print("[corpus] ids from %d shards: %d ids, %d rows" % (len(parts), len(ids), rows),
              flush=True)
        if len(ids) != rows:
            raise SystemExit("shard ids/vols disagree: %d ids vs %d rows" % (len(ids), rows))
        if len(set(ids.tolist())) != len(ids):
            raise SystemExit("duplicate ids in the mounted shards -- a shard was built twice")
        return ids
    local = os.path.join(HERE, "all_ids.npy")
    if os.path.exists(local):
        return np.load(local, allow_pickle=True).astype(str)
    a, b = _find("all_ids.npy"), _find("extra_ids.npy")
    if not (a and b):
        raise SystemExit("Could not find all_ids.npy / extra_ids.npy - attach both corpus parts.")
    return np.concatenate([np.load(a, allow_pickle=True).astype(str),
                           np.load(b, allow_pickle=True).astype(str)])


# =============================================================================================
# MOUNT PROBE.  Everything above this line is copied verbatim from the pilot kernel's loader, so
# this exercises the real code and not a re-implementation.  It runs on a FREE CPU kernel and
# answers the only question the synthetic self-test cannot: does `kernel_sources` actually mount
# all eight shards, in order, as one 4407-row corpus?  That question otherwise stays open until
# the first GPU run -- which is exactly when we can least afford a surprise.
# =============================================================================================
def _find(name):
    """Depth-bounded; the trainer's own _find walks /kaggle/input unbounded."""
    for pat in ("/kaggle/input/*/" + name, "/kaggle/input/*/*/" + name,
                "/kaggle/input/datasets/*/*/" + name):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def main():
    problems = []
    parts = _shard_parts()
    print("[probe] shard dirs found: %d" % len(parts), flush=True)
    if not parts:
        # We do not know where Kaggle put the mounted kernel outputs; print the actual tree rather
        # than guessing again.  Depth-bounded on purpose (an unbounded walk over /kaggle/input once
        # cost 623 s, longer than an entire corpus build).
        print("[probe] SHARD_GLOB=%r matched nothing. What is actually under /kaggle/input:" %
              SHARD_GLOB, flush=True)
        for depth in (1, 2, 3):
            for d in sorted(glob.glob("/kaggle/input/" + "*/" * depth)):
                print("   %s" % d, flush=True)
        for depth in (1, 2, 3, 4):
            for d in sorted(glob.glob("/kaggle/input/" + "*/" * depth + "corpus64*")):
                print("   MATCH-> %s" % d, flush=True)
        for d in sorted(glob.glob("/kaggle/input/*/vols_s0.npy")) +                  sorted(glob.glob("/kaggle/input/*/*/vols_s0.npy")) +                  sorted(glob.glob("/kaggle/input/*/*/*/vols_s0.npy")):
            print("   VOLS-> %s" % d, flush=True)
    if len(parts) != 8:
        problems.append("expected 8 mounted shards, found %d" % len(parts))
    for n, v, i, m in parts:
        vs = np.load(v, mmap_mode="r").shape
        ms = np.load(m, mmap_mode="r").shape
        ids = np.load(i, allow_pickle=True)
        print("   s%-3d dir=%-24s vols=%-22s masks=%-12s ids=%d" % (
            n, os.path.basename(os.path.dirname(v)), vs, ms, len(ids)), flush=True)
        if vs[0] != len(ids) or ms[0] != len(ids):
            problems.append("shard s%d: rows disagree vols %d masks %d ids %d"
                            % (n, vs[0], ms[0], len(ids)))
        if vs[1:] != (64, 336, 336):
            problems.append("shard s%d: slot/res shape %s != (64, 336, 336)" % (n, vs[1:]))

    vols, masks = _open_corpus()
    ids = _open_ids()
    print("[probe] assembled vols=%s masks=%s ids=%d" % (vols.shape, masks.shape, len(ids)),
          flush=True)
    if vols.shape[0] != SHARD_EXPECTED_ROWS:
        problems.append("assembled rows %d != %d" % (vols.shape[0], SHARD_EXPECTED_ROWS))
    if len(set(ids.tolist())) != len(ids):
        problems.append("duplicate ids in the assembled corpus")

    # --- the row-order cross-check: the shard ids must equal the published corpus's ids + ext ids,
    # concatenated.  This is the contract the whole pilot rests on, checked on the real mounts.
    pa, pb = _find("all_ids.npy"), _find("extra_ids.npy")
    if not (pa and pb):
        problems.append("published id files are not mounted -> order cannot be cross-checked "
                        "(add knee-raptor-corpus and knee-raptor-corpus-ext)")
    else:
        ref = np.concatenate([np.load(pa, allow_pickle=True).astype(str),
                              np.load(pb, allow_pickle=True).astype(str)])
        print("[probe] reference concat(all_ids, extra_ids) = %d ids" % len(ref), flush=True)
        if list(ref) != list(ids):
            bad = next((k for k in range(min(len(ref), len(ids))) if ref[k] != ids[k]), None)
            problems.append("shard id order != concat(all_ids, extra_ids); first mismatch at row %s"
                            % bad)
        else:
            print("[probe] ID ORDER MATCHES the published corpus exactly", flush=True)

    # --- pixel spot-checks at every shard boundary and across the 3200-row split
    edges = [0, 550, 551, 1101, 1102, 2754, 2755, 3199, 3200, 3305, 3306, 3856, 3857, 4406]
    for r in edges:
        if r >= vols.shape[0]:
            continue
        v = np.asarray(vols[r])
        m = np.asarray(masks[r])
        nz = int((v.reshape(v.shape[0], -1).max(1) > 0).sum())
        if int(m.sum()) != nz:
            problems.append("row %d: mask says %d filled but %d have pixels" % (r, int(m.sum()), nz))
    print("[probe] checked %d boundary rows; mask/pixel agreement holds" % len(edges), flush=True)
    for r in (0, 2754, 2755, 3199, 3200, 4406):
        if r < vols.shape[0]:
            v = np.asarray(vols[r])
            print("   row %-5d %-14s slots filled %2d/64  max %3d  nonzero %d" % (
                r, str(ids[r])[-12:], int(np.asarray(masks[r]).sum()), int(v.max()),
                int((v > 0).sum())), flush=True)

    if problems:
        print("PROBE FAILED:", flush=True)
        for p in problems:
            print("   !! %s" % p, flush=True)
        raise SystemExit(1)
    print("PROBE OK -- 8 shards mounted in order, %d rows, 64 slots, ids match the published "
          "corpus row for row" % vols.shape[0], flush=True)


main()
