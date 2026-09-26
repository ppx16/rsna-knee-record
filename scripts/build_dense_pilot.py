# -*- coding: utf-8 -*-
"""Build the dense-corpus (64-slot) pilot kernel: the trainer plus an N-part corpus loader.

WHY THIS EXISTS (HANDOFF 152.3).  The 64-slot corpus is 4407 x 64 x 336 x 336 = **31.9 GB**, which
is over Kaggle's 20 GB kernel-output limit.  So it can never exist as one file, and it can never be
downloaded-and-reuploaded either (the shard outputs come off www.kaggleusercontent.com at ~1.6 MB/s,
about 10x slower than dataset GCS -> ~5.5 h just to fetch).  The only practical route is
`kernel_sources`: Kaggle mounts each shard kernel's output read-only at /kaggle/input/<slug>/, so the
training kernel sees the whole corpus with zero transfer and zero disk.

The trainer is built for exactly TWO corpus parts (`_TwoPartVols`).  This builder therefore patches
in a general N-part memory-mapped view plus a shard-first `_open_corpus` / `_open_ids`.

DANGER THIS GUARDS AGAINST.  A memmap view is order-sensitive and fails SILENTLY: if the shards are
concatenated in the wrong order, or a shard is missing, then row k no longer corresponds to the
study that id k names -- and every label is attached to the wrong volumes.  Nothing raises; the
numbers just quietly become meaningless.  Two defences:

  1. shards are ordered by the INTEGER in `_s<N>.npy`, never lexicographically (`sorted()` would put
     s10 before s2 once there are >= 10 shards);
  2. the count invariant is asserted (sum of shard rows == len(ids) == expected total).

`--selftest` proves both on synthetic shards, each filled with a value that identifies its own row,
so a mis-ordering cannot pass.

    python scripts/build_dense_pilot.py --selftest
    python scripts/build_dense_pilot.py --shards 8 --total 4407
    python scripts/build_dense_pilot.py --shards 8 --total 4407 --push
"""
import argparse
import ast
import json
import pathlib
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parent.parent
from kpaths import kdir          # kernels/ layout, see kpaths.py
SRC = kdir("kernel-raptor-train") / "knee-mri-training-the-twelve-finding-model.py"

# ---------------------------------------------------------------------------------------------
# The three patch bodies.  They live here as literals so that --selftest can exec exactly the code
# that gets written into the kernel -- testing a copy would test nothing.
# ---------------------------------------------------------------------------------------------
NPART_SRC = '''
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
SHARD_REQUIRED = False


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
'''

OPEN_CORPUS_SRC = '''
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
'''

OPEN_IDS_SRC = '''
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
'''

ANCHOR_NPART = "def _open_corpus():"
ANCHOR_IDS = "def _open_ids():"


def patch(src, total=4407, require_shards=False):
    """Insert _NPartVols above _open_corpus, then replace both loader bodies."""
    if src.count(ANCHOR_NPART) != 1:
        raise SystemExit("anchor %r appears %d times" % (ANCHOR_NPART, src.count(ANCHOR_NPART)))
    if src.count(ANCHOR_IDS) != 1:
        raise SystemExit("anchor %r appears %d times" % (ANCHOR_IDS, src.count(ANCHOR_IDS)))

    # Replace _open_corpus: from its def up to the next top-level blank-line-separated def.
    def replace_def(text, anchor, new_src):
        start = text.index(anchor)
        rest = text[start:]
        # the function ends at the next line that starts at column 0 with 'def ' or a non-space
        lines = rest.split("\n")
        end = len(lines)
        for k in range(1, len(lines)):
            ln = lines[k]
            if ln and not ln[0].isspace() and not ln.startswith(")"):
                end = k
                break
        return text[:start] + new_src.lstrip("\n") + "\n" + "\n".join(lines[end:])

    out = src
    out = replace_def(out, ANCHOR_NPART, NPART_SRC + "\n" + OPEN_CORPUS_SRC)
    out = replace_def(out, ANCHOR_IDS, OPEN_IDS_SRC)
    out = out.replace("SHARD_EXPECTED_ROWS = 4407", "SHARD_EXPECTED_ROWS = %d" % total)
    if require_shards:
        out = out.replace("SHARD_REQUIRED = False", "SHARD_REQUIRED = True")
    for probe in ("SHARD_GLOB", "class _NPartVols", "parts = _shard_parts()",
                  "SHARD_EXPECTED_ROWS = %d" % total):
        if probe not in out:
            raise SystemExit("patch verification failed: %r missing" % probe)
    if "_TwoPartVols(np.load(av" not in out:
        raise SystemExit("the two-part fallback was lost -- the 44-slot arm would break")
    ast.parse(out)
    return out


# ---------------------------------------------------------------------------------------------
PROBE_MAIN = '''

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
        for d in sorted(glob.glob("/kaggle/input/*/vols_s0.npy")) + \
                 sorted(glob.glob("/kaggle/input/*/*/vols_s0.npy")) + \
                 sorted(glob.glob("/kaggle/input/*/*/*/vols_s0.npy")):
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
'''


def build_probe(out, shard_sources, total, require_shards=True):
    """A CPU-only kernel that mounts the same 8 shards and asserts they assemble correctly."""
    stub = '''
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
'''
    code = (NPART_SRC + "\n" + stub + "\n" + OPEN_CORPUS_SRC + "\n" + OPEN_IDS_SRC + PROBE_MAIN)
    code = code.replace("SHARD_EXPECTED_ROWS = 4407", "SHARD_EXPECTED_ROWS = %d" % total)
    if require_shards:
        code = code.replace("SHARD_REQUIRED = False", "SHARD_REQUIRED = True")
    code = "import os, glob, sys\nimport numpy as np\nHERE = '/kaggle/working'\n" + code
    ast.parse(code)
    out.mkdir(exist_ok=True)
    (out / "probe.py").write_text(code, encoding="utf-8")
    meta = {
        "id": "ppxl16/rsna-knee-d64probe",
        "title": "RSNA knee d64probe",
        "code_file": "probe.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,          # free: no GPU, no TPU, no quota -- only CPU session slots
        "enable_internet": False,
        "dataset_sources": [
            "dreaddevelopment/knee-raptor-corpus",
            "dreaddevelopment/knee-raptor-corpus-ext",
        ],
        "kernel_sources": shard_sources,
        "competition_sources": ["rsna-knee-abnormality-detection"],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("built %s (CPU only, %d shard sources)" % (out, len(shard_sources)))
    return out


def selftest():
    """Prove the loader on synthetic shards whose every row identifies itself.

    Each shard's volumes are filled with a constant derived from (shard, row), so a wrong
    concatenation order makes the read-back assertion fail instead of passing quietly.
    """
    import gc
    import tempfile
    import numpy as np

    counts = [3, 2, 4, 1, 2, 2, 3, 2]          # deliberately unequal, like the real s7 (550)
    total = sum(counts)

    def make_tree(present, root):
        for n in present:
            c = counts[n]
            d = root / ("rsna-knee-corpus64-s%d" % n)
            d.mkdir()
            V = np.zeros((c, 4, 2, 2), np.uint8)
            for r in range(c):
                V[r] = (n * 100 + r) % 251
            np.save(d / ("vols_s%d.npy" % n), V)
            np.save(d / ("masks_s%d.npy" % n), np.ones((c, 4), np.uint8))
            np.save(d / ("ids_s%d.npy" % n), np.array(["id_%d_%d" % (n, r) for r in range(c)],
                                                      dtype=object))

    def ns_for(root, expected, flat=False):
        code = NPART_SRC + "\n" + OPEN_CORPUS_SRC + "\n" + OPEN_IDS_SRC
        code = code.replace("SHARD_EXPECTED_ROWS = 4407", "SHARD_EXPECTED_ROWS = %d" % expected)
        # Override at the END: exec runs top to bottom and _shard_parts reads the global when it is
        # called, so a trailing assignment is what takes effect.
        if flat:
            # the published-dataset shape: no per-shard subdirectories, all files in one dir
            code += ("\nSHARD_GLOBS = ()\nSHARD_FLAT_GLOBS = (%r,)\nSHARD_GLOB = 'FLAT'\n"
                     % (str(root) + "/"))
        else:
            code += ("\nSHARD_GLOBS = (%r,)\nSHARD_GLOB = SHARD_GLOBS[0]\n"
                     % (str(root) + "/rsna-knee-corpus64-s*/"))
        ns = {"np": np, "os": __import__("os"), "glob": __import__("glob"), "HERE": str(root)}
        exec(compile(code, "<patch>", "exec"), ns)
        return ns

    problems = []
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="shardselftest_"))
    tmp_bad = pathlib.Path(tempfile.mkdtemp(prefix="shardselftest_bad_"))
    tmp_flat = pathlib.Path(tempfile.mkdtemp(prefix="shardselftest_flat_"))
    try:
        make_tree(range(8), tmp)
        ns = ns_for(tmp, total)
        vols, masks = ns["_open_corpus"]()
        ids = ns["_open_ids"]()
        if vols.shape != (total, 4, 2, 2):
            problems.append("vols shape %s != %s" % (vols.shape, (total, 4, 2, 2)))
        if masks.shape[0] != total or len(ids) != total:
            problems.append("masks %d / ids %d != %d" % (masks.shape[0], len(ids), total))
        if len(set(ids.tolist())) != total:
            problems.append("ids are not unique")
        k = 0
        for n in range(8):
            for r in range(counts[n]):
                want_v, want_id = (n * 100 + r) % 251, "id_%d_%d" % (n, r)
                if int(vols[k][0, 0, 0]) != want_v:
                    problems.append("row %d: vol %d != %d" % (k, int(vols[k][0, 0, 0]), want_v))
                if str(ids[k]) != want_id:
                    problems.append("row %d: id %r != %r" % (k, str(ids[k]), want_id))
                k += 1

        # A MISSING shard must be a loud failure, not a silently shifted corpus: keep the full
        # expected total while mounting only 7 of the 8 shards.
        make_tree([n for n in range(8) if n != 3], tmp_bad)
        ns_bad = ns_for(tmp_bad, total)
        try:
            ns_bad["_open_corpus"]()
            problems.append("a missing shard was NOT detected")
        except SystemExit as e:
            if "rows but" not in str(e):
                problems.append("missing shard raised the wrong error: %s" % e)

        del vols, masks, ids
        gc.collect()

        # --- FLAT layout, i.e. exactly how the published dataset will arrive ---------------------
        for n in range(8):
            c = counts[n]
            V = np.zeros((c, 4, 2, 2), np.uint8)
            for r in range(c):
                V[r] = (n * 100 + r) % 251
            np.save(tmp_flat / ("vols_s%d.npy" % n), V)
            np.save(tmp_flat / ("masks_s%d.npy" % n), np.ones((c, 4), np.uint8))
            np.save(tmp_flat / ("ids_s%d.npy" % n),
                    np.array(["id_%d_%d" % (n, r) for r in range(c)], dtype=object))
        ns_flat = ns_for(tmp_flat, total, flat=True)
        fv, fm = ns_flat["_open_corpus"]()
        fi = ns_flat["_open_ids"]()
        if fv.shape != (total, 4, 2, 2):
            problems.append("FLAT: vols shape %s != %s" % (fv.shape, (total, 4, 2, 2)))
        if len(fi) != total or fm.shape[0] != total:
            problems.append("FLAT: ids %d / masks %d != %d" % (len(fi), fm.shape[0], total))
        k = 0
        for n in range(8):
            for r in range(counts[n]):
                if int(fv[k][0, 0, 0]) != (n * 100 + r) % 251 or str(fi[k]) != "id_%d_%d" % (n, r):
                    problems.append("FLAT: row %d is the wrong study" % k)
                k += 1
        del fv, fm, fi
        gc.collect()
    finally:
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp_bad, ignore_errors=True)
        shutil.rmtree(tmp_flat, ignore_errors=True)

    if problems:
        print("SELFTEST FAILED:")
        for p in problems[:10]:
            print("   ", p)
        return 1
    print("SELFTEST OK -- %d rows over %d shards; order, uniqueness and the missing-shard guard "
          "all hold" % (total, len(counts)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--total", type=int, default=4407)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="d64cn0")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--arch", default="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="also build kernel-d64probe (CPU-only mount check, free)")
    ap.add_argument("--probe-push", action="store_true")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if a.probe:
        probe = build_probe(kdir("kernel-d64probe"),
                            ["ppxl16/rsna-knee-corpus64-s%d" % i for i in range(a.shards)], a.total)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_names.py"),
                            str(probe / "probe.py")], capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
        if r.returncode != 0:
            raise SystemExit("check_names failed on the probe")
        if a.probe_push:
            subprocess.run(["kaggle", "kernels", "push", "-p", str(probe)], check=True)
        if not a.push:
            return 0

    out = kdir("kernel-d64pilot")
    out.mkdir(exist_ok=True)
    code = patch(SRC.read_text(encoding="utf-8"), a.total, require_shards=True)
    (out / "knee-mri-training-the-twelve-finding-model.py").write_text(code, encoding="utf-8")

    shard_sources = ["ppxl16/rsna-knee-corpus64-s%d" % i for i in range(a.shards)]
    meta = {
        "id": "ppxl16/rsna-knee-d64pilot",
        "title": "RSNA knee d64pilot",
        "code_file": "knee-mri-training-the-twelve-finding-model.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "keywords": ["gpu"],
        # The 44-slot corpus is deliberately NOT mounted: the shard branch takes precedence anyway,
        # and mounting 22 GB of a corpus we are not using only costs input IO.
        "dataset_sources": [
            "flight0234/rsna-knee-hybrid-report-labels",
            "ppxl16/rsna-knee-cv-folds",
        ],
        "kernel_sources": shard_sources,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "machine_shape": "NvidiaTeslaT4",
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out / "RUN.txt").write_text(
        "# how to launch the pilot (script kernels take no argv, so these are baked in)\n"
        "python knee-mri-training-the-twelve-finding-model.py --labels $(ls /kaggle/input/"
        "rsna-knee-hybrid-report-labels/*.parquet | head -1) --folds 5 --fold %d "
        "--fold_file /kaggle/input/rsna-knee-cv-folds/fold_file.json --arch %s --res %d --bs %d "
        "--k %d --k_eval %d --epochs %d --grad_ckpt --tag %s --workers 2\n"
        % (a.fold, a.arch, a.res, a.bs, a.k, a.k * 2, a.epochs, a.tag), encoding="utf-8")

    print("built %s" % out)
    print("  shards   : %d (%s .. %s)" % (len(shard_sources), shard_sources[0], shard_sources[-1]))
    print("  declared : %d rows total" % a.total)
    print("  config   : fold %d, %s @%d, bs %d, k %d, %d epochs, tag %s"
          % (a.fold, a.arch, a.res, a.bs, a.k, a.epochs, a.tag))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_names.py"),
                        str(out / "knee-mri-training-the-twelve-finding-model.py")],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        raise SystemExit("check_names failed")

    if a.push:
        shards_ready = all(
            "COMPLETE" in subprocess.run(["kaggle", "kernels", "status", s],
                                         capture_output=True, text=True).stdout
            for s in shard_sources)
        if not shards_ready:
            raise SystemExit("refusing to push: not every shard kernel is COMPLETE yet")
        subprocess.run(["kaggle", "kernels", "push", "-p", str(out)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
