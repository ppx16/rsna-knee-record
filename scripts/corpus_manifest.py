# -*- coding: utf-8 -*-
"""Checksum ledger for the TRAINING corpus, so it can be moved to rented compute safely.

Why this exists
---------------
The 2026-09 GPU-quota policy is: GPU only for submissions, experiments on TPU, and if the
TPU side is still too slow the owner buys GPU time elsewhere and MIGRATES THE DATASET.  The
training instrument (`kernel-raptor-train`, the TPU ports) never touches the 570 GB
competition DICOM: it reads an already-decoded corpus

    dreaddevelopment/knee-raptor-corpus      all_vols.npy   (3200,44,336,336) uint8  15.9 GB
    dreaddevelopment/knee-raptor-corpus-ext  extra_vols.npy (1207,44,336,336) uint8   6.0 GB

plus two tiny label/fold datasets.  That is ~22 GB, licence CC0, so it moves freely --
unlike the competition data, which may not be redistributed at all.  What was missing was
a way to prove, on the far side, that the copy that arrived is the copy we trained on.

    python scripts/corpus_manifest.py build          # hash everything, write the ledger
    python scripts/corpus_manifest.py build --root /data/corpus
    python scripts/corpus_manifest.py verify         # re-hash, exit 1 on any drift
    python scripts/corpus_manifest.py summary

Ledger lands in `state/CORPUS_MANIFEST.json`.  `verify` is the gate a rented box must pass
before any fold is launched: same sha256, or the run is not comparable to our CV.
"""
import argparse
import ast
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT_DEFAULT = pathlib.Path(r"D:\kaggle\corpus")
LEDGER = pathlib.Path(__file__).resolve().parent.parent / "state" / "CORPUS_MANIFEST.json"

# What the training path actually mounts.  Order matters only for display.
DATASETS = [
    ("dreaddevelopment/knee-raptor-corpus",
     "Knee MRI preprocessed slice stacks, part 1 of 2 (all_vols.npy = 3200 studies)"),
    ("dreaddevelopment/knee-raptor-corpus-ext",
     "part 2 of 2 (extra_vols.npy = the remaining 1207 studies)"),
    ("flight0234/rsna-knee-hybrid-report-labels",
     "the training labels the raptor fine-tune reads"),
    ("ppxl16/rsna-knee-cv-folds",
     "OUR fold assignment -- without it a rerun is not fold-comparable to our OOF"),
]

# Datasets that ship a single loose file rather than a bundle: corpus_fetch stages these into
# data/<slug>/ as well, so the Kaggle layout mirror stays uniform.
SINGLE_FILE = {
    "flight0234/rsna-knee-hybrid-report-labels": "report_labels_v4hybrid.csv",
    "ppxl16/rsna-knee-cv-folds": "fold_file.json",
}

# Shapes we expect to be true on the far side.  Anything else means a different corpus.
EXPECT = {
    "all_vols.npy": (3200, 44, 336, 336),
    "extra_vols.npy": (1207, 44, 336, 336),
}

CHUNK = 8 << 20

# Bookkeeping, not corpus: completion sidecars written by kaggle_resume.py and the CLI's resume
# markers.  Hashing them would make `verify` fail on a machine that finished its downloads at a
# different minute -- i.e. a false alarm on the one check that must be trustworthy.
SKIP_SUFFIX = (".ok", ".kaggle-partial")
# `packs/` holds the code zip built by make_pack.py FROM this repo.  It changes every time the
# code changes, so hashing it would make `verify` fail for a reason that has nothing to do with
# the corpus -- and the pack is what carries this ledger, so including it is also circular.
# `layout/` is the Kaggle-shaped mirror of `data/` (hardlinks to the same bytes).  Where it lives
# is machine-specific -- on the rented box it is /kaggle, i.e. OUTSIDE --root -- so a ledger that
# listed it would report MISSING on exactly the machine the ledger exists for.
SKIP_DIRS = {"packs", "layout"}


def _skipped(p: pathlib.Path) -> bool:
    return p.name.endswith(SKIP_SUFFIX) or bool(SKIP_DIRS & set(p.parts))


def _hdr(path: pathlib.Path) -> dict:
    """Read a .npy header without touching the payload (a 16 GB array has a ~100 B header)."""
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            return {"error": "not a .npy (%r)" % magic}
        major, _minor = f.read(2)
        n = int.from_bytes(f.read(2 if major == 1 else 4), "little")
        meta = ast.literal_eval(f.read(n).decode("latin1"))
    return {"shape": list(meta["shape"]), "dtype": str(meta["descr"]), "fortran": bool(meta["fortran_order"])}


def sha256(path: pathlib.Path, label: str = "") -> str:
    h = hashlib.sha256()
    done = 0
    total = path.stat().st_size
    t0 = time.time()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
            done += len(b)
            if total > (1 << 31):  # only narrate multi-GB files
                el = max(time.time() - t0, 1e-6)
                pct = 100.0 * done / total
                sys.stderr.write("\r  %-22s %5.1f%%  %6.1f MB/s   " % (label or path.name, pct, done / el / 1e6))
                sys.stderr.flush()
    if total > (1 << 31):
        sys.stderr.write("\r" + " " * 60 + "\r")
    return h.hexdigest()


def _kaggle_files(ref: str) -> list:
    """Server-side listing = an integrity anchor that does not depend on our own hashing."""
    try:
        out = subprocess.run(["kaggle", "datasets", "files", ref], capture_output=True, text=True, timeout=120)
    except Exception as e:  # kaggle CLI missing / offline: not fatal, just unanchored
        return [{"error": str(e)}]
    if out.returncode != 0:
        return [{"error": (out.stderr or out.stdout).strip()[:200]}]
    rows = []
    for line in out.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3 and parts[1].isdigit():
            rows.append({"name": parts[0], "bytes": int(parts[1]), "created": " ".join(parts[2:])})
    return rows


def build(root: pathlib.Path, with_kaggle: bool, out: pathlib.Path) -> int:
    if not root.exists():
        print("!! %s does not exist -- nothing to hash" % root)
        return 2
    files = sorted(p for p in root.rglob("*") if p.is_file() and not _skipped(p))
    if not files:
        print("!! %s is empty" % root)
        return 2
    recs = []
    for p in files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        size = p.stat().st_size
        print("[%2d/%2d] %-40s %8.3f GB" % (len(recs) + 1, len(files), rel, size / 1e9))
        rec = {"path": rel, "bytes": size, "sha256": sha256(p, rel)}
        if p.suffix == ".npy":
            rec["npy"] = _hdr(p)
        recs.append(rec)
    anchors = {ref: _kaggle_files(ref) for ref, _ in DATASETS} if with_kaggle else {}
    doc = {
        "root": str(root),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_bytes": sum(r["bytes"] for r in recs),
        "n_files": len(recs),
        "datasets": [{"ref": r, "why": w} for r, w in DATASETS],
        "expected_shapes": {k: list(v) for k, v in EXPECT.items()},
        "kaggle_server_files": anchors,
        "files": recs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print("\n== wrote %s  (%d files, %.2f GB)" % (out, len(recs), doc["total_bytes"] / 1e9))
    return check(doc, quiet=True)


def check(doc: dict, quiet: bool = False) -> int:
    """Shape sanity on top of the hashes: a corpus with the wrong shape is a different corpus."""
    bad = 0
    for r in doc["files"]:
        want = EXPECT.get(pathlib.PurePosixPath(r["path"]).name)
        got = r.get("npy", {}).get("shape")
        if want and got and list(want) != list(got):
            print("!! SHAPE %s = %s, expected %s" % (r["path"], got, list(want)))
            bad += 1
    ids = {pathlib.PurePosixPath(r["path"]).name for r in doc["files"] if r["path"].endswith("_ids.npy")}
    if not quiet:
        print("   id files present: %s" % (sorted(ids) or "NONE"))
    return 1 if bad else 0


def verify(root: pathlib.Path, out: pathlib.Path) -> int:
    if not out.exists():
        print("!! no ledger at %s -- run `build` first" % out)
        return 2
    doc = json.loads(out.read_text(encoding="utf-8"))
    print("ledger %s  built %s  %d files  %.2f GB" % (out.name, doc["generated_utc"], doc["n_files"], doc["total_bytes"] / 1e9))
    have = {str(p.relative_to(root)).replace("\\", "/"): p for p in root.rglob("*")
            if p.is_file() and not _skipped(p)} if root.exists() else {}
    bad = 0
    for r in doc["files"]:
        p = have.pop(r["path"], None)
        if p is None:
            print("!! MISSING %s" % r["path"])
            bad += 1
            continue
        if p.stat().st_size != r["bytes"]:
            print("!! SIZE    %s  %d != %d" % (r["path"], p.stat().st_size, r["bytes"]))
            bad += 1
            continue
        got = sha256(p, r["path"])
        if got != r["sha256"]:
            print("!! SHA256  %s\n     got %s\n     want %s" % (r["path"], got, r["sha256"]))
            bad += 1
        else:
            print("ok         %s" % r["path"])
    for extra in have:
        print("?? EXTRA   %s (not in ledger)" % extra)
    bad += check(doc)
    print("\n== %s" % ("VERIFY FAILED: %d problem(s)" % bad if bad else "VERIFY OK -- this copy is the copy we trained on"))
    return 1 if bad else 0


def summary(out: pathlib.Path) -> int:
    if not out.exists():
        print("!! no ledger at %s" % out)
        return 2
    doc = json.loads(out.read_text(encoding="utf-8"))
    print("corpus root : %s" % doc["root"])
    print("generated   : %s" % doc["generated_utc"])
    print("total       : %d files, %.2f GB" % (doc["n_files"], doc["total_bytes"] / 1e9))
    print("\n%-46s %12s  %s" % ("file", "GB", "shape"))
    for r in sorted(doc["files"], key=lambda r: -r["bytes"]):
        npy = r.get("npy", {})
        shape = "x".join(str(s) for s in npy.get("shape", [])) + (" " + npy.get("dtype", "") if npy else "")
        print("%-46s %12.3f  %s" % (r["path"], r["bytes"] / 1e9, shape))
    print("\nkaggle anchors: %s" % ("present" if doc.get("kaggle_server_files") else "absent"))
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="corpus checksum ledger / verifier")
    ap.add_argument("cmd", choices=["build", "verify", "summary", "refs"])
    ap.add_argument("--root", type=pathlib.Path, default=ROOT_DEFAULT)
    ap.add_argument("--out", type=pathlib.Path, default=LEDGER)
    ap.add_argument("--no-kaggle", action="store_true", help="skip the server-side anchor calls")
    a = ap.parse_args()
    if a.cmd == "build":
        return build(a.root, not a.no_kaggle, a.out)
    if a.cmd == "verify":
        return verify(a.root, a.out)
    if a.cmd == "summary":
        return summary(a.out)
    for ref, why in DATASETS:
        print("%-48s %s" % (ref, why))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
