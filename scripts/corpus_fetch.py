# -*- coding: utf-8 -*-
"""Land the training corpus on a FRESH box (rented GPU / a colleague's machine / a new laptop).

The 2026-09 policy: Kaggle GPU is for submissions only, experiments run on TPU, and if the
TPU queue is still the bottleneck the owner rents GPU elsewhere.  A rented box needs exactly
this much state to rerun our folds:

    python3 scripts/corpus_fetch.py --dest /data/corpus                 # download + unpack
    python3 scripts/corpus_fetch.py --dest /data/corpus --layout /kaggle  # + Kaggle layout
    python3 scripts/corpus_manifest.py verify --root /data/corpus        # MUST print VERIFY OK

~22 GB, all of it CC0 (plus one tiny competition CSV we are entitled to download), none of it
bulk competition data.  Do NOT copy the 570 GB competition DICOM: it is licence-restricted and
the training path never reads it (the corpus is already decoded slices).

WHY `--layout` IS THE WHOLE TRICK
    `kernel-raptor-train` discovers its inputs by walking `/kaggle/input` and writes to
    `HERE = /kaggle/working`.  Recreating that directory shape on the rented box -- as
    hardlinks, so it costs no extra space -- means **the same script file runs unmodified**,
    with no local fork to drift from the Kaggle runs.  For a CV instrument that must stay
    comparable fold-for-fold, code identity is worth more than convenience.

Cross-platform on purpose: this is the script that runs on Linux, not on the Windows box.
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    from corpus_manifest import DATASETS, EXPECT, SINGLE_FILE  # single source of truth
except Exception:  # stand-alone copy: keep it identical to corpus_manifest.DATASETS
    DATASETS = [
        ("dreaddevelopment/knee-raptor-corpus", "part 1 of 2 (all_vols.npy, 3200 studies)"),
        ("dreaddevelopment/knee-raptor-corpus-ext", "part 2 of 2 (extra_vols.npy, 1207 studies)"),
        ("flight0234/rsna-knee-hybrid-report-labels", "training labels"),
        ("ppxl16/rsna-knee-cv-folds", "OUR folds -- required for fold-comparable reruns"),
    ]
    EXPECT = {"all_vols.npy": (3200, 44, 336, 336), "extra_vols.npy": (1207, 44, 336, 336)}
    SINGLE_FILE = {
        "flight0234/rsna-knee-hybrid-report-labels": "report_labels_v4hybrid.csv",
        "ppxl16/rsna-knee-cv-folds": "fold_file.json",
    }

COMPETITION = "rsna-knee-abnormality-detection"
COMP_FILES = ["train.csv"]      # only the label/list CSV; the 570 GB of DICOM is NOT migrated


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return "%.2f %s" % (n, u)
        n /= 1024.0


def env_report(dest: pathlib.Path) -> None:
    print("python      %s" % sys.version.split()[0])
    print("platform    %s" % sys.platform)
    exe = shutil.which("kaggle")
    print("kaggle CLI  %s" % (exe or "MISSING -- pip install kaggle, then put a token in ~/.kaggle/"))
    try:
        import torch
        print("torch       %s  cuda=%s  %s" % (torch.__version__, torch.cuda.is_available(),
                                               torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))
    except Exception as e:
        print("torch       not importable (%s)" % e)
    try:
        import timm
        print("timm        %s" % timm.__version__)
    except Exception as e:
        print("timm        not importable (%s)" % e)
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(str(dest)).free
    print("dest        %s   free %s   (need ~%.0f GB: zips + extracted)" % (dest, human(free), 22 * 2.2))
    if free < 22 * 2.2 * (1 << 30):
        print("!! WARNING: tight -- zips alone would fit but extraction may not")


def download(dest: pathlib.Path) -> int:
    """Prefer `kaggle_resume.py` when it sits next to us.

    `kaggle datasets download` truncates a partial file and restarts whenever the server
    answers a Range request with 200; on 2026-09-24 that threw away 7.30 GB -> 6.80 GB of a
    22 GB move.  The resume script only ever appends, and retries the flaky redirect to
    storage.googleapis.com.  Fall back to the CLI if the sibling script is missing.
    """
    zdir = dest / "zips"
    zdir.mkdir(parents=True, exist_ok=True)
    resumer = pathlib.Path(__file__).resolve().parent / "kaggle_resume.py"
    rc = 0
    for ref, why in DATASETS:
        print("\n=== dataset %s  (%s)" % (ref, why), flush=True)
        if resumer.exists():
            # --outdir, not --out: the resume script then keeps each dataset's REAL file name
            # (knee-raptor-corpus.zip / report_labels_v4hybrid.csv / fold_file.json), which is
            # what the manifest ledger and the layout mirror both key off.
            p = subprocess.run([sys.executable, str(resumer), "--ref", ref, "--outdir", str(zdir)])
        else:
            p = subprocess.run(["kaggle", "datasets", "download", "-d", ref, "-p", str(zdir)])
        if p.returncode != 0:
            print("!! download failed rc=%d" % p.returncode)
            rc = 1
    for f in COMP_FILES:
        print("\n=== competition %s / %s" % (COMPETITION, f), flush=True)
        for attempt in range(1, 6):
            p = subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION,
                                "-f", f, "-p", str(zdir)])
            if p.returncode == 0 and (zdir / f).exists():
                break
            print("   retry %d/5 for %s (flaky redirect is normal)" % (attempt, f), flush=True)
            time.sleep(5 * attempt)
        else:
            print("!! competition file failed; attach %s by hand if your token lacks access" % f)
            rc = 1
    return rc


def unzip(dest: pathlib.Path) -> int:
    """Stream-extract every zip into data/<slug>/.

    One directory per dataset, because that is what the Kaggle layout mirror needs.  zipfile,
    not the unzip binary -- and not `Expand-Archive`, which chokes on 11 GB members.
    """
    zdir, ddir = dest / "zips", dest / "data"
    ddir.mkdir(parents=True, exist_ok=True)
    for z in sorted(zdir.glob("*.zip")):
        slug = z.stem
        out = ddir / slug
        out.mkdir(parents=True, exist_ok=True)
        print("\n=== unpack %s -> %s (%s)" % (z.name, out, human(z.stat().st_size)), flush=True)
        t0 = time.time()
        with zipfile.ZipFile(z) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            total = sum(m.file_size for m in members) or 1
            done = 0
            for m in members:
                target = out / m.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.stat().st_size == m.file_size:
                    done += m.file_size
                    continue
                with zf.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, 8 << 20)
                done += m.file_size
                sys.stdout.write("\r  %-30s %5.1f%%  %6.1f MB/s  " % (
                    m.filename[:30], 100.0 * done / total, done / max(time.time() - t0, 1e-6) / 1e6))
                sys.stdout.flush()
        sys.stdout.write("\r" + " " * 74 + "\r")
    # Single-file datasets (labels CSV, folds JSON) are not zips: stage them into data/<slug>/
    # so that (a) the layout mirror is uniform and (b) the ledger covers the extracted copy too.
    for ref, fname in SINGLE_FILE.items():
        src = zdir / fname
        if src.exists():
            out = ddir / ref.split("/")[-1]
            out.mkdir(parents=True, exist_ok=True)
            how = link(src, out / fname)
            print("=== staged %-38s -> data/%s/  (%s)" % (fname, ref.split("/")[-1], how))
    return 0


def link(src: pathlib.Path, dst: pathlib.Path) -> str:
    """Hardlink, else symlink, else copy.  Returns which one happened."""
    if dst.exists():
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def layout(dest: pathlib.Path, root: pathlib.Path) -> int:
    """Recreate /kaggle/input/... so `kernel-raptor-train` runs with ZERO code changes."""
    data = dest / "data"
    made = {}
    for ref, _ in DATASETS:
        owner, slug = ref.split("/")
        src = data / slug
        if not src.is_dir():
            continue
        target = root / "input" / "datasets" / owner / slug
        for f in sorted(src.rglob("*")):
            if f.is_file():
                how = link(f, target / f.relative_to(src))   # hardlink/symlink/copy
                made[how] = made.get(how, 0) + 1
    # competition: train.csv only
    cav = dest / "zips" / "train.csv"
    if cav.exists():
        link(cav, root / "input" / "competitions" / COMPETITION / "train.csv")
    work = root / "working"
    work.mkdir(parents=True, exist_ok=True)
    print("\n=== layout under %s" % root)
    for k, v in sorted(made.items()):
        print("   %-9s %d files" % (k, v))
    print("   %s" % (root / "input" / "datasets" / "*" / "*"))
    print("   %s  (HERE for kernel-raptor-train's outputs)" % work)
    print("   -> run:  python3 <repo>/kernel-raptor-train/knee-mri-training-the-twelve-finding-model.py \\")
    print("              --labels /kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/"
          "report_labels_v4hybrid.csv \\")
    print("              --folds 5 --fold 0 --fold_file /kaggle/input/datasets/ppxl16/rsna-knee-cv-folds/"
          "fold_file.json --arch convnext_base --res 336 --epochs 12 --tag rent0")
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="fetch the training corpus onto a fresh box")
    ap.add_argument("--dest", type=pathlib.Path, default=pathlib.Path("/data/corpus"))
    ap.add_argument("--layout", type=pathlib.Path, default=None,
                    help="mirror the Kaggle mount layout under this root (e.g. /kaggle). "
                         "This is what lets kernel-raptor-train run unmodified.")
    ap.add_argument("--zips-only", action="store_true")
    ap.add_argument("--unpack-only", action="store_true")
    ap.add_argument("--layout-only", action="store_true")
    ap.add_argument("--env", action="store_true", help="just report the environment")
    a = ap.parse_args()
    env_report(a.dest)
    if a.env:
        return 0
    rc = 0
    if not (a.unpack_only or a.layout_only):
        rc |= download(a.dest)
    if not (a.zips_only or a.layout_only):
        rc |= unzip(a.dest)
    if a.layout:
        rc |= layout(a.dest, a.layout)
    print("\n== next: python3 scripts/corpus_manifest.py verify --root %s" % a.dest)
    print("   (ship state/CORPUS_MANIFEST.json with the code, or copy it from the Windows box)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
