# -*- coding: utf-8 -*-
"""Package trained weights from one or more training runs into a private Kaggle dataset.

The training notebook saves `weights/` (manifest.json + m_*.pt) per run. One run = one
SEED = N_FOLDS members. To grow the ensemble you re-run it with another SEED, then merge
the runs here into a single dataset whose manifest lists every member. Member ids are a
hash of decode config + fold + seed, so the same (fold, seed) re-run deduplicates and
different seeds accumulate.

Workflow:
    1.  Push + run the training notebook (trains, saves weights/ + manifest.json):
            kaggle kernels push -p kernel-train
        (optionally edit the SEED constant in the notebook and push again for more seeds)

    2.  ALWAYS dry-run first. It builds and validates the whole dataset directory and
        prints the exact Kaggle call, without touching the account:
            python scripts/package_weights.py \
                --from-dir downloads/weights-out-224 \
                --dataset ppxl16/rsna-knee-train-weights-224 \
                --title "RSNA Knee train weights 224x6 slices (seed2026)" \
                --dataset-dir downloads/weights-dataset-224 \
                --dry-run

    3.  Drop --dry-run to publish. Existing datasets are versioned automatically;
        new ones are created.

        To merge several runs (one per seed), pass them comma-separated:
            python scripts/package_weights.py \
                --kernel ppxl16/rsna-knee-train-v1,ppxl16/rsna-knee-train-v1-seed2 \
                --dataset ppxl16/rsna-knee-train-weights

    Then add "ppxl16/rsna-knee-train-weights" to an inference notebook's dataset_sources;
    find_weights() locates the manifest and infer_from_package() rank-means the members.

The dataset is created PRIVATE by default, so the weights and the pseudo-labels they
encode stay out of the public leaderboard.

Notes on the Kaggle CLI (all measured on this machine, 2026-09-11):
  * `kaggle datasets create` fails when the *title* is already in use anywhere on Kaggle,
    even by someone else. That - not any path bug - is the collision to expect when
    re-publishing weights. This script detects an existing dataset of your own and
    switches to `datasets version`.
  * Both backslash and forward-slash `-p` paths were probed and both read
    `dataset-metadata.json` correctly, so the path form does not matter. (HANDOFF.md §6
    used to claim otherwise; that claim did not reproduce and has been corrected.)
  * `PYTHONUTF8=1` must be set for the CLI on this zh-CN Windows box or it fails to read
    UTF-8 files. The script sets it for every child process it spawns.
"""

# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded
# `print("... \u26a0 ...")` raises UnicodeEncodeError at that line and kills the script MID-REPORT, while
# everything printed earlier is already on screen -- so the run looks complete. Self-contained on purpose:
# it imports `sys` itself and depends on nothing else in this tree.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream: leave it alone, prints still degrade
    pass
# ----------------------------------------------------------------------------------------------------
import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE = "kaggle"


def child_env():
    """The CLI needs UTF-8 mode on this box; do not rely on the caller's shell."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    return env


def run(cmd, dry=False):
    print("+", " ".join(str(c) for c in cmd))
    if dry:
        return ""
    p = subprocess.run(cmd, check=False, env=child_env(),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.stdout:
        # The CLI's "outdated version" banner goes to stdout and is noise.
        for line in p.stdout.splitlines():
            if line.startswith("Warning: Looks like you're using an outdated"):
                continue
            print("  |", line)
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(str(c) for c in cmd)}")
    return p.stdout or ""


def my_dataset_refs():
    """Refs of every dataset this account owns, for the create-vs-version decision.

    The CLI prints its "outdated version" banner to STDOUT, *ahead of* the CSV header
    (measured 2026-09-11). Both that banner and any blank lines have to be removed
    before parsing: DictReader would otherwise take the banner as the header row, every
    lookup of "ref" would come back empty, and this function would silently degrade to
    "no datasets exist" - which means `create` against a dataset that already exists,
    i.e. a hard failure at exactly the wrong moment.
    """
    p = subprocess.run([KAGGLE, "datasets", "list", "-m", "--csv"],
                       check=False, env=child_env(),
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if p.returncode != 0 or not p.stdout:
        print("  (could not list my datasets; assuming a new dataset)")
        return set()
    lines = [ln for ln in p.stdout.splitlines()
             if ln.strip() and not ln.startswith("Warning: Looks like")]
    if not lines:
        print("  (dataset list came back empty; assuming a new dataset)")
        return set()
    refs = set()
    for row in csv.DictReader(io.StringIO("\n".join(lines))):
        ref = (row.get("ref") or "").strip()
        if ref:
            refs.add(ref.lower())
    print(f"  ({len(refs)} dataset(s) owned by this account)")
    return refs


def load_manifest(root: Path):
    """Locate a manifest.json anywhere under root; fail loudly if none or many."""
    hits = sorted(root.rglob("manifest.json"), key=lambda p: len(p.parts))
    if not hits:
        raise SystemExit(f"no manifest.json found under {root} — did the training run "
                         f"finish and write /kaggle/working/weights/?")
    man_path = hits[0]
    man = json.loads(man_path.read_text(encoding="utf-8"))
    members = man.get("members", [])
    if not members:
        raise SystemExit(f"{man_path} has no members")
    for m in members:
        if not (man_path.parent / m["file"]).is_file():
            raise SystemExit(f"member file missing: {man_path.parent / m['file']}")
    return man_path, man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="ppxl16/rsna-knee-train-v1",
                    help="one or more kernel slugs, comma-separated")
    ap.add_argument("--dataset", default="ppxl16/rsna-knee-train-weights",
                    help="target dataset ref (owner/slug)")
    ap.add_argument("--out", default="downloads/weights-out")
    ap.add_argument("--dataset-dir", default="downloads/weights-dataset")
    ap.add_argument("--title", default=None,
                    help="Kaggle dataset title; MUST be globally unique. Defaults to a "
                         "title derived from --dataset.")
    ap.add_argument("--message", default=None,
                    help="version message, used only when the dataset already exists")
    ap.add_argument("--from-dir", default=None,
                    help="merge already-downloaded seed subdirs under this dir "
                         "(skip the kernels-output download; one subdir per seed)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and validate the dataset directory, print the Kaggle "
                         "call, but do not create or version anything")
    args = ap.parse_args()

    if args.from_dir:
        out = Path(args.from_dir)
        if not out.is_dir():
            raise SystemExit(f"--from-dir {out} is not a directory")
        kernels = [d.name for d in sorted(out.iterdir()) if d.is_dir()]
        if not kernels:
            raise SystemExit(f"no seed subdirs under {out}")
    else:
        kernels = [k.strip() for k in args.kernel.split(",") if k.strip()]
        out = Path(args.out)
        if out.exists() and not args.dry_run:
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

    # 1. Download each run's output (unless --from-dir), then merge the members.
    merged = {"format": 2, "members": [], "history": [], "targets": []}
    by_id = {}
    manifest_dirs = []
    for k in kernels:
        slug = k.replace("/", "_")
        kdir = out / slug
        kdir.mkdir(parents=True, exist_ok=True)
        if not args.from_dir:
            run([KAGGLE, "kernels", "output", k, "-p", str(kdir)], dry=args.dry_run)
        man_path, man = load_manifest(kdir)
        manifest_dirs.append(man_path.parent)
        merged.setdefault("targets", man.get("targets", []))
        for m in man["members"]:
            by_id.setdefault(m["id"], m)  # first run wins; ids are equal by design
        print(f"  {k}: {len(man['members'])} member(s) at {man_path.parent}")

    merged["members"] = list(by_id.values())
    print(f"merged {len(merged['members'])} member(s) from {len(kernels)} run(s)")
    if not merged["members"]:
        raise SystemExit("merged manifest has no members")

    # 2. Flatten into one dataset directory (find_weights expects member files beside
    #    the manifest).
    ddir = Path(args.dataset_dir)
    if ddir.exists():
        shutil.rmtree(ddir)
    ddir.mkdir(parents=True)
    copied = 0
    for m in merged["members"]:
        fname = m["file"]
        for man_dir in manifest_dirs:
            if (man_dir / fname).is_file():
                shutil.copy(man_dir / fname, ddir / fname)
                copied += 1
                break
    (ddir / "manifest.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"copied {copied}/{len(merged['members'])} member files into {ddir}")
    if copied != len(merged["members"]):
        raise SystemExit("some member files were not found next to any manifest — refusing "
                         "to publish an incomplete dataset")

    # 3. dataset-metadata.json, then create (new) or version (already mine).
    title = args.title or args.dataset.split("/")[-1].replace("-", " ").title()
    meta = {"title": title, "id": args.dataset,
            "licenses": [{"name": "Apache-2.0"}]}
    (ddir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2),
                                                encoding="utf-8")

    exists = args.dataset.lower() in my_dataset_refs()
    if exists:
        cmd = [KAGGLE, "datasets", "version", "-p", str(ddir), "--dir-mode", "zip",
               "-m", args.message or f"refresh {len(merged['members'])} member(s)"]
        action = "version (dataset already exists)"
    else:
        cmd = [KAGGLE, "datasets", "create", "-p", str(ddir), "--dir-mode", "zip"]
        action = "create (new dataset)"

    print(f"\ntitle  : {title}")
    print(f"target : {args.dataset}")
    print(f"action : {action}")
    run(cmd, dry=args.dry_run)

    if args.dry_run:
        print(f"\nDRY RUN — nothing was created. {len(merged['members'])} member(s) staged "
              f"in {ddir}. Re-run without --dry-run to publish.")
    else:
        print(f"\ndone. dataset '{args.dataset}' published (private, "
              f"{len(merged['members'])} members). Add it to an inference notebook's "
              f"dataset_sources to use the weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
