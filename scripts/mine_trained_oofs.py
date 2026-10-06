"""Mine the OOFs of arms we already trained but never gated -- the cheapest Phase 2 step.

WHY THIS EXISTS
    48 training kernels exist; 29 of their OOFs were never fetched, so 29 arms have been trained,
    paid for, and never put through `member_add_test.py`.  Before any new TPU hour is spent, those
    must be gated: they cost nothing and one of them may already pass.

    It also exists because the first dig corrected the plan twice (FACTS A160): seven architectures
    have already been tried (ConvNeXt among them), and gating the never-tested convnext_large@224
    returned no signal -- so "train a stronger 2D backbone" was about to become repeat work.

TRAP THAT COST TIME
    `kernels_output(file_pattern=...)` takes a REGEX, not a glob.  Passing 'raptor_oof_*.npz' raises
    nothing and quietly matches nothing, so the call returns only the .log and looks like a success.
    The working pattern is 'raptor_oof_.*fold\\d+\\.npz'.  ⚠️ It still pulls the kernel log alongside.

    Also: some older kernels answer `Permission 'kernels.get' was denied` (they are gone or not
    ours), and `kernels_output` re-downloads the 300 MB checkpoint unless the pattern excludes it --
    hence the pattern, and hence the .pt cleanup below.

Usage:
    python scripts/mine_trained_oofs.py                 # report what is missing
    python scripts/mine_trained_oofs.py --fetch         # fetch the missing OOF npz files
    python scripts/mine_trained_oofs.py --fetch --only nxl2240,c2240
"""
from __future__ import annotations


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
import glob
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def survey():
    """-> list of dicts: kernel dir, id, tag, fold, arch, res, and whether we hold its OOF."""
    have = set()
    for pat in ("downloads/cv/**/raptor_oof_*.npz", "downloads/rent/**/raptor_oof_*.npz",
                "downloads/a5lite_*/raptor_oof_*.npz"):
        for p in glob.glob(str(ROOT / pat), recursive=True):
            m = re.search(r"raptor_oof_([A-Za-z0-9]+?)(?:_(ep\d+|epens|swa))?_fold(\d+)\.npz$",
                          os.path.basename(p))
            if m and not m.group(2):
                have.add("%s@f%s" % (m.group(1), m.group(3)))

    out = []
    for d in sorted(glob.glob(str(ROOT / "kernels" / "kernel-*"))):
        src = os.path.join(d, "knee-mri-training-the-twelve-finding-model.py")
        meta = os.path.join(d, "kernel-metadata.json")
        if not (os.path.isfile(src) and os.path.isfile(meta)):
            continue
        t = open(src, encoding="utf-8", errors="replace").read()
        arch = re.findall(r"'--arch',\s*'([^']+)'", t)
        tag = re.findall(r"'--tag',\s*'([^']+)'", t)
        fold = re.findall(r"'--fold',\s*'(\d+)'", t)
        res = re.findall(r"'--res',\s*'?(\d+)", t)
        if not (arch and tag):
            continue
        try:
            kid = json.load(open(meta, encoding="utf-8")).get("id")
        except Exception:
            kid = None
        key = "%s@f%s" % (tag[-1], fold[-1] if fold else "?")
        out.append({"dir": os.path.basename(d), "id": kid, "tag": tag[-1],
                    "fold": fold[-1] if fold else "?", "arch": arch[-1],
                    "res": res[-1] if res else "?", "key": key, "have": key in have})
    return out


def fetch(rows, only=None):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    # regex, NOT a glob -- see the module docstring
    pattern = r"raptor_oof_.*fold\d+\.npz"
    done, failed = [], []
    for r in rows:
        if r["have"] or not r["id"]:
            continue
        if only and r["tag"] not in only:
            continue
        dst = ROOT / "downloads" / "cv" / r["tag"]
        dst.mkdir(parents=True, exist_ok=True)
        try:
            fs, _ = api.kernels_output(r["id"], path=str(dst), file_pattern=pattern, quiet=True)
        except Exception as e:
            failed.append((r["tag"], type(e).__name__, str(e)[:90]))
            print("[fetch] %-12s FAILED  %s %s" % (r["tag"], type(e).__name__, str(e)[:80]))
            continue
        # the pattern cannot exclude the log, and must never keep the 300 MB checkpoint
        for junk in dst.glob("*.pt"):
            junk.unlink()
        got = [os.path.basename(x) for x in (fs or []) if x.endswith(".npz")]
        got = got or [p.name for p in dst.glob("raptor_oof_*.npz")]
        if got:
            done.append((r["tag"], got))
            print("[fetch] %-12s OK      %s" % (r["tag"], got))
        else:
            failed.append((r["tag"], "no-npz", "returned %d files" % len(fs or [])))
            print("[fetch] %-12s EMPTY   (kernel had no OOF output)" % r["tag"])
    return done, failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--only", default="", help="comma list of tags")
    a = ap.parse_args(argv)
    rows = survey()
    miss = [r for r in rows if not r["have"]]
    print("training kernels : %d" % len(rows))
    print("already gated    : %d" % (len(rows) - len(miss)))
    print("missing OOF      : %d  <-- free gate tests" % len(miss))
    print()
    print("%-12s %-52s %-6s %s" % ("tag", "arch", "res", "kernel"))
    for r in miss:
        print("%-12s %-52s %-6s %s" % (r["tag"], r["arch"][:52], r["res"], r["dir"]))
    if not a.fetch:
        print("\n(run with --fetch to pull them; no GPU/TPU quota is touched)")
        return 0
    only = [t.strip() for t in a.only.split(",") if t.strip()] or None
    print("\n--- fetching ---")
    done, failed = fetch(rows, only)
    print("\nfetched %d, failed %d" % (len(done), len(failed)))
    if done:
        print("now gate them:  python scripts/member_sweep.py --only %s"
              % ",".join("%s@f%s" % (t, next(r["fold"] for r in rows if r["tag"] == t))
                         for t, _ in done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
