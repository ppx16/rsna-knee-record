"""Inventory the mounted sources for published OOF / audit artifacts.

Why this matters: the fusion-layer knobs (Rad calibration weight, the `_RAD_EXCLUDE` list, the
per-label outer routing) were closed as "unjudgeable" because the only instruments we had were
gold-58 (+-0.021) and the public LB (+-0.008).  But the community publishes OOF predictions --
`sofiaanjenje/rsna-knee-e13-train` alone carries a 4407-study, 12-label OOF with fold and is_gold
columns.  If enough members have published OOFs, those knobs can be re-opened with a real
instrument, on CPU, for free -- and that decides whether renting a GPU box is needed at all.

Note: `kaggle kernels files` reports nonsense sizes (a 63 MB checkpoint listed as 885 bytes), so
this only inventories NAMES.  Sizes must be confirmed by actually downloading.

Usage:
    python scripts/find_oof_artifacts.py
"""
import argparse
import json
import re
import sys

PAT = re.compile(r"oof|out.?of.?fold|audit|fold|valid|holdout|\bcv\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="kernel-0943/kernel-metadata.json")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    meta = json.load(open(args.meta, encoding="utf-8"))
    hits = 0
    for kind, key in (("dataset", "dataset_sources"), ("kernel", "kernel_sources")):
        for ref in meta.get(key) or []:
            try:
                if kind == "dataset":
                    files = api.dataset_files_list(ref)
                else:
                    files = api.kernels_list_files(ref).files
            except Exception as exc:
                print("%-8s %-52s LIST FAILED %s: %s" % (kind, ref, type(exc).__name__, exc))
                continue
            names = [getattr(f, "name", str(f)) for f in files]
            marked = [n for n in names if PAT.search(n)]
            print("%-8s %-52s %2d file(s)%s" % (
                kind, ref, len(names),
                ("   <-- " + ", ".join(marked)) if marked else ""))
            hits += len(marked)
    print()
    print("candidate OOF-ish artifacts: %d" % hits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
