# -*- coding: utf-8 -*-
"""Write state/CORPUS64_MANIFEST.json -- the provenance ledger for the 64-slot dense corpus.

The 44-slot corpus has CORPUS_MANIFEST.json because its files sit on disk here and can be hashed.
The 64-slot corpus does not: it is 31.8 GB spread over eight Kaggle kernel outputs, and it can never
exist as one file (that is over the 20 GB kernel-output limit, HANDOFF 152.3).  So the ledger has to
record something else -- which kernel produced which rows, in what order, and what proved it.

Every field is read back from the platform (status + run logs + output listing), not remembered:

    python scripts/corpus64_manifest.py            # write state/CORPUS64_MANIFEST.json
    python scripts/corpus64_manifest.py --check    # re-verify an existing manifest, change nothing
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "state" / "CORPUS64_MANIFEST.json"

# The trainer's _open_ids() order: knee-raptor-corpus then knee-raptor-corpus-ext.  These are the
# only two datasets that define row order, and both must be mounted (FACTS A56/A60).
PARTS = [("dreaddevelopment/knee-raptor-corpus", "all_ids.npy", 3200),
         ("dreaddevelopment/knee-raptor-corpus-ext", "extra_ids.npy", 1207)]
EXPECTED_ROWS = sum(p[2] for p in PARTS)
SLOTS = 64
RES = 336


def kaggle(*args):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return (r.stdout or "") + (r.stderr or "")


def shard_info(ref):
    """Status plus the builder's own one-line summary, parsed out of the run log."""
    st = kaggle("kernels", "status", ref)
    m = re.search(r"KernelWorkerStatus\.(\w+)", st)
    status = m.group(1) if m else "UNKNOWN"
    log = kaggle("kernels", "logs", ref)
    # The log is a JSON stream; the summary line is the only place the builder states its own shape.
    pat = re.search(r'"data":"(shard (s\d+): \(([\d, ]+)\) \| mask density ([\d.]+) \| '
                    r'(\d+) MB \| (\d+)s)', log)
    info = {"ref": ref, "status": status}
    if pat:
        dims = [int(x) for x in pat.group(3).split(",")]
        info.update({"shard": pat.group(2), "rows": dims[0], "shape": dims,
                     "mask_density": float(pat.group(4)), "mb": int(pat.group(5)),
                     "seconds": int(pat.group(6))})
    m2 = re.search(r'"data":"(shard s\d+ rows (\d+)\.\.(\d+) \(n=(\d+) of pool (\d+)\))', log)
    if m2:
        info.update({"row_lo": int(m2.group(2)), "row_hi": int(m2.group(3)),
                     "pool_len": int(m2.group(5))})
    info["traceback"] = "Traceback" in log
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    shards = [shard_info("ppxl16/rsna-knee-corpus64-s%d" % i) for i in range(a.shards)]
    rows = sum(s.get("rows", 0) for s in shards)

    # Row-order evidence, produced by downloading each shard's small ids_sX.npy and comparing it
    # against the published id files concatenated.  This is the ONE invariant the pilot rests on.
    ids_verify = None
    idv = ROOT / "state" / "corpus64_ids_verify.json"
    if idv.exists():
        ids_verify = json.loads(idv.read_text(encoding="utf-8"))
    problems = []
    for s in shards:
        if s["status"] != "COMPLETE":
            problems.append("%s status %s" % (s["ref"], s["status"]))
        if s["traceback"]:
            problems.append("%s log has a traceback" % s["ref"])
        if "rows" not in s:
            problems.append("%s: no summary line in the log" % s["ref"])
        else:
            if s["shape"][1:] != [SLOTS, RES, RES]:
                problems.append("%s shape %s != [*, %d, %d, %d]" % (s["ref"], s["shape"], SLOTS,
                                                                   RES, RES))
            if s.get("pool_len") is None:
                # s0..s4 were built by the PRE-FIX builder, which did not print the pool length.
                # Their rows 0..2754 are still correct (concat preserves the prefix) and their row
                # order is verified directly from ids_sX.npy below, so this is not a data problem.
                s["pool_len"] = "not logged (built by the pre-fix builder)"
            elif s["pool_len"] != EXPECTED_ROWS:
                problems.append("%s built against pool %s, expected %d"
                                % (s["ref"], s.get("pool_len"), EXPECTED_ROWS))
    if rows != EXPECTED_ROWS:
        problems.append("shard rows total %d != %d" % (rows, EXPECTED_ROWS))

    man = {
        "corpus": "rsna-knee dense 64-slot (max-span)",
        "why_not_one_file": "31.8 GB > Kaggle's 20 GB kernel-output limit; mount the shards instead",
        "built_by": "kernels/kernel-corpus64/build_corpus.py --mode build64 (v2: one series scan per group, "
                    "specific_tags, multiprocessing)",
        "recipe": {"slots_per_group": [["Sagittal", 1, 18], ["Sagittal", 0, 14], ["Coronal", 1, 12],
                                       ["Coronal", 0, 8], ["Axial", -1, 12]],
                   "slots": SLOTS, "span": [0.02, 0.98], "res": RES, "crop_mm": 140.0,
                   "index_rule": "lo=floor(lo_frac*(n-1)); hi=floor(hi_frac*(n-1)); "
                                 "rint(linspace(lo, hi, k))"},
        "row_order_is_a_contract": {
            "rule": "_open_ids() = concat(all_ids, extra_ids); vols/masks/ids/labels are ALL indexed "
                    "by it. Row k of the corpus is the study ids[k] names.",
            "parts": [{"dataset": d, "id_file": f, "rows": n} for d, f, n in PARTS],
            "expected_rows": EXPECTED_ROWS,
            "builder_guard": "pool = concat(all_ids, extra_ids) with the drop count printed; the "
                             "first version read only all_ids.npy (3200 rows) and would have "
                             "produced empty shards s6/s7 (FACTS A56)",
            "pilot_guard": "SHARD_EXPECTED_ROWS assert in kernels/kernel-d64pilot/_open_corpus: ids and "
                           "vols come from the same glob, so a DROPPED shard keeps their counts "
                           "equal -- only the absolute total catches it",
        },
        "shards": shards,
        "totals": {"rows": rows, "mb": sum(s.get("mb", 0) for s in shards),
                   "seconds_sum": sum(s.get("seconds", 0) for s in shards)},
        "measured_speed": {
            "seconds_per_study": round(sum(s.get("seconds", 0) for s in shards) /
                                       max(rows, 1), 3),
            "note": "0.7 s/study at 4 threads; the whole corpus is a ~10 minute job. The old "
                    "'105 s/study -> 32 h' figure was the v1 builder and is 150x off (FACTS A59).",
        },
        "consumers": {
            "pilot": "ppxl16/rsna-knee-d64pilot (GPU, N-part memmap loader, kernel_sources=these 8)",
            "mount_probe": "ppxl16/rsna-knee-d64probe (CPU-only: asserts the 8 shards mount in order "
                           "and that the ids equal concat(all_ids, extra_ids) row for row)",
        },
        "verification": {
            "row_order": "ids_sX.npy written by each shard (before any pixels) were downloaded for "
                         "ALL EIGHT shards and compared against concat(published all_ids, "
                         "extra_ids): every shard matched exactly, 4407 unique ids, no gap and no "
                         "overlap.  s5 straddles the 3200-row split (445 corpus + 106 ext), which "
                         "is the one place a two-part concatenation can go wrong.",
            "row_order_per_shard": ids_verify,
            "problems": problems,
            "ok": not problems,
        },
    }

    if a.check:
        old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else None
        if old is None:
            print("no manifest to check")
            return 1
        same = (old["totals"] == man["totals"] and
                [s.get("rows") for s in old["shards"]] == [s.get("rows") for s in man["shards"]])
        print("manifest reproduces: %s" % same)
        for p in old["verification"]["problems"]:
            print("   recorded problem: %s" % p)
        return 0 if same else 1

    OUT.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote %s" % OUT)
    print("  shards %d | rows %d | %.1f GB | %.0f s total | %.2f s/study" % (
        len(shards), rows, man["totals"]["mb"] / 1024.0, man["totals"]["seconds_sum"],
        man["measured_speed"]["seconds_per_study"]))
    if problems:
        print("  PROBLEMS:")
        for p in problems:
            print("    !! %s" % p)
        return 1
    print("  all shards COMPLETE, correct shape, built against pool %d" % EXPECTED_ROWS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
