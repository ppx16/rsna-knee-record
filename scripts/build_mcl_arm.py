"""Build `kernel-0943cfm`: 0943cf plus the MCL exclusion — the one remaining label-level lever.

WHY THIS ARM EXISTS
    `_RAD_EXCLUDE` decides which findings the RadImageNet head is blended into. It shipped as
    `("Baker's", 'Fracture')`, hand-picked, with a 58-study audit as post-hoc justification.
    C81 falsified the `'Fracture'` half on a 4185-study set and we adopted the fix (`0943cf`).

    The audit's OWN worst label, however, was **MCL** (deltas -0.0159 / -0.0408 / -0.0828 at
    alpha 0.20/0.35/0.50 — worst of all twelve by 2x) and **MCL was never excluded**. So the
    exclusion set was never derived from the audit it cites. Re-measured on the same large-n
    instrument that validated the Fracture fix, MCL is:

        n=3802, 210 positives:  0.8930 -> 0.8821 = -0.0109
        paired-bootstrap CI [-0.0215, -0.0006],  P(>0) = 0.022
        negative in 35/35 {6 label sources} x {5 Rad assemblies}
        per-fold 3/5 negative

    Expected macro gain from excluding MCL: **+0.000904** (the Fracture fix was +0.000824).

WHY THIS IS NOT ONE OF THE CLOSED LEVERS
    The closed list forbids *tuning* fusion weights / per-label routing. This is a different shape:
    it is a BINARY set-membership correction, its direction was PRE-REGISTERED by an earlier
    independent measurement (the 58-study audit named MCL worst, independently of the large-n
    run), no parameter is fitted, and the sign is invariant across 35 perturbations. The arm that
    has already been falsified out-of-sample is `coatnet_w` (-0.0003, n=4407), not this one.

HONEST WEAKNESS (do not bury it)
    MCL is weaker than the Fracture case: the CI only just excludes zero and only 3/5 folds agree.
    The instrument's parent is stage-1 (public20), not the deployed stage-2. So this is built as a
    clean one-token A/B against `0943cf`, and a public-LB move of 0.001 is NOT evidence either way
    (board SE ~= 0.008) — the submission is a guard, not a measurement.

Usage:  python scripts/build_mcl_arm.py [--push]
        python scripts/build_mcl_arm.py --src kernel-0943cfmsft2 \
            --dst kernel-0943cfmsft2m --id ppxl16/rsna-knee-0943cfmsft2m \
            --title "RSNA knee 0943cfmsft2m" [--push]

WHY THE ARGUMENTS EXIST (2026-10-04)
    `0943cfmsft2` (cn0 + fracture fix + our msft2 member) is the one arm with a MEASURED gain
    (rank 708 to 487, in-block percentile 38.89% to 13.66%). MCL is the last change this repo has a
    large-n local measurement for and is not on that arm yet, so the natural next arm is all three
    stacked. Generalising this builder rather than copying it means the ONE-TOKEN edit and its four
    assertions stay identical across every arm built from them.
    Defaults are unchanged, so the original `0943cf -> 0943cfm` invocation still works.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_NAME = "kernel-0943cf"
DST_NAME = "kernel-0943cfm"
KID = "ppxl16/rsna-knee-0943cfm"
TITLE = "RSNA knee 0943cfm"

OLD = '_RAD_EXCLUDE = ("Baker\'s",)'
NEW = '_RAD_EXCLUDE = ("Baker\'s", "MCL")'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_NAME)
    ap.add_argument("--dst", default=DST_NAME)
    ap.add_argument("--id", dest="kid", default=KID)
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    src = ROOT / "kernels" / a.src
    dst = ROOT / "kernels" / a.dst
    if not src.is_dir():
        print("!! source arm missing: %s" % src)
        return 2
    nbs = list(src.glob("*.ipynb"))
    if len(nbs) != 1:
        print("!! expected exactly 1 notebook in %s, found %d" % (src, len(nbs)))
        return 2
    nb_path = nbs[0]

    nb = json.loads(nb_path.read_text(encoding="utf-8", errors="replace"))
    n_cells_before = len(nb["cells"])

    hits = 0
    for c in nb["cells"]:
        txt = "".join(c.get("source", []))
        if OLD in txt:
            hits += txt.count(OLD)
            c["source"] = [txt.replace(OLD, NEW)]
    if hits != 1:
        print("!! _RAD_EXCLUDE matched %d times in %s (need exactly 1) -- refusing to guess"
              % (hits, a.src))
        return 3

    # ---- assertions on the artefact we are about to push --------------------------------
    whole = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert NEW in whole, "the MCL change did not land"
    assert OLD not in whole, "the old exclusion line survived"
    assert 'Fracture' not in whole.split("_RAD_EXCLUDE")[1][:60], \
        "Fracture reappeared in the exclusion set"
    assert len(nb["cells"]) == n_cells_before, "cell count changed"
    print("src=%s -> dst=%s" % (a.src, a.dst))
    print("cells: %d (unchanged)   _RAD_EXCLUDE matched %d time(s)" % (len(nb["cells"]), hits))
    print("new line: %s" % NEW)

    dst.mkdir(parents=True, exist_ok=True)
    (dst / nb_path.name).write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    meta = json.loads((src / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = a.kid
    meta["title"] = a.title
    (dst / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote %s" % dst)
    print("  code_file=%s  gpu=%s  internet=%s  kernel_type=%s  datasets=%d"
          % (meta.get("code_file"), meta.get("enable_gpu"),
             meta.get("enable_internet"), meta.get("kernel_type"),
             len(meta.get("dataset_sources") or [])))

    if not a.push:
        print("\nnot pushed (pass --push).")
        return 0
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        api.kernels_push(str(dst))
        print("pushed -> %s" % a.kid)
    except Exception as e:
        print("push failed: %s: %s" % (type(e).__name__, str(e)[:500]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
