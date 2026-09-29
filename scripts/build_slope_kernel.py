"""Build `kernels/kernel-slope` — the SUBMISSION pipeline run over N TRAINING studies.

WHY THIS KERNEL EXISTS
    Efficiency = AUC / (Benchmark - max AUC) + RuntimeSeconds / 32400, and our AUC is pinned by the
    ensemble, so the ONLY term we can move is runtime. `docs/EFFICIENCY-SLOPE.md` measured one stage
    properly (`a5` = 0.849 s/study -> 19.0 min at 1322 studies) but found that the other six per-study
    stages have **no second study count anywhere on disk**, so their slopes are extrapolation. This
    kernel produces that second point by running the whole pipeline over ~200 training studies.

WHY IT IS A PORT AND NOT A NEW HARNESS
    `kernel-e2oof` already contains the mechanism: it repoints the COHORT rather than copying files. But
    its notebook is TRUNCATED — 23 code cells, ending at the `dinov2` stage — so it can never measure the
    CoAt family or Raptor. `kernel-0943cf` has the full 32-cell pipeline but no OOF switch. This script
    takes the full one and ports the OOF block into it, extracting that block VERBATIM from `kernel-e2oof`
    rather than retyping it.

⚠️ THE PATCH NOBODY ELSE NEEDED, AND THE ONE THAT WOULD SILENTLY RUIN THE RUN
    `kernel-0943cf` cell[37] hard-codes `SERIES_ROOT = COMP / 'test_series'` and then
        raise FileNotFoundError('Missing test_series; train-series fallback is forbidden')
    `kernel-e2oof` never hit this because it stops before cell[37]. So pointing the cohort at training
    studies while the images still come from `test_series` would either crash (loud, fine) or, if the
    guard were simply removed, measure 200 training study IDs against 3 test studies' pixels — a
    silent-degradation failure of exactly the kind `kernel-e2oof`'s own comment warns about. This script
    therefore repoints `SERIES_ROOT` to `train_series` whenever the OOF cohort is active.

Every patch is applied by exact-anchor replacement and asserts it matched EXACTLY ONCE, and every patched
code cell is `compile()`d before anything is written. A patch that silently does not apply is the failure
mode this file is built to prevent.

Usage:
    python scripts/build_slope_kernel.py --limit 200
    python scripts/build_slope_kernel.py --limit 200 --push
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kernels" / "kernel-0943cf" / "rsna-knee-restructured-version-3.ipynb"
OOF = ROOT / "kernels" / "kernel-e2oof" / "rsna-knee-restructured-version-3.ipynb"
DST = ROOT / "kernels" / "kernel-slope"
NB_NAME = "rsna-knee-restructured-version-3.ipynb"


def cell_source(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def set_source(cell, text: str) -> None:
    # Keep the notebook's own storage convention: a list of lines, each keeping its newline.
    cell["source"] = text.splitlines(keepends=True)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"ANCHOR FAILURE [{label}]: expected exactly 1 match, found {n}")
    return text.replace(old, new, 1)


def extract_oof_block() -> str:
    """Pull the OOF block verbatim out of kernel-e2oof's cell[32]."""
    nb = json.loads(OOF.read_text(encoding="utf-8"))
    src = cell_source(nb["cells"][32])
    start = src.index("    # ---- OOF study selection")
    end = src.index("    log(f'test header pass:")
    block = src[start:end]
    if not block.rstrip().endswith("hte = annotate(walk('train_series'))"):
        raise SystemExit("ANCHOR FAILURE [oof-block]: block does not end where expected")
    for must in ("RSNA_OOF_FOLD", "RSNA_OOF_LIMIT", "_RSNA_OOF_ONLY", "rsna_initialize(_want)"):
        if must not in block:
            raise SystemExit(f"ANCHOR FAILURE [oof-block]: missing {must!r}")
    return block


def indent(block: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "".join((pad + ln if ln.strip() else ln) for ln in block.splitlines(keepends=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200,
                    help="how many TRAINING studies to run the pipeline over")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # ── patch 1: the walk() study filter ─────────────────────────────────────────────
    c19 = cell_source(cells[19])
    c19 = replace_once(
        c19,
        "def walk(split):\n",
        "# ---- slope-probe patch: restrict the header scan to the OOF cohort --------------\n"
        "# Empty set means no filtering, i.e. exactly the parent's behaviour. Without this the header\n"
        "# pass reads every series in the split (a measured 8.7 h for a full train_series pass), which\n"
        "# would consume the run before any model loads.\n"
        "_RSNA_OOF_ONLY = set()\n\n\n"
        "def walk(split):\n",
        "walk-def",
    )
    c19 = replace_once(
        c19,
        "    for study in os.scandir(base):\n        if study.is_dir():\n",
        "    for study in os.scandir(base):\n"
        "        if _RSNA_OOF_ONLY and study.name not in _RSNA_OOF_ONLY:\n"
        "            continue\n"
        "        if study.is_dir():\n",
        "walk-filter",
    )
    set_source(cells[19], c19)

    # ── patch 2: the cohort repoint, replacing the parent's test.csv setup ───────────
    block = indent(extract_oof_block(), 4)
    c32 = cell_source(cells[32])
    parent_setup = (
        "    test_df = pd.read_csv(ROOT / 'test.csv')\n"
        "    test_series = pd.read_csv(ROOT / 'test_series.csv')\n"
        "    plane_map = dict(zip(test_series['SeriesInstanceUID'], test_series['Anatomical_Plane']))\n"
        "    hte = annotate(walk('test_series'))\n"
    )
    oof_setup = (
        "    # ---- slope-probe: OOF cohort, or exactly the parent's test cohort ----------------\n"
        "    _oof_fold = os.environ.get('RSNA_OOF_FOLD', '0')\n"
        "    _oof_limit = int(os.environ.get('RSNA_OOF_LIMIT', '0') or 0)\n"
        "    if _oof_limit:\n"
        + block
        + "    else:\n"
        + indent(parent_setup, 4)
    )
    c32 = replace_once(c32, parent_setup, oof_setup, "cohort-setup")
    set_source(cells[32], c32)

    # ── patch 3: the image source. THE ONE THAT SILENTLY RUINS THE RUN ──────────────
    c37 = cell_source(cells[37])
    old_root = (
        "SERIES_ROOT = COMP / 'test_series'\n"
        "if not SERIES_ROOT.is_dir():\n"
        "    raise FileNotFoundError('Missing test_series; train-series fallback is forbidden')\n"
    )
    new_root = (
        "# ---- slope-probe patch. In OOF mode the cohort is TRAINING studies, so the pixels must come\n"
        "# from train_series. The parent forbids that fallback on purpose (a train fallback in a real\n"
        "# submission would score nothing), so the switch is explicit and keyed on the same cohort set\n"
        "# the OOF block fills in -- never on a missing directory.\n"
        "_OOF_ACTIVE = bool(globals().get('_RSNA_OOF_ONLY'))\n"
        "SERIES_ROOT = COMP / ('train_series' if _OOF_ACTIVE else 'test_series')\n"
        "if not SERIES_ROOT.is_dir():\n"
        "    raise FileNotFoundError(\n"
        "        'Missing %s; cohort is %s'\n"
        "        % (SERIES_ROOT.name, 'OOF/train' if _OOF_ACTIVE else 'test'))\n"
        "if _OOF_ACTIVE:\n"
        "    log('slope probe: OOF active -> reading pixels from %s for %d studies'\n"
        "        % (SERIES_ROOT, len(_RSNA_OOF_ONLY)))\n"
    )
    c37 = replace_once(c37, old_root, new_root, "series-root")
    set_source(cells[37], c37)

    # ── patch 4: set the env the OOF block reads ────────────────────────────────────
    c2 = cell_source(cells[2])
    c2 = replace_once(
        c2,
        'print(f"[config] preset={PRESET} ',
        "# ---- slope-probe: run the pipeline over TRAINING studies so the per-study slope can be\n"
        "# measured. See scripts/build_slope_kernel.py and docs/EFFICIENCY-SLOPE.md.\n"
        "os.environ['RSNA_OOF_FOLD'] = '%s'\n"
        "os.environ['RSNA_OOF_LIMIT'] = '%d'\n\n"
        'print(f"[config] preset={PRESET} ' % (a.fold, a.limit),
        "env-config",
    )
    set_source(cells[2], c2)

    # ── patch 5: the frontier validator's reference cohort ──────────────────────────
    # The parent validates the frontier submission against the REAL test.csv. That is right for a
    # submission and wrong for an OOF run: the OOF block wrote _pipeline_stage.csv for the TRAINING
    # cohort, so the check fails on row count (200 vs 3 visible studies) with
    #     ValueError: public frontier: row count or StudyInstanceUID uniqueness failed
    # which is what killed the first push of this kernel -- and, by the same mechanism, the e2oof
    # kernel before it, whose log also stops right after the dinov2 stage. The check is KEPT, not
    # deleted: its job is to catch a frontier submission that does not match the cohort it claims,
    # and that job is still worth doing -- it just needs the right cohort.
    c33 = cell_source(cells[33])
    # ⚠️ NO trailing "\n" on either side: this line is the LAST line of cell[33], so it carries no
    # newline, and the first version of this patch failed its own exactly-once assertion (found 0) for
    # that reason alone. Asserting is what turned a silently-unpatched notebook into a loud error.
    old_validate = (
        "    _v37_validate_submission('_pipeline_stage.csv', pd.read_csv(ROOT / 'test.csv'), 'public frontier')"
    )
    new_validate = (
        "    # ---- slope-probe patch: validate against the cohort this run actually predicted ------\n"
        "    _pf_ref = pd.read_csv(ROOT / 'test.csv')\n"
        "    if globals().get('_RSNA_OOF_ONLY'):\n"
        "        _pf_ref = pd.DataFrame({'StudyInstanceUID': sorted(_RSNA_OOF_ONLY)})\n"
        "        log('slope probe: frontier validated against the OOF cohort (%d studies)' % len(_pf_ref))\n"
        "    _v37_validate_submission('_pipeline_stage.csv', _pf_ref, 'public frontier')"
    )
    c33 = replace_once(c33, old_validate, new_validate, "frontier-validate")
    set_source(cells[33], c33)

    # ── compile gate: a patched cell that does not parse must never reach Kaggle ────
    bad = 0
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        try:
            compile(cell_source(c), f"cell{i}", "exec")
        except SyntaxError as e:
            bad += 1
            print("SYNTAX ERROR in cell[%d]: %s (line %s)" % (i, e.msg, e.lineno))
    if bad:
        raise SystemExit("%d patched cell(s) do not compile; nothing written" % bad)

    # ── metadata: same sources as the arm we are profiling, new id/title ────────────
    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = "ppxl16/rsna-knee-slope"
    meta["title"] = "RSNA knee slope"
    meta["code_file"] = NB_NAME
    meta["enable_gpu"] = True
    meta["enable_tpu"] = False
    meta["enable_internet"] = False
    meta["is_private"] = True

    DST.mkdir(parents=True, exist_ok=True)
    (DST / NB_NAME).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote %s  (OOF limit=%d fold=%s, %d code cells, all compile)"
          % (DST.relative_to(ROOT), a.limit, a.fold,
             sum(1 for c in cells if c.get("cell_type") == "code")))

    if a.push:
        from kaggle import api                      # imported here: auth is a network call
        api.kernels_push(str(DST))
        print("PUSHED ppxl16/rsna-knee-slope")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
