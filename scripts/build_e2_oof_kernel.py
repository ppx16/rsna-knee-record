"""Build the E2-OOF kernel: run the DINOv2 parent arm over TRAINING studies, keep per-member preds.

WHY
---
Every fusion knob we care about (the Rad blend alpha, the per-label `_RAD_EXCLUDE` list, the
calibrator weight, the arm weights) needs `rank(E2)` -- the 20-member DINOv2 parent arm's
out-of-fold ranking over training studies.  The Rad arm's OOF is published (`v52_oof.csv`);
**E2's is not**, and every published artifact is a checkpoint rather than an input, so `rank(E2)`
has to be produced by running the arm against the competition DICOM.  That is why this runs on
Kaggle and not on a rented box: there the DICOM is already mounted (FACTS A84).

WHAT IT CHANGES, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------
The E2 stage is self-contained (`infer_from_package` reads the study list, builds the cache, runs
the members, `_combine`s them, writes the stage csv), so the edit set is four small changes:

  1. `walk()` gets a study filter.  Scanning all of `train_series` costs **8.7 h of header reads
     alone** -- the frontier's own audit records `available_at_start_seconds: 31299` -- so without
     this the run would exhaust its budget before loading a model.
  2. The study list comes from the **published `v52_oof.csv` fold assignment** (that dataset is
     already a source of the arm we derive from), restricted to one fold, with an optional count
     for the smoke test.  Using the publisher's own split is what makes the resulting OOF
     comparable with the published Rad OOF.
  3. `test_series.csv`/`test_series` -> `train_series.csv`/`train_series`, restricted to those
     studies.  `train_series.csv` carries the same required columns (`Anatomical_Plane`,
     `Fat_Suppression`, `Fluid_Sensitive`), asserted by the notebook itself at L1252/L1262.
  4. `main()` replaces the submission validation against `test.csv` -- which cannot hold for a
     training subset -- with a check that the stage covered exactly the studies requested.

**No per-member plumbing is added.**  `bank()` already calls
`rsna_save_predictions('dino_'+id, ids, ...)` for every member, which writes
`/kaggle/working/diagnostics/dino_<id>.npz`.  So per-member predictions come out for free, and
**fold discipline is applied locally**: this kernel must not decide which members are entitled to
predict which study, because that decision has to stay inspectable and re-checkable.

Nothing downstream of the E2 stage is touched, so the A5/Rad stage's hard `test_series` requirement
(`train-series fallback is forbidden`) is never reached.

Usage:
    python scripts/build_e2_oof_kernel.py --fold 0 --limit 48     # smoke test
    python scripts/build_e2_oof_kernel.py --fold 0                # full fold
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
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "kernels"
SRC = KERNELS / "kernel-0943"
if not SRC.is_dir():
    SRC = ROOT / "kernel-0943"          # older flat layout, kept working
SRC_NB = SRC / "rsna-knee-restructured-version-3.ipynb"
NB_NAME = "rsna-knee-restructured-version-3.ipynb"

# ---- the four edits (each must match exactly once in the whole notebook) ------
OLD_WALK = "def walk(split):"
NEW_WALK = """# This `walk` is the parent's, plus a study filter.  Without the filter it reads every series
# header in the split -- the frontier's own audit records available_at_start_seconds of 31299
# (8.7 h) for a full train_series pass -- which would consume the whole run before any model
# loads.  An empty set means no filtering, i.e. exactly the parent's behaviour.
_RSNA_OOF_ONLY = set()


def walk(split):"""

OLD_SCAN = ("    for study in os.scandir(base):\n"
            "        if study.is_dir():")
NEW_SCAN = ("    for study in os.scandir(base):\n"
            "        if _RSNA_OOF_ONLY and study.name not in _RSNA_OOF_ONLY:\n"
            "            continue\n"
            "        if study.is_dir():")

OLD_STUDIES = ("    test_df = pd.read_csv(ROOT / 'test.csv')\n"
               "    test_series = pd.read_csv(ROOT / 'test_series.csv')\n"
               "    plane_map = dict(zip(test_series['SeriesInstanceUID'], "
               "test_series['Anatomical_Plane']))\n"
               "    hte = annotate(walk('test_series'))\n")

NEW_STUDIES = """    # ---- OOF study selection: this kernel's entire reason to exist ----------
    _oof_fold = os.environ.get('RSNA_OOF_FOLD', '@FOLD@')
    _oof_limit = int(os.environ.get('RSNA_OOF_LIMIT', '@LIMIT@') or 0)
    # Kaggle has mounted datasets under at least three conventions:
    #   /kaggle/input/<slug>/...                     (flat, the classic one)
    #   /kaggle/input/datasets/<owner>/<slug>/...
    #   /kaggle/input/notebooks/<owner>/<slug>/...
    # Enumerate explicitly to depth 3 rather than using a recursive glob: `**/` over this tree walks
    # every study and series directory of a 570 GB competition (A60 measured 623 s for an unbounded
    # glob).  The first version of this finder covered only depth 2 and died with a FileNotFoundError
    # -- which is the good outcome, because it failed closed in a minute instead of silently running
    # the wrong cohort.  And when it does fail it must SAY WHAT IT SAW: a finder that reports "not
    # found" without printing the layout gets re-run blind.
    _fold_csv = None
    _seen = []
    _tried = []
    for _l1 in sorted(Path('/kaggle/input').iterdir()):
        if not _l1.is_dir():
            continue
        _seen.append(_l1.name)
        _cands = [_l1 / 'v52_oof.csv']
        for _l2 in sorted(_l1.iterdir()):
            if _l2.is_dir():
                _cands.append(_l2 / 'v52_oof.csv')
                for _l3 in sorted(_l2.iterdir()):
                    if _l3.is_dir():
                        _cands.append(_l3 / 'v52_oof.csv')
        for _c in _cands:
            _tried.append(str(_c))
            if _c.is_file():
                _fold_csv = _c
                break
        if _fold_csv is not None:
            break
    if _fold_csv is None:
        log('OOF: /kaggle/input holds: %s' % sorted(_seen))
        log('OOF: %d candidate paths checked, e.g. %s' % (len(_tried), _tried[:8]))
        raise FileNotFoundError('OOF run needs the published v52_oof.csv for the fold assignment')
    log('OOF: fold assignment from %s' % _fold_csv)
    _fold_df = pd.read_csv(_fold_csv, dtype={'StudyInstanceUID': str})
    _want = _fold_df[_fold_df['fold'].astype(str) == str(_oof_fold)].StudyInstanceUID.tolist()
    if _oof_limit:
        _want = _want[:_oof_limit]
    if not _want:
        raise RuntimeError('fold %r selected zero studies from %s' % (_oof_fold, _fold_csv))
    globals()['_RSNA_OOF_ONLY'] = set(_want)
    log('OOF: fold %s -> %d training studies (limit=%d)'
        % (_oof_fold, len(_want), _oof_limit))
    # ⚠️ THE COHORT GLOBAL MUST BE REPOINTED, and this is not optional.
    # `build_cache` takes its study list from `_RSNA_TEST_IDS` (L1553:
    #     studies = list(_RSNA_TEST_IDS) if _RSNA_TEST_IDS is not None else sorted(slot_map)
    # ), and that global was set much earlier by `rsna_initialize(test.csv...)` to the 3 PUBLIC TEST
    # studies.  Without this line the cache silently drops our training cohort: it notices the
    # mismatch, records a `coverage_mismatch` event, and substitutes empty slots
    # (`slot_map={s:{} for s in studies}`) -- so the run COMPLETES, the 20/20 fingerprint gate PASSES,
    # and every member's saved prediction is 3 all-zero rows for the test studies.  That is exactly
    # what the first two smoke runs produced; a silent-degradation failure, not a crash.
    rsna_initialize(_want)
    if set(_RSNA_TEST_IDS) != set(_want):
        raise RuntimeError('cohort global did not take the OOF study list')
    test_df = pd.DataFrame({'StudyInstanceUID': _want})
    _all_series = pd.read_csv(ROOT / 'train_series.csv',
                              dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    test_series = _all_series[_all_series['StudyInstanceUID'].isin(set(_want))].copy()
    if not len(test_series):
        raise RuntimeError('OOF run found no train_series rows for the selected studies')
    log('OOF: %d series over %d studies' % (len(test_series),
                                            test_series['StudyInstanceUID'].nunique()))
    plane_map = dict(zip(test_series['SeriesInstanceUID'], test_series['Anatomical_Plane']))
    hte = annotate(walk('train_series'))
"""

# ⚠️ No trailing newline: this line is the LAST line of its cell (the next cell starts with
# `rsna_phase('dinov2', 'START')`).  Requiring a "\n" here silently matched zero times.
OLD_MAIN_VALIDATE = ("    _v37_validate_submission('_pipeline_stage.csv', "
                     "pd.read_csv(ROOT / 'test.csv'), 'public frontier')")
NEW_MAIN_VALIDATE = """    # The parent validates the staged submission against test.csv.  Here the cohort is a fold of the
    # TRAINING set, so that check cannot hold; the meaningful one is that the stage covered exactly
    # the studies we asked for.
    # ⚠️ The cohort must come from `_RSNA_OOF_ONLY`, which `infer_from_package` set as a global.
    # `test_df` is a LOCAL of that function and is not visible here -- referencing it cost a whole
    # smoke run (NameError after 155 s of correct work).
    _want_ids = set(globals().get('_RSNA_OOF_ONLY') or ())
    if not _want_ids:
        # Refuse to pass vacuously: an empty cohort would make every comparison below trivially true,
        # which is the "gate that cannot see" failure this project has already paid for twice.
        raise RuntimeError('OOF coverage check has no cohort; refusing to pass vacuously')
    _staged = pd.read_csv(Path('/kaggle/working/_pipeline_stage.csv'),
                          dtype={'StudyInstanceUID': str})
    _miss = _want_ids - set(_staged['StudyInstanceUID'])
    if _miss:
        raise RuntimeError('OOF stage missed %d of the %d requested studies'
                           % (len(_miss), len(_want_ids)))
    log('OOF: E2 stage covered %d studies; per-member predictions are in '
        '/kaggle/working/diagnostics/' % len(_staged))
"""

EDITS = ((OLD_WALK, NEW_WALK, "walk() filter hook"),
         (OLD_SCAN, NEW_SCAN, "walk() study filter"),
         (OLD_STUDIES, NEW_STUDIES, "E2 study list -> training fold"),
         (OLD_MAIN_VALIDATE, NEW_MAIN_VALIDATE, "main() validation -> OOF coverage check"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0")
    ap.add_argument("--limit", default="0", help="0 = the whole fold; small value = smoke test")
    ap.add_argument("--dir", default=None, help="output dir (default kernel-e2oof)")
    args = ap.parse_args()

    dst = KERNELS / (args.dir or "kernel-e2oof")
    if not SRC_NB.is_file():
        raise SystemExit("!! missing %s" % SRC_NB)
    dst.mkdir(parents=True, exist_ok=True)

    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

    # Per-cell editing, the way build_0943_arms.py does it: join each cell's source, substitute,
    # and write it back as a single element.  Editing a global join and re-splitting would have to
    # re-derive the cell boundaries, and the edits change the line count.
    counts = {old: 0 for old, _, _ in EDITS}
    for c in nb["cells"]:
        txt = "".join(c.get("source", []))
        if not txt:
            continue
        orig = txt
        for old, new, _ in EDITS:
            counts[old] += txt.count(old)
            if old in txt:
                subst = new.replace("@FOLD@", args.fold).replace("@LIMIT@", args.limit)
                txt = txt.replace(old, subst)
        if txt != orig:
            c["source"] = [txt]

    for old, _, what in EDITS:
        if counts[old] != 1:
            raise SystemExit("!! %s: matched %d times (need exactly 1)" % (what, counts[old]))

    # Assert the walk substitution on the EDITED-BUT-NOT-YET-TRUNCATED notebook.  `walk('test_series')`
    # legitimately appears a second time in the A5/Rad stage, so the honest statement is "exactly one
    # occurrence was replaced"; after truncation that stage is gone and the count must be zero, which
    # is asserted separately below.  Checking it only after truncation silently compared against a
    # different text (and the first version of this assertion failed for exactly that reason).
    edited_joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    src_joined = "\n".join("".join(c.get("source", [])) for c in
                           json.loads(SRC_NB.read_text(encoding="utf-8"))["cells"])
    was = src_joined.count("walk('test_series')")
    now_edited = edited_joined.count("walk('test_series')")
    assert now_edited == was - 1, \
        "expected exactly one test_series walk to be replaced (%d -> %d)" % (was, now_edited)

    # ---- truncate: everything after the E2 stage is not ours to run -----------
    # The A5/Rad stage that follows is built around the TEST set -- it hard-requires
    # `test_series` ("train-series fallback is forbidden") and takes its cohort from the test split --
    # so in an OOF run on training studies it necessarily fails.  It contributes nothing to
    # `rank(E2)`, and its failure turned two otherwise-correct runs into ERRORs, which is exactly the
    # kind of misleading signal this project keeps paying for.  Cut it and everything after it.
    cut = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code" and "rsna_phase('a5', 'START')" in "".join(c.get("source", [])):
            cut = i
            break
    if cut is None:
        raise SystemExit("!! could not find the A5-stage boundary to truncate at")
    dropped_code = sum(1 for c in nb["cells"][cut:] if c.get("cell_type") == "code")
    nb["cells"] = nb["cells"][:cut]

    joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])

    # ---- assertions on the artefact we are about to push ---------------------
    want_fold = "os.environ.get('RSNA_OOF_FOLD', '%s')" % args.fold
    want_limit = "os.environ.get('RSNA_OOF_LIMIT', '%s')" % args.limit
    assert want_fold in joined, "fold default was not baked in"
    assert want_limit in joined, "limit default was not baked in"
    assert "walk('train_series')" in joined, "the E2 walk was not redirected to train_series"
    # after truncation the only remaining test_series walk must be gone as well
    assert "walk('test_series')" not in joined, "a test_series walk survived the truncation"
    assert "_RSNA_OOF_ONLY and study.name not in _RSNA_OOF_ONLY" in joined, "study filter missing"
    assert "and study.name not in _RSNA_OOF_ONLY:\n            continue" in joined, "filter body wrong"
    # The E2 stage's own gate must survive the truncation, and the A5 stage must be gone.
    assert "_DINOV2_MATCHED_MEMBERS" in joined, "the DINOv2 fingerprint gate was truncated away"
    assert "def main():" in joined, "main() was truncated away"
    assert "train-series fallback is forbidden" not in joined, "the A5 stage was NOT truncated away"
    assert "EXPECTED_DINOV2_MEMBERS = 20" in joined, "member count changed"
    # compile every code cell
    bad = 0
    for i, c in enumerate([c for c in nb["cells"] if c.get("cell_type") == "code"]):
        try:
            compile("".join(c.get("source", [])), "<cell%d>" % i, "exec")
        except SyntaxError as e:
            bad += 1
            print("  !! cell %d SyntaxError: %s" % (i, e))
    if bad:
        raise SystemExit("!! %d code cell(s) do not compile" % bad)

    (dst / NB_NAME).write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")

    m = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    slug = dst.name.replace("kernel-", "rsna-knee-")
    m["id"] = "ppxl16/%s" % slug
    m["title"] = "RSNA knee %s" % dst.name.replace("kernel-", "")
    m["is_private"] = True
    m["enable_internet"] = False          # FACTS A78: a code-comp submission is refused otherwise
    m["enable_gpu"] = True
    m["enable_tpu"] = False
    m.pop("id_no", None)
    (dst / "kernel-metadata.json").write_text(
        json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("built %s" % dst)
    print("  id       : %s" % m["id"])
    print("  fold     : %s   limit: %s" % (args.fold, args.limit))
    print("  datasets : %d   kernels: %d   internet: %s"
          % (len(m.get("dataset_sources") or []), len(m.get("kernel_sources") or []),
             m.get("enable_internet")))
    print("  cells    : %d total, %d code, all compile (dropped %d code cell(s) after the E2 stage)"
          % (len(nb["cells"]), sum(1 for c in nb["cells"] if c.get("cell_type") == "code"),
             dropped_code))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
