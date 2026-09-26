"""Build the A5-OOF kernel: run the A5 stage over a TRAINING fold and keep its per-fold predictions.

WHY THE A5 ARM AND NOT THE DINOv2 PARENT
----------------------------------------
The parent approach was measured invalid (FACTS A93): the twenty published DINOv2 members are not
held out -- on the same studies the parent scored 0.9295 macro against a genuine OOF's 0.8323, a gap
the frontier's own gold-58 numbers say cannot be real.  Their `fold` field does not demarcate a
held-out split either (the fold-0 and non-fold-0 members score identically on fold-0 studies).

The A5 arm is different: `mattiaangeli/knee-mri-fold-weights` ships **`m_f0.pt .. m_f4.pt`**, one
genuine per-fold checkpoint each, and the stage's own loader asserts the file/fold identity
(`ckpt_path.name != f'm_f{fold}.pt'` -> raise).  So `raw[k]` -- the prediction of `m_f{k}.pt` -- is an
honest out-of-fold prediction for fold-k studies.  A5 also carries one of the two weights the
frontier moved off the 0.941 lineage (`a5_w` 0.45 -> 0.52, alongside `rad_alpha` 0.50 -> 0.55), so a
genuine A5 OOF is exactly what re-litigating those knobs needs.

WHAT NEEDS CHANGING: FOUR EDITS AND A TRUNCATION
------------------------------------------------
1. **Skip the DINOv2/E2 stage.**  It contributes nothing here and costs ~45 min at cohort scale.
   The skip is explicit and logged, and the parent's post-stage gate is made conditional on the
   skip flag rather than being fed a fake pass -- a gate that is satisfied by a lie is how this
   project has already lost time twice (A79, A87).
2. **`SERIES_ROOT` -> `train_series`.**  The parent hard-forbids that fallback because its cohort is
   the test split; here the split is deliberately reversed.
3. **The cohort** comes from the published `v52_oof.csv` fold assignment, restricted to one fold,
   then `rsna_initialize(...)` to repoint the cohort global (without it `_RSNA_TEST_IDS` still holds
   test.csv's three studies, which is exactly the A87 failure).
4. **Truncate after the A5 stage** so the Rad/Raptor stages -- which are built around `test_series`
   and the test cohort -- never run and cannot turn a good run into a misleading ERROR.

**No plumbing is added for the predictions**: the stage already writes
`/kaggle/working/speed_a5_raw.npz` with `study_uids`, `raw_probabilities` of shape
[fold, study, finding], and `applicable`.  Fold discipline is therefore applied LOCALLY, where it can
be inspected.

Usage:
    python scripts/build_a5_oof_kernel.py --fold 0 --limit 48 --dir kernel-a5oof-smoke
    python scripts/build_a5_oof_kernel.py --fold 0
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "kernel-0943"
SRC_NB = SRC / "rsna-knee-restructured-version-3.ipynb"
NB_NAME = "rsna-knee-restructured-version-3.ipynb"

OLD_E2_CALL = "rsna_phase('dinov2', 'START')\nmain()\nlog('done')"
NEW_E2_CALL = """rsna_phase('dinov2', 'START')
# ---- SKIPPED BY DESIGN: this kernel runs ONLY the A5 stage --------------------------------
# The DINOv2 parent stage is not needed for the A5 out-of-fold predictions, and at cohort scale it
# costs roughly 45 minutes.  `_E2_SKIPPED` makes the skip explicit so the parent's post-stage gate
# can decline to fire on its own terms rather than being handed a fabricated pass.
globals()['_E2_SKIPPED'] = True
log('OOF: DINOv2/E2 stage SKIPPED by design (this is an A5-only kernel)')"""

OLD_DINO_GATE = "if globals().get('_DINOV2_MATCHED_MEMBERS') != 20:"
NEW_DINO_GATE = ("if not globals().get('_E2_SKIPPED') and "
                 "globals().get('_DINOV2_MATCHED_MEMBERS') != 20:")

OLD_SERIES_ROOT = """SERIES_ROOT = COMP / 'test_series'
if not SERIES_ROOT.is_dir():
    raise FileNotFoundError('Missing test_series; train-series fallback is forbidden')"""
NEW_SERIES_ROOT = """# OOF run: the cohort is a TRAINING fold, so the series live under train_series.  The parent forbids
# that fallback because ITS cohort is the test split -- here the split is deliberately reversed, and
# the guard is rewritten rather than deleted so the intent stays visible.
SERIES_ROOT = COMP / 'train_series'
if not SERIES_ROOT.is_dir():
    raise FileNotFoundError('Missing train_series; this A5-OOF kernel needs the training split')"""

OLD_A5_COHORT = """sub_df = pd.read_csv(COMP / 'sample_submission.csv',dtype={'StudyInstanceUID':str})
ser_csv = pd.read_csv(COMP / 'test_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
if set(sub_df.StudyInstanceUID)!=set(_RSNA_TEST_IDS):
    raise RuntimeError('A5 test/sample UID mismatch')
ser_csv = ser_csv.loc[:, ~ser_csv.columns.duplicated()]
studies = sub_df.StudyInstanceUID.tolist()
by = {s: g.to_dict('records') for s, g in ser_csv[ser_csv.StudyInstanceUID.isin(set(studies))].groupby('StudyInstanceUID')}
print(f'{len(studies):,} test studies, {len(by):,} with series metadata')"""

NEW_A5_COHORT = """# ---- A5 OOF cohort: the requested fold of the TRAINING set ------------------------------
_oof_fold = os.environ.get('RSNA_OOF_FOLD', '@FOLD@')
_oof_limit = int(os.environ.get('RSNA_OOF_LIMIT', '@LIMIT@') or 0)
# Kaggle mounts datasets under at least three conventions (/kaggle/input/<slug>/,
# /kaggle/input/datasets/<owner>/<slug>/, /kaggle/input/notebooks/<owner>/<slug>/).  Enumerate to
# depth 3 explicitly instead of globbing recursively: `**` would walk every series directory of a
# 570 GB competition (A60 measured 623 s for an unbounded glob).  And on failure, say what was seen.
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
    raise FileNotFoundError('A5 OOF needs the published v52_oof.csv for the fold assignment')
_fold_df = pd.read_csv(_fold_csv, dtype={'StudyInstanceUID': str})
# `--fold all` takes every study in the fold assignment.  This is the cheap way to the WHOLE OOF:
# the stage runs all five fold models over the cohort anyway and saves [fold, study, finding], so one
# pass over the training set yields every fold's held-out predictions, to be selected by fold
# LOCALLY.  Measured cost from the smoke run: 0.875 s/study, i.e. ~64 min for 4407.
if str(_oof_fold).lower() in ('all', '*'):
    studies = _fold_df.StudyInstanceUID.tolist()
else:
    studies = _fold_df[_fold_df['fold'].astype(str) == str(_oof_fold)].StudyInstanceUID.tolist()
if _oof_limit:
    studies = studies[:_oof_limit]
if not studies:
    raise RuntimeError('fold %r selected zero studies from %s' % (_oof_fold, _fold_csv))
# Repoint the cohort global BEFORE anything reads it.  `_RSNA_TEST_IDS` still holds test.csv's three
# studies from the parent's `rsna_initialize`, and A5 takes its cohort from it (the same trap that
# produced a completely wrong, completely passing run at FACTS A87).
rsna_initialize(studies)
if set(_RSNA_TEST_IDS) != set(studies):
    raise RuntimeError('A5 cohort global did not take the OOF study list')
sub_df = pd.DataFrame({'StudyInstanceUID': studies})
ser_csv = pd.read_csv(COMP / 'train_series.csv',
                      dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
ser_csv = ser_csv.loc[:, ~ser_csv.columns.duplicated()]
by = {s: g.to_dict('records') for s, g in
      ser_csv[ser_csv.StudyInstanceUID.isin(set(studies))].groupby('StudyInstanceUID')}
log('OOF: A5 cohort = fold %s, %d studies, %d with series metadata (via %s)'
    % (_oof_fold, len(studies), len(by), _fold_csv.name))"""

# (old, new, what, expected_count).  The DINOv2 gate legitimately appears TWICE -- once right after
# the E2 stage and once after the Rad stage -- and the second is removed by the truncation below, but
# counting happens before truncation, so its expectation is 2 rather than 1.
EDITS = ((OLD_E2_CALL, NEW_E2_CALL, "skip the E2 stage", 1),
         (OLD_DINO_GATE, NEW_DINO_GATE, "make the DINOv2 gate conditional on the skip", 2),
         (OLD_SERIES_ROOT, NEW_SERIES_ROOT, "series root -> train_series", 1),
         (OLD_A5_COHORT, NEW_A5_COHORT, "A5 cohort -> training fold", 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0")
    ap.add_argument("--limit", default="0", help="0 = the whole fold; a small value = smoke test")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()

    dst = ROOT / (args.dir or "kernel-a5oof")
    if not SRC_NB.is_file():
        raise SystemExit("!! missing %s" % SRC_NB)
    dst.mkdir(parents=True, exist_ok=True)

    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    counts = {old: 0 for old, _, _, _ in EDITS}
    for c in nb["cells"]:
        txt = "".join(c.get("source", []))
        if not txt:
            continue
        orig = txt
        for old, new, _, _ in EDITS:
            counts[old] += txt.count(old)
            if old in txt:
                txt = txt.replace(old, new.replace("@FOLD@", args.fold).replace("@LIMIT@", args.limit))
        if txt != orig:
            c["source"] = [txt]
    for old, _, what, want in EDITS:
        if counts[old] != want:
            raise SystemExit("!! %s: matched %d times (need exactly %d)" % (what, counts[old], want))

    # ---- truncate after the A5 stage ----------------------------------------
    cut = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code" and "rsna_phase('a5', 'COMPLETE'" in "".join(c.get("source", [])):
            cut = i + 1
            break
    if cut is None:
        raise SystemExit("!! could not find the A5 COMPLETE boundary")
    dropped = sum(1 for c in nb["cells"][cut:] if c.get("cell_type") == "code")
    nb["cells"] = nb["cells"][:cut]

    joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert "OOF: A5 cohort = fold" in joined, "A5 cohort rewrite missing"
    assert "rsna_initialize(studies)" in joined, "cohort global repoint missing"
    assert "SERIES_ROOT = COMP / 'train_series'" in joined, "series root not redirected"
    assert "globals()['_E2_SKIPPED'] = True" in joined, "E2 skip missing"
    # `def main():` contains "main()" as a substring, so match the BARE CALL form only.
    assert "\nmain()\n" not in joined, "the E2 pipeline is still being called"
    assert "speed_a5_raw.npz" in joined, "the raw per-fold save disappeared"
    assert "rsna_phase('a5', 'COMPLETE'" in joined, "the A5 stage was truncated away"
    # `test_series` legitimately appears in the EDA and ROOT-discovery cells, so assert on the two
    # places that actually decide the A5 cohort rather than on the bare string.
    assert "SERIES_ROOT = COMP / 'test_series'" not in joined, "series root not redirected"
    assert "ser_csv = pd.read_csv(COMP / 'train_series.csv'" in joined, "A5 series csv not redirected"
    assert "m_f{i}.pt" in joined, "per-fold checkpoint loading changed"

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
    m["enable_internet"] = False
    m["enable_gpu"] = True
    m["enable_tpu"] = False
    m.pop("id_no", None)
    (dst / "kernel-metadata.json").write_text(
        json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("built %s" % dst)
    print("  id      : %s" % m["id"])
    print("  fold    : %s   limit: %s" % (args.fold, args.limit))
    print("  cells   : %d total, %d code, all compile (dropped %d code cell(s) after A5)"
          % (len(nb["cells"]), sum(1 for c in nb["cells"] if c.get("cell_type") == "code"), dropped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
