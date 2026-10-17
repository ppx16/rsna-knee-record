"""Build the band32-leg kernel: our credited best (r7 + c224 @0.5) plus ricecakar's PUBLIC band32 leg.

Why a SECOND independent leg, and why this one (FACTS C173/C174/C175):
  * rank-blending an INDEPENDENT model into the base is the only axis with measured upside (c224
    @0.5 was credited over @0.3; bliverrigpdoge took 0.956 -> 0.957 by adding one extra model);
  * the r9all leg is a CoAtNet, i.e. the SAME family as the Raptor arm already in the base, so its
    diversity is limited.  `ricecakar/rsna-knee-band32-ours` is a **DINOv2 ViT + per-finding slot
    attention** over 8 seeds -- a different architecture family entirely, which is exactly the
    property that makes an added leg buy something;
  * its inference code is carried verbatim in the public notebook
    `ricecakar/rsna-knee-public-0-941-band32-adjacency-leg` (cell 52, 33,577 chars), is fully
    self-contained (writes its own CSV from `RSNA_OUT`), and declares its own member manifest.

Single-variable discipline: this kernel differs from our credited 0.956 recipe by (a) the c224 leg
fixed at 0.5 (the value the platform credits) and (b) ONE appended leg at ONE weight.  The blend is
FAIL-SAFE: a failure, a schema drift, a mostly-0.5 leg or a timeout restores the base file unchanged.

⚠️ The leg fills studies it could not score with a constant 0.5, so coverage is measured STRUCTURALLY
(a study counts only if its 12 values are not all exactly 0.5) and uncovered studies are set to NaN
so they keep the base rank -- a naive "all rows present -> fine" check would blend 0.5 into them.

Usage: python work/build_c224_band32.py
"""
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'kernels/kernel-r7-c224leg'
LEG_SRC = ROOT / 'scratch/band32/_band32_leg.py'
LEG_DATASET = 'ricecakar/rsna-knee-band32-ours'
LEG_MODEL = 'metaresearch/dinov2/PyTorch/small/1'

OLD = '0.7 * np.nan_to_num(rr) + 0.3 * np.nan_to_num(_r2)'
NEW = '0.5 * np.nan_to_num(rr) + 0.5 * np.nan_to_num(_r2)'
OLD_CMT = '# [c224 leg] 0.7 r384 / 0.3 r224, rank space'
NEW_CMT = '# [c224 leg] 0.5 r384 / 0.5 r224, rank space'
OLD_PRINT = "[c224] fused at 0.3 over %d studies"
NEW_PRINT = "[c224] fused at 0.5 over %d studies"

# (suffix, LEG_W)
VARIANTS = [('w15', 0.15), ('w30', 0.30), ('w08', 0.08)]

MD = """## Stage 4: ricecakar's public band32 leg, rank-blended at the very end

`_band32_leg.py` (carried verbatim from the public notebook
`ricecakar/rsna-knee-public-0-941-band32-adjacency-leg`, cell 52) scores the test set with a
fine-tuned DINOv2 ViT plus per-finding slot attention over 8 seeds from the attached public dataset
`ricecakar/rsna-knee-band32-ours`.  It is a different architecture family from the CoAtNet arms the
base is built on, which is the property that makes an added leg worth anything.

Its output is rank-blended per finding into the submission the base already wrote:

    final = (1 - w) * rank01(base) + w * rank01(leg)

The leg fills studies it could not score with a constant 0.5; those studies are detected
structurally and keep their base rank.  Any failure -- missing checkpoint, non-zero exit, schema
drift, under 50 % real coverage -- restores the base file unchanged."""

RUNNER = '''# Stage 4: run the public band32 leg as its own process and blend it in at LEG_W.
# The base submission is stashed first and EVERY failure path puts it back unchanged.
import gc as _gc
import json as _json
import os as _os
import shutil as _shutil
import signal as _signal
import subprocess as _subprocess
import sys as _sys
import time as _time

LEG_W = __LEG_W__                      # fixed before the run
LEG_SCRIPT = Path('/kaggle/working/_band32_leg.py')
LEG_START_MAX_S = 6.5 * 3600           # only start the leg if the base finished inside this
LEG_ABORT_S = 8.4 * 3600               # stop the whole stage at this elapsed time
LEG_MIN_COVERAGE = 0.5                 # share of studies that must carry a REAL prediction
_leg_final = WORK / 'submission.csv'
_leg_stash = WORK / '_base_submission.csv'
_leg_out = WORK / '_band32.csv'
_leg_report = {'status': 'not started', 'w': LEG_W, 'start_max_s': LEG_START_MAX_S, 'abort_s': LEG_ABORT_S}


def _leg_free_memory():
    _gc.collect()
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            for _i in range(_torch.cuda.device_count()):
                with _torch.cuda.device(_i):
                    _torch.cuda.empty_cache()
    except Exception:
        pass


try:
    assert _leg_final.is_file(), 'the base stage wrote no submission.csv'
    _shutil.copyfile(_leg_final, _leg_stash)                      # stash before anything else
    _leg_elapsed = _time.time() - T0
    _leg_report['elapsed_at_start_s'] = round(_leg_elapsed, 1)
    if LEG_W <= 0:
        _leg_report['status'] = 'skipped: w is 0'
    elif _leg_elapsed >= LEG_START_MAX_S:
        _leg_report['status'] = 'skipped: elapsed %.0f s >= %.0f s' % (_leg_elapsed, LEG_START_MAX_S)
    else:
        assert LEG_SCRIPT.is_file(), 'the leg script was not written to /kaggle/working'
        _leg_free_memory()
        for _p in (_leg_out,):
            if _p.exists():
                _p.unlink()
        _leg_remain = LEG_ABORT_S - (_time.time() - T0)
        assert _leg_remain > 60, 'no time left for the leg'
        # ⚠️ The leg REFUSES to run unless it is told how many members the weights dataset declares:
        # its own guard printed "[X] expected 3 models, found 10" and exited 1 (it defaults to 3).
        # The count is DECLARED by the dataset's own members.json, never guessed (its author's rule).
        # ⚠️⚠️ With TWO leg datasets attached, "the first members.json found" is the WRONG one -- in
        # the three-way stack it picked the r9all manifest (1 member) and the band32 leg refused.
        # So pick the manifest whose own directory tree actually holds the band32 checkpoints.
        import glob as _glob
        _cands = []
        for _d in range(1, 6):
            _cands += sorted(_glob.glob('/kaggle/input/' + '*/' * _d + 'members.json'))
        _n_exp, _n_ck_best, _src = None, -1, None
        for _m in _cands:
            _root = _os.path.dirname(_m)
            _n_ck = len(_glob.glob(_os.path.join(_root, '**', 'model_v3.pt'), recursive=True))
            if _n_ck <= _n_ck_best:
                continue
            try:
                _n_mem = len(_json.loads(open(_m, encoding='utf-8').read())['members'])
            except Exception:
                continue
            _n_exp, _n_ck_best, _src = _n_mem, _n_ck, _m
        _leg_report['members_json'] = _src
        _leg_report['model_v3_files'] = _n_ck_best
        assert _n_exp, 'could not find the band32 members.json under /kaggle/input'
        _leg_report['n_expected'] = _n_exp
        _leg_env = dict(_os.environ, RSNA_OUT=str(_leg_out), RSNA_SPLIT='test',
                        RSNA_N_EXPECTED=str(_n_exp), PYTHONDONTWRITEBYTECODE='1')
        _leg_t0 = _time.time()
        with (WORK / '_band32_child.log').open('w') as _leg_log:
            _leg_child = _subprocess.Popen([_sys.executable, '-u', str(LEG_SCRIPT)], cwd=str(WORK),
                                           env=_leg_env, stdout=_leg_log, stderr=_subprocess.STDOUT,
                                           start_new_session=True)
            try:
                _leg_rc = _leg_child.wait(timeout=_leg_remain)
            except BaseException:                                  # the whole group
                for _sig in (_signal.SIGTERM, getattr(_signal, 'SIGKILL', _signal.SIGTERM)):
                    try:
                        _os.killpg(_leg_child.pid, _sig)
                    except Exception:
                        try:
                            _leg_child.kill()
                        except Exception:
                            pass
                    try:
                        _leg_child.wait(timeout=20)
                        break
                    except Exception:
                        continue
                raise
        _leg_report['leg_seconds'] = round(_time.time() - _leg_t0, 1)
        _leg_report['child_returncode'] = _leg_rc
        assert _leg_rc == 0, 'the leg exited with code %d' % _leg_rc
        _leg_base = pd.read_csv(_leg_stash, dtype={_LEG_ID: str})
        _leg_mine = pd.read_csv(_leg_out, dtype={_LEG_ID: str})
        assert list(_leg_base.columns) == [_LEG_ID] + list(labels), 'base file columns drift'
        assert list(_leg_mine.columns) == [_LEG_ID] + list(labels), 'leg columns drift: %s' % list(_leg_mine.columns)
        _leg_v = _leg_mine[list(labels)].to_numpy(np.float64)
        assert np.isfinite(_leg_v).all(), 'the leg wrote a non-finite value'
        assert _leg_v.min() >= -1e-9 and _leg_v.max() <= 1 + 1e-9, 'the leg wrote a value outside [0, 1]'
        # STRUCTURAL coverage: the leg fills studies it could not score with a constant 0.5
        _leg_real = ~np.all(np.abs(_leg_v - 0.5) < 1e-12, axis=1)
        _leg_cov = float(_leg_real.mean())
        _leg_report['coverage'] = round(_leg_cov, 4)
        _leg_report['studies_with_leg'] = int(_leg_real.sum())
        assert _leg_cov >= LEG_MIN_COVERAGE, 'the leg really scored only %d of %d studies' % (int(_leg_real.sum()), len(_leg_mine))
        assert float(_leg_v[_leg_real].std(axis=0).min()) > 1e-9, 'the leg has a constant column'
        _leg_mine.loc[~_leg_real, list(labels)] = np.nan          # unscored studies keep the base rank
        _leg_blended, _leg_n = _leg_blend(_leg_base, _leg_mine, LEG_W, labels)
        assert list(_leg_blended.columns) == [_LEG_ID] + list(labels), 'blended columns drift'
        assert _leg_blended[_LEG_ID].astype(str).tolist() == [str(u) for u in test_ids], 'blended ids/order drift'
        _leg_bv = _leg_blended[list(labels)].to_numpy(np.float64)
        assert np.isfinite(_leg_bv).all() and _leg_bv.min() >= -1e-9 and _leg_bv.max() <= 1 + 1e-9
        _leg_tmp = _leg_final.with_suffix('.csv.tmp')
        with _leg_tmp.open('w') as _leg_h:
            _leg_blended.to_csv(_leg_h, index=False)
            _leg_h.flush()
            _os.fsync(_leg_h.fileno())
        _os.replace(_leg_tmp, _leg_final)
        _leg_report['status'] = 'blended'
except BaseException as _leg_exc:                                   # publish the stash unchanged
    _leg_report['status'] = 'fallback to the base file: %s: %s' % (type(_leg_exc).__name__, str(_leg_exc)[:300])
    try:
        if _leg_stash.is_file():
            _shutil.copyfile(_leg_stash, _leg_final)
    except Exception as _leg_restore:
        _leg_report['restore_error'] = repr(_leg_restore)
finally:
    _leg_report['total_stage_seconds'] = round(_time.time() - T0, 1)
    try:
        (WORK / 'band32_leg_report.json').write_text(_json.dumps(_leg_report, indent=1, default=str))
    except Exception:
        pass
print('BAND32_LEG ' + _json.dumps(_leg_report, default=str)[:1200], flush=True)
assert pd.read_csv(_leg_final, nrows=0).columns.tolist() == [_LEG_ID] + list(labels), 'final header drift'
'''


def slugify(title: str) -> str:
    return re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', title.lower())).strip('-')


def main() -> int:
    src_nb = next(SRC.glob('*.ipynb'))
    src_meta = json.loads((SRC / 'kernel-metadata.json').read_text(encoding='utf-8'))
    raw = src_nb.read_bytes().decode('utf-8')
    for needle, what in ((OLD, 'fusion expression'), (OLD_CMT, 'fusion comment'), (OLD_PRINT, 'fusion print')):
        if raw.count(needle) != 1:
            raise SystemExit('REFUSING: the %s matched %d times' % (what, raw.count(needle)))
    base = raw.replace(OLD, NEW).replace(OLD_CMT, NEW_CMT).replace(OLD_PRINT, NEW_PRINT)
    for needle in (OLD, OLD_CMT, OLD_PRINT):
        if needle in base:
            raise SystemExit('REFUSING: %r survived the c224 weight change' % needle)
    nb0 = json.loads(base)
    if len(nb0['cells']) != 9:
        raise SystemExit('REFUSING: expected 9 cells in the source notebook, found %d' % len(nb0['cells']))

    leg = LEG_SRC.read_text(encoding='utf-8')
    if 'RSNA_OUT' not in leg or 'LABEL_COLUMNS' not in leg:
        raise SystemExit('REFUSING: the leg source does not look like the band32 leg')
    compile(leg, 'band32_leg', 'exec')                       # it must be valid Python on its own

    # helpers cell is shared with the r9all builder (identical semantics)
    helpers = (ROOT / 'kernels' / 'kernel-r7-c224-r9all-w45').glob('*.ipynb')
    hn = json.loads(next(helpers).read_bytes().decode('utf-8'))
    helpers_src = ''.join(hn['cells'][10]['source'])
    if '_leg_blend' not in helpers_src or '_leg_rank01' not in helpers_src:
        raise SystemExit('REFUSING: could not recover the leg helper cell')

    if LEG_DATASET in src_meta['dataset_sources']:
        raise SystemExit('REFUSING: the leg dataset is already attached to the source')
    for suffix, w in VARIANTS:
        nb = copy.deepcopy(nb0)
        for kind, body in (('markdown', MD),
                           ('code', '%%writefile /kaggle/working/_band32_leg.py\n' + leg),
                           ('code', helpers_src),
                           ('code', RUNNER.replace('__LEG_W__', repr(w)))):
            nb['cells'].append({'cell_type': kind, 'metadata': {}, 'source': body.splitlines(keepends=True),
                                **({'execution_count': None, 'outputs': []} if kind == 'code' else {})})
        slug = 'rsna-knee-r7-c224-band32-' + suffix
        title = 'RSNA Knee r7 c224 band32 ' + suffix
        if slugify(title) != slug:
            raise SystemExit('REFUSING[%s]: title %r slugifies to %r' % (suffix, title, slugify(title)))
        meta = copy.deepcopy(src_meta)
        meta.pop('id_no', None)
        meta['id'] = 'ppxl16/' + slug
        meta['title'] = title
        meta['code_file'] = slug + '.ipynb'
        meta['is_private'] = True
        meta['dataset_sources'] = list(src_meta['dataset_sources']) + [LEG_DATASET]
        meta['model_sources'] = list(src_meta.get('model_sources') or []) + [LEG_MODEL]
        for k, v in src_meta.items():
            if k in ('id', 'title', 'code_file', 'is_private', 'id_no', 'dataset_sources', 'model_sources'):
                continue
            if json.dumps(meta.get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
                raise SystemExit('REFUSING[%s]: environment field %s changed' % (suffix, k))
        # ensure_ascii=True: the leg source contains characters the Windows pusher cannot encode in
        # GBK (A-note: a U+8DEF character already cost us one kernel).  Escaping every non-ASCII
        # character keeps the uploaded bytes pure ASCII while the JSON payload -- i.e. what Jupyter
        # executes -- is IDENTICAL, which is asserted below rather than assumed.
        out = json.dumps(nb, ensure_ascii=True)
        if json.loads(out) != nb:
            raise SystemExit('REFUSING[%s]: the ASCII-escaped notebook does not round-trip' % suffix)
        if not out.isascii():
            raise SystemExit('REFUSING[%s]: the notebook is still not pure ASCII' % suffix)
        try:
            out.encode('utf-8').decode('gbk')
        except UnicodeDecodeError as exc:
            raise SystemExit('REFUSING[%s]: not GBK-decodable: %s' % (suffix, exc))
        dst = ROOT / 'kernels' / ('kernel-r7-c224-band32-' + suffix)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / meta['code_file']).write_text(out, encoding='utf-8')
        (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print('%s: LEG_W=%s  cells %d->%d  %s  (%d B, sha %s)'
              % (suffix, w, len(nb0['cells']), len(nb['cells']), dst.name, len(out),
                 hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('audit: c224 leg at 0.5 + band32 leg verbatim + helpers + runner; env fields untouched')
    print('       fail-safe: any leg failure restores the base submission.csv byte-for-byte')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
