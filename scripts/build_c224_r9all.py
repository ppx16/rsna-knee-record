"""Build the r9all-leg kernel: our currently-credited best (r7 + c224 @0.5) plus bliverrigpdoge's
PUBLIC independent leg, rank-blended at the very end.

Why this artifact (FACTS C173/C174):
  * the live board moved `LastSubmissionDate` to 57163476 (`rr = 0.5*rank(r384) + 0.5*rank(c224)`),
    so an INDEPENDENT model leg rank-blended into the base is the one axis with measured upside;
  * `bliverrigpdoge/r7-raptor-knee-centred-convnext` reached 0.957 with exactly that shape (base
    rank-blended 0.8/0.2 with one extra model of his) -- but its weights are private, so a fork is
    worth 0.956 and nothing else;
  * his PUBLIC dataset `bliverrigpdoge/rsna-knee-leg-r9all` ships a complete, self-contained leg
    (`leg_main.py` + `loop_R9_all.pt` CoAtNet-RMLP-2-384, FineSpacing geometry, sha a7d63688...,
    standalone 0.942) with its own public blending rule `leg_weight_for_score(>=0.93) -> 0.45`.

The leg runs as its OWN process (separate CUDA context) after the base has written submission.csv,
and the blend is FAIL-SAFE: any failure, missing file, low coverage or column drift restores the
base file byte-for-byte.  So a failed experiment cannot cost us the 0.956 we already hold.

Usage: python work/build_c224_r9all.py
"""
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'kernels/kernel-r7-c224leg'
LEG_DATASET = 'bliverrigpdoge/rsna-knee-leg-r9all'

# the c224 leg weight we keep: 0.5 is the value the platform currently credits (C173)
OLD = '0.7 * np.nan_to_num(rr) + 0.3 * np.nan_to_num(_r2)'
NEW = '0.5 * np.nan_to_num(rr) + 0.5 * np.nan_to_num(_r2)'
OLD_CMT = '# [c224 leg] 0.7 r384 / 0.3 r224, rank space'
NEW_CMT = '# [c224 leg] 0.5 r384 / 0.5 r224, rank space'
OLD_PRINT = "[c224] fused at 0.3 over %d studies"
NEW_PRINT = "[c224] fused at 0.5 over %d studies"

# (suffix, LEG_W)
VARIANTS = [('w45', 0.45), ('w25', 0.25), ('w10', 0.10), ('w60', 0.60)]

MD = """## Stage 4: bliverrigpdoge's public r9all leg, rank-blended at the very end

`leg_main.py` from the attached public dataset `rsna-knee-leg-r9all` scores the test set with a
CoAtNet-RMLP-2-384 ("R9_all", FineSpacing geometry, standalone 0.942).  Its output is rank-blended
per finding into the submission the base already wrote:

    final = (1 - w) * rank01(base) + w * rank01(leg)

A study the leg did not score keeps its base rank, and ANY failure -- missing file, non-zero exit,
column drift, under 50 % coverage -- restores the base file unchanged."""

HELPERS = '''# Stage 4 helpers: per-finding rank blend of the base submission and the leg's own output.
# Public recipe from bliverrigpdoge's notebook (his `leg_rank01` / `leg_blend`), re-implemented here
# so this cell is self-contained.
_LEG_ID = 'StudyInstanceUID'


def _leg_rank01(values):
    """Average ranks scaled to [0, 1]; NaN stays NaN; a single value maps to 0.5."""
    s = pd.Series(np.asarray(values, dtype=np.float64))
    valid = s.notna()
    out = np.full(len(s), np.nan)
    n = int(valid.sum())
    if n == 0:
        return out
    if n == 1:
        out[valid.to_numpy()] = 0.5
        return out
    out[valid.to_numpy()] = ((s[valid].rank(method='average') - 1.0) / (n - 1.0)).to_numpy()
    return out


def _leg_blend(base, mine, w, labels):
    """Blend per finding; a study the leg did not score keeps its base rank. Returns (frame, n_with_leg)."""
    assert 0.0 <= w <= 1.0, 'leg weight outside [0, 1]'
    labels = list(labels)
    base = base[[_LEG_ID] + labels].copy()
    base[_LEG_ID] = base[_LEG_ID].astype(str)
    mine = mine[[_LEG_ID] + labels].copy()
    mine[_LEG_ID] = mine[_LEG_ID].astype(str)
    assert not mine[_LEG_ID].duplicated().any(), 'the leg wrote a study twice'
    unknown = set(mine[_LEG_ID]) - set(base[_LEG_ID])
    assert not unknown, 'the leg wrote %d studies outside the base' % len(unknown)
    mine = mine.set_index(_LEG_ID).reindex(base[_LEG_ID])
    out = base.copy()
    for name in labels:
        base_rank = _leg_rank01(base[name].to_numpy())
        leg_rank = _leg_rank01(mine[name].to_numpy())
        have = ~np.isnan(leg_rank)
        out[name] = np.where(have, (1.0 - w) * base_rank + w * leg_rank, base_rank)
    return out, int((~mine[labels].isna().all(axis=1)).sum())
'''

RUNNER = '''# Stage 4: run the public r9all leg as its own process and blend it in at LEG_W.
# The base submission is stashed first and EVERY failure path puts it back unchanged.
import gc as _gc
import json as _json
import os as _os
import shutil as _shutil
import signal as _signal
import subprocess as _subprocess
import sys as _sys
import time as _time

LEG_W = __LEG_W__                      # fixed before the run; 0.45 is the author's own rule for a leg scoring >= 0.93
LEG_DATASET = 'rsna-knee-leg-r9all'
LEG_START_MAX_S = 6.5 * 3600           # only start the leg if the base finished inside this
LEG_ABORT_S = 8.4 * 3600               # stop the whole stage at this elapsed time
LEG_MARGIN_S = 300.0                   # the child stops taking studies this long before the abort
LEG_MIN_COVERAGE = 0.5                 # a leg covering less of the test set than this is not blended
_leg_final = WORK / 'submission.csv'
_leg_stash = WORK / '_base_submission.csv'
_leg_report = {'status': 'not started', 'w': LEG_W, 'start_max_s': LEG_START_MAX_S, 'abort_s': LEG_ABORT_S}


def _leg_find_main():
    for base in (Path('/kaggle/input') / LEG_DATASET,
                 Path('/kaggle/input/datasets') / 'bliverrigpdoge' / LEG_DATASET):
        if (base / 'leg_main.py').is_file():
            return base / 'leg_main.py'
    for dirpath, dirnames, filenames in _os.walk('/kaggle/input'):
        if 'leg_main.py' in filenames:
            return Path(dirpath) / 'leg_main.py'
    return None


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
        _leg_main = _leg_find_main()
        assert _leg_main is not None, 'leg_main.py not found under /kaggle/input'
        _leg_report['leg_main'] = str(_leg_main)
        _leg_free_memory()
        _leg_out = WORK / '_r9all.csv'
        _leg_rep = WORK / '_r9all_report.json'
        for _p in (_leg_out, _leg_rep):
            if _p.exists():
                _p.unlink()
        _leg_remain = LEG_ABORT_S - (_time.time() - T0)
        assert _leg_remain > 60, 'no time left for the leg'
        _leg_env = dict(_os.environ, PYTHONPATH=str(_leg_main.parent) + _os.pathsep + _os.environ.get('PYTHONPATH', ''))
        _leg_deadline = T0 + LEG_ABORT_S - LEG_MARGIN_S
        _leg_t0 = _time.time()
        with (WORK / '_r9all_child.log').open('w') as _leg_log:
            _leg_child = _subprocess.Popen([_sys.executable, str(_leg_main), '--out', str(_leg_out),
                                            '--report', str(_leg_rep), '--deadline', repr(_leg_deadline)],
                                           cwd=str(WORK), env=_leg_env, stdout=_leg_log,
                                           stderr=_subprocess.STDOUT, start_new_session=True)
            try:
                _leg_rc = _leg_child.wait(timeout=_leg_remain)
            except BaseException:                                  # the whole group, pool workers too
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
        if _leg_rep.is_file():
            _leg_report['child'] = _json.loads(_leg_rep.read_text())
        _leg_base = pd.read_csv(_leg_stash, dtype={_LEG_ID: str})
        _leg_mine = pd.read_csv(_leg_out, dtype={_LEG_ID: str})
        assert list(_leg_base.columns) == [_LEG_ID] + list(labels), 'base file columns drift'
        assert list(_leg_mine.columns) == [_LEG_ID] + list(labels), 'leg columns drift: %s' % list(_leg_mine.columns)
        _leg_v = _leg_mine[list(labels)].to_numpy(np.float64)
        assert np.isfinite(_leg_v).all(), 'the leg wrote a non-finite value'
        assert _leg_v.min() >= -1e-9 and _leg_v.max() <= 1 + 1e-9, 'the leg wrote a value outside [0, 1]'
        _leg_cov = _leg_mine[_LEG_ID].nunique() / max(len(test_ids), 1)
        _leg_report['coverage'] = round(_leg_cov, 4)
        assert _leg_cov >= LEG_MIN_COVERAGE, 'the leg scored %d of %d studies' % (_leg_mine[_LEG_ID].nunique(), len(test_ids))
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
        _leg_report['studies_with_leg'] = _leg_n
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
        (WORK / 'r9all_leg_report.json').write_text(_json.dumps(_leg_report, indent=1, default=str))
    except Exception:
        pass
print('R9ALL_LEG ' + _json.dumps(_leg_report, default=str)[:1200], flush=True)
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
    if base.count(NEW) != 1 or base.count(NEW_CMT) != 1 or base.count(NEW_PRINT) != 1:
        raise SystemExit('REFUSING: the new c224 strings are not each unique')
    nb0 = json.loads(base)
    if len(nb0['cells']) != 9:
        raise SystemExit('REFUSING: expected 9 cells in the source notebook, found %d' % len(nb0['cells']))

    if LEG_DATASET in src_meta['dataset_sources']:
        raise SystemExit('REFUSING: the leg dataset is already attached to the source')
    for suffix, w in VARIANTS:
        nb = copy.deepcopy(nb0)
        for kind, body in (('markdown', MD), ('code', HELPERS), ('code', RUNNER.replace('__LEG_W__', repr(w)))):
            nb['cells'].append({'cell_type': kind, 'metadata': {}, 'source': body.splitlines(keepends=True),
                                **({'execution_count': None, 'outputs': []} if kind == 'code' else {})})
        slug = 'rsna-knee-r7-c224-r9all-' + suffix
        title = 'RSNA Knee r7 c224 r9all ' + suffix
        if slugify(title) != slug:
            raise SystemExit('REFUSING[%s]: title %r slugifies to %r' % (suffix, title, slugify(title)))
        meta = copy.deepcopy(src_meta)
        meta.pop('id_no', None)
        meta['id'] = 'ppxl16/' + slug
        meta['title'] = title
        meta['code_file'] = slug + '.ipynb'
        meta['is_private'] = True
        meta['dataset_sources'] = list(src_meta['dataset_sources']) + [LEG_DATASET]
        for k, v in src_meta.items():
            if k in ('id', 'title', 'code_file', 'is_private', 'id_no', 'dataset_sources'):
                continue
            if json.dumps(meta.get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
                raise SystemExit('REFUSING[%s]: environment field %s changed' % (suffix, k))
        out = json.dumps(nb, ensure_ascii=False)
        try:
            out.encode('utf-8').decode('gbk')
        except UnicodeDecodeError as exc:
            raise SystemExit('REFUSING[%s]: not GBK-decodable: %s' % (suffix, exc))
        dst = ROOT / 'kernels' / ('kernel-r7-c224-r9all-' + suffix)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / meta['code_file']).write_text(out, encoding='utf-8')
        (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print('%s: LEG_W=%s  cells %d->%d  %s  (%d B, sha %s)'
              % (suffix, w, len(nb0['cells']), len(nb['cells']), dst.name, len(out),
                 hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('audit: c224 leg at 0.5 (the value the platform credits) + 3 appended cells; env fields untouched')
    print('       fail-safe: any leg failure restores the base submission.csv byte-for-byte')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
