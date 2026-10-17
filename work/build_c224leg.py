"""Transplant rabari9999's standalone infer224.py into our 0.956 recipe as a SEPARATE process (FACTS C167).

Why a separate process and not a second ARMS entry: heliosli's r7 fork specialised the upstream
multi-resolution loader into a single-resolution one -- `assert observed_sha == '7e5315da...'`,
`assert res == 384`, `INPUT_RES = 320`, and `assert res == INPUT_RES` inside the windowing.  rabari's
infer224.py keeps the UPSTREAM generic loader, so transplanting his file is the minimal correct move.

FAIL-SAFE BY DESIGN: the r224 leg is optional at fusion time.  If the subprocess fails, or the CSV is
absent, or the row set does not match, the stage-3 blend falls back to exactly today's 0.956 path.
A failed experiment must not cost us the submission we already hold.

Audit: every ORIGINAL cell must be unchanged except the two we name; the new cells are appended.
"""
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'kernels/kernel-r7-onclick'
RABARI = ROOT / 'scratch/rabari/rsna-knee-sota-dual-engine-stack-lb-0-951.ipynb'
DST = ROOT / 'kernels/kernel-r7-c224leg'
NEW_ID = 'ppxl16/rsna-knee-r7-c224leg'
NEW_TITLE = 'RSNA Knee r7 c224leg'
NEW_CODE = 'rsna-knee-r7-c224leg.ipynb'

MARK = '%%writefile /kaggle/working/c224_src/infer224.py'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    src = json.loads(next(SRC.glob('*.ipynb')).read_text(encoding='utf-8'))
    rab = json.loads(RABARI.read_text(encoding='utf-8'))

    # ---- 1. lift rabari's infer224.py verbatim ----
    payload = None
    for c in rab['cells']:
        s = ''.join(c.get('source', []))
        if s.startswith(MARK) and len(s) > 50000:
            payload = s[len(MARK):].lstrip('\n')
            break
    if payload is None:
        raise SystemExit('REFUSING: could not find the infer224.py cell in rabari\'s notebook')
    print('lifted infer224.py: %d chars' % len(payload))
    if 'def load_model(pt_path' not in payload:
        raise SystemExit('REFUSING: the lifted payload has no load_model')
    if "== 384" in payload and 'Expected original 384' in payload:
        raise SystemExit('REFUSING: the lifted payload still carries the single-res assertion')

    # ---- 2. our own writefile cell (bash-free: write it from Python in the runner instead) ----
    # The notebook is executed by papermill; a %%writefile cell is fine, so keep rabari's mechanism.
    wf = '%%writefile /tmp/raptor224_run.py\n' + payload

    # ---- 3. runner cell, non-fatal ----
    runner = '''# Stage 1b (ours): the r224 CoAtNet as its OWN process, so heliosli's single-res assertions
# in the r384 stage are untouched (FACTS C167).  NON-FATAL: a failure here must not cost us the
# submission we already hold -- stage 3 reads _c224.csv only if it is valid.
import os as _o, sys as _s, subprocess as _sp, time as _t
_C224_OK = False
_t0 = _t.time()
try:
    _rc = _sp.run([_s.executable, '-u', '/tmp/raptor224_run.py'], timeout=2 * 3600).returncode
    _f = '/kaggle/working/_c224.csv'
    _C224_OK = (_rc == 0) and _o.path.isfile(_f) and _o.path.getsize(_f) > 1000
    print('[c224] rc=%s ok=%s %.0fs' % (_rc, _C224_OK, _t.time() - _t0), flush=True)
except Exception as _e:
    print('[c224] FAILED (non-fatal): %r' % (_e,), flush=True)
'''

    # ---- 4. patch stage 3 to fuse the c224 leg at 0.3, fail-safe ----
    blend_idx = None
    for i, c in enumerate(src['cells']):
        if c.get('cell_type') == 'code' and 'W_RAPTOR, W_OURS' in ''.join(c.get('source', [])):
            blend_idx = i
            break
    if blend_idx is None:
        raise SystemExit('REFUSING: could not find the stage-3 blend cell')

    old3 = ''.join(src['cells'][blend_idx]['source'])
    anchor = "rsub = rsub.set_index('StudyInstanceUID').loc[test_ids, labels]\n"
    if old3.count(anchor) != 1:
        raise SystemExit('REFUSING: stage-3 anchor matched %d times' % old3.count(anchor))
    inject = anchor + '''
# --- [c224 leg] optional third CoAtNet prediction set, fused into the Raptor ranks at 0.3 ---
# Fail-safe: any problem leaves the r384-only blend exactly as it was (today's 0.956 path).
_c224 = None
try:
    import os as _o2
    _p = '/kaggle/working/_c224.csv'
    if _o2.path.isfile(_p):
        _d = pd.read_csv(_p, dtype={'StudyInstanceUID': str})
        if set(_d.columns) >= set(labels) and sorted(_d['StudyInstanceUID']) == sorted(test_ids):
            _c224 = _d.set_index('StudyInstanceUID').loc[test_ids, labels]
            print('[c224] fused at 0.3 over %d studies' % len(_c224), flush=True)
        else:
            print('[c224] columns/rows mismatch -> NOT fused', flush=True)
    else:
        print('[c224] no _c224.csv -> NOT fused', flush=True)
except Exception as _e:
    _c224 = None
    print('[c224] load failed -> NOT fused: %r' % (_e,), flush=True)
'''
    new3 = old3.replace(anchor, inject)
    # the fusion itself: replace the per-finding Raptor rank with 0.7*r384 + 0.3*c224
    old_rank = "    rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(ours[c].values, ~our_fail)"
    if new3.count(old_rank) != 1:
        raise SystemExit('REFUSING: the rr/ro line matched %d times' % new3.count(old_rank))
    new_rank = ("    rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(ours[c].values, ~our_fail)\n"
                "    if _c224 is not None:                        # [c224 leg] 0.7 r384 / 0.3 r224, rank space\n"
                "        _r2 = pct_on(_c224[c].values, ~rap_fail)\n"
                "        _m = np.isfinite(rr) & np.isfinite(_r2)\n"
                "        rr = np.where(_m, 0.7 * np.nan_to_num(rr) + 0.3 * np.nan_to_num(_r2), rr)")
    new3 = new3.replace(old_rank, new_rank)

    # ---- 5. assemble: insert the two new cells right after the r384 writefile cell ----
    cells = copy.deepcopy(src['cells'])
    wf_idx = next(i for i, c in enumerate(cells)
                  if c.get('cell_type') == 'code' and 'raptor_run.py' in ''.join(c.get('source', [])))
    new_cells = [{'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': wf.splitlines(keepends=True)},
                 {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': runner.splitlines(keepends=True)}]
    cells[wf_idx:wf_idx] = new_cells
    cells[blend_idx + 2]['source'] = new3.splitlines(keepends=True)

    nb = dict(src)
    nb['cells'] = cells

    # ---- audit: every original cell identical except the blend cell ----
    changed = 0
    for i, (a, b) in enumerate(zip(src['cells'], [c for j, c in enumerate(cells) if j not in (wf_idx, wf_idx + 1)])):
        if ''.join(a.get('source', [])) != ''.join(b.get('source', [])):
            changed += 1
            if a is not src['cells'][blend_idx]:
                print('  changed cell %d (expected only the blend cell)' % i)
    print('cells: %d -> %d ; changed originals: %d (expect 1: the blend cell)' % (len(src['cells']), len(cells), changed))
    if changed != 1:
        raise SystemExit('REFUSING: %d original cells changed, expected exactly 1' % changed)

    out = json.dumps(nb, ensure_ascii=False)
    meta = json.loads((SRC / 'kernel-metadata.json').read_text(encoding='utf-8'))
    env = ('dataset_sources', 'kernel_sources', 'model_sources', 'competition_sources', 'docker_image',
           'machine_shape', 'enable_gpu', 'enable_tpu', 'enable_internet', 'language', 'kernel_type', 'keywords')
    before = {k: json.dumps(meta.get(k), sort_keys=True) for k in env}
    meta.pop('id_no', None)
    meta['id'], meta['title'], meta['code_file'], meta['is_private'] = NEW_ID, NEW_TITLE, NEW_CODE, True
    for k in env:
        if json.dumps(meta.get(k), sort_keys=True) != before[k]:
            raise SystemExit('REFUSING: environment field %s changed' % k)
    DST.mkdir(parents=True, exist_ok=True)
    (DST / NEW_CODE).write_text(out, encoding='utf-8')
    (DST / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('environment inherited unchanged (%d fields)' % len(env))
    print('wrote %s\n  notebook %d bytes sha %s' % (DST, len(out), sha(out.encode())[:20]))
    print('  NOT PUSHED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
