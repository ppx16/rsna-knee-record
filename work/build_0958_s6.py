"""Add nartaa's S6 ConvNeXt all-data arm to the 0.958 base (the one recipe-level model we lack).

Why (FACTS C184/C185/C190): the public 0.957 recipe = Raptor-384 + Raptor-224 (mirror) + R7 + **S6
all-data (0.950 alone)**, with `compact = 0.60*rank(R7) + 0.40*rank(S6)`.  Our base has no S6 arm.
That model is the only genuinely independent, **non-OAI** model we can still add (its README: "No OAI
data was used for this model"), so it is both a diversity play for the private split and a hedge
against an adverse OAI ruling.

How it plugs in (C190 discovery): the base's compact worker takes ALL its geometry from the bundle's
`manifest.json` (`--backbone/--size/--crop/--eval-stride/--views/--slices/--phys-mm`).  So the S6 arm
needs no inference code of its own -- only
  (a) the S6 checkpoint copied into a bundle dir with a self-consistent manifest.json,
  (b) the same stage-2 controller run a SECOND time against that bundle,
  (c) one inserted line in the blend cell that folds S6 into the `ours` (compact) half at 0.40.

Everything is fail-safe: if the payload, the checkpoint, the run or the CSV is bad, the kernel keeps
exactly the submission it already had.

Usage: python work/build_0958_s6.py [<s6 weight in compact half> ...]
"""
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/kernel-0958-blend'
S6_DATASET = 'nartaa/rsna-knee-s6fall-weights'
S6_CKPT = 'E_prior_s42_fall.pt'
S6_SHA = '0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12'
S6_ARM = 'E_prior'
S6_BUNDLE_ID = 's6all-0950-ourown-20261017'
ANATOMY_SHA = '5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e'
ANCHOR_OLD = """ours = ours.set_index('StudyInstanceUID').loc[test_ids, labels]"""

PREP = '''# Stage 2c [s6]: prepare nartaa's S6 all-data arm (0.950 alone, non-OAI) as a SECOND compact bundle.
# The base's compact worker takes all its geometry from a bundle manifest, so the S6 arm needs no
# inference code of its own -- only a bundle dir with a self-consistent manifest.json.
# FAIL-SAFE: any problem here leaves _S6_DIR = None and stage 3 keeps exactly today's submission.
import io as _io, tarfile as _tarfile
S6_CKPT, S6_SHA, S6_ARM = '__CKPT__', '__SHA__', '__ARM__'
S6_BUNDLE_ID = '__BID__'
ANATOMY_SHA_FILE = 'anatomy.pt'
ANATOMY_SHA256 = '__ASHA__'
_S6_DIR = None
_s6_rep = {'status': 'not started'}
try:
    _ds = None
    for _p in Path('/kaggle/input').glob('**/' + S6_CKPT):
        _ds = _p.parent
        break
    assert _ds is not None, 'S6 checkpoint not found under /kaggle/input'
    _ck = _ds / S6_CKPT
    _sha = hashlib.sha256(_ck.read_bytes()).hexdigest()
    assert _sha == S6_SHA, 'S6 checkpoint sha256 mismatch: %s' % _sha
    _s6 = WORK / '_s6_bundle'
    _s6.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_ck, _s6 / S6_CKPT)
    # the code overlay: s6_extras.bin = magic 'K1PAYLOAD1\\n' (11 bytes) + an UNCOMPRESSED tar
    _extras = _ds / 's6_extras.bin'
    _code2 = WORK / '_s6_code'
    _code2.mkdir(parents=True, exist_ok=True)
    if _extras.is_file():
        _b = _extras.read_bytes()
        assert _b[:11] == b'K1PAYLOAD1\\n', 'unexpected payload magic %r' % _b[:11]
        with _tarfile.open(fileobj=_io.BytesIO(_b[11:])) as _t:
            _t.extractall(_code2)
        _s6_rep['extras_members'] = len(list(_code2.rglob('*')))
    else:
        _s6_rep['extras_members'] = 0
        print('[s6] s6_extras.bin absent -- running with the base code dir only', flush=True)
    _man = {'bundle_id': S6_BUNDLE_ID, 'arm': S6_ARM, 'backbone': 'convnext_small', 'size': 320,
            'crop': 288, 'eval_stride': 1, 'views': 'v4', 'slices': 3, 'phys_mm': 3.5,
            'models': [S6_CKPT], 'sha256': {S6_CKPT: S6_SHA}, 'gold_in_training': 0}
    (_s6 / 'manifest.json').write_text(json.dumps(_man, indent=1))
    _S6_DIR = _s6
    _s6_rep.update(status='prepared', bundle=str(_s6), sha256=_sha)
except BaseException as _e:
    _S6_DIR = None
    _s6_rep['status'] = 'fallback (kept today\\'s submission): %s: %s' % (type(_e).__name__, str(_e)[:250])
print('S6_PREP ' + json.dumps(_s6_rep, default=str)[:600], flush=True)
'''

RUN = '''# Stage 2d [s6]: run the SAME stage-2 controller against the S6 bundle.  FAIL-SAFE.
_S6_OK = False
try:
    assert _S6_DIR is not None, 'S6 bundle was not prepared'
    _man = json.loads((_S6_DIR / 'manifest.json').read_text())
    # ⚠️ the v3k views need the SKM-TEA anchor checkpoint at inference, else every study raises
    # "views v3k need --anchor-weights" and the receipt is ALL_FALLBACK (FACTS C191).  anatomy.pt
    # ships in heliosli's orthofoundation dataset, which this base already attaches.
    _anch = None
    for _p in Path('/kaggle/input').glob('**/' + ANATOMY_SHA_FILE):
        if hashlib.sha256(_p.read_bytes()).hexdigest() == ANATOMY_SHA256:
            _anch = _p
            break
    # accept any anatomy.pt when the pinned sha is not the one shipped, but say so
    if _anch is None:
        _any = sorted(Path('/kaggle/input').glob('**/' + ANATOMY_SHA_FILE))
        assert _any, 'no anatomy.pt under /kaggle/input'
        _anch = _any[0]
        print('[s6] WARNING: anatomy.pt sha256 does not match the pinned %s' % ANATOMY_SHA256, flush=True)
    _args = ['--backbone', _man['backbone'], '--size', str(_man['size']), '--crop', str(_man['crop']),
             '--eval-stride', str(_man['eval_stride']), '--views', _man['views'],
             '--slices', str(_man['slices']), '--phys-mm', str(_man['phys_mm']),
             '--anchor-weights', str(_anch)]
    _code = str(WORK / '_s6_code') + os.pathsep + str(CODE)
    # ⚠️ the extras' code must be importable in the CHILD too: the worker imports by module name,
    # so the overlay dir has to be on PYTHONPATH (passing it only via --code was not enough --
    # that produced ALL_FALLBACK on the first attempt, FACTS C191).
    _env = dict(os.environ, PYTHONPATH=str(WORK / '_s6_code') + os.pathsep + str(CODE) +
                os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
    _pred6 = TMP / 'prediction_s6'
    _rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA), '--split', 'test',
                          '--code', _code, '--bundle', str(_S6_DIR), '--arm', S6_ARM, '--output', str(_pred6),
                          '--amp', 'auto', '--loader-workers', '2', '--deadline', str(deadline),
                          '--worker', WORKER, '--worker-args', json.dumps(_args),
                          '--max-fallback-fraction', '1.0'], env=_env).returncode
    # copy the worker logs whatever happens -- without them an ALL_FALLBACK is undiagnosable
    for _log in sorted(_pred6.glob('worker*/log.txt')):
        try:
            shutil.copyfile(_log, WORK / ('s6_' + _log.parent.name + '_log.txt'))
        except Exception:
            pass
    if _rc != 0:
        raise RuntimeError('S6 controller exited %d' % _rc)
    shutil.copyfile(_pred6 / 'submission.csv', WORK / 's6_sub.csv')
    _rec = json.loads((_pred6 / 'submission_receipt.json').read_text())
    assert _rec.get('status') == 'PASS', 'S6 receipt status %r' % _rec.get('status')
    _S6_OK = True
    print('S6_STAGE_COMPLETE ' + json.dumps({k: _rec.get(k) for k in ('status', 'arm', 'rows', 'scored',
          'fallback_studies', 'seconds')}, default=str), flush=True)
except BaseException as _e:
    _S6_OK = False
    print('S6_STAGE_FALLBACK %s: %s' % (type(_e).__name__, str(_e)[:300]), flush=True)
'''

BLEND_INS = '''
# [s6 leg] fold nartaa's S6 all-data arm into the compact half at __W__ (fail-safe: only if the stage ran)
if globals().get('_S6_OK'):
    _s6f = pd.read_csv(WORK / 's6_sub.csv', dtype={'StudyInstanceUID': str})
    if set(labels) <= set(_s6f.columns) and sorted(_s6f['StudyInstanceUID']) == sorted(test_ids):
        _s6f = _s6f.set_index('StudyInstanceUID').loc[test_ids, labels]
        ours = (__COMP__ * ours.rank(pct=True) + __S6W__ * _s6f.rank(pct=True))
        print('[s6] folded into the compact half at __S6W__ over %d studies' % len(_s6f), flush=True)
    else:
        print('[s6] s6_sub.csv columns/rows mismatch -> NOT folded', flush=True)
else:
    print('[s6] stage did not run -> NOT folded', flush=True)
'''


def text_of(c):
    return c['source'] if isinstance(c['source'], str) else ''.join(c['source'])


def set_text(c, s):
    c['source'] = s if isinstance(c['source'], str) else s.splitlines(keepends=True)


def main(argv: list[str]) -> int:
    weights = [float(a) for a in argv[1:]] or [0.40]
    src_nb = next(BASE.glob('*.ipynb'))
    meta0 = json.loads((BASE / 'kernel-metadata.json').read_text(encoding='utf-8'))
    nb0 = json.loads(src_nb.read_bytes().decode('utf-8-sig'))
    if len(nb0['cells']) != 9:
        raise SystemExit('REFUSING: the base has %d cells, expected 9' % len(nb0['cells']))
    blend = text_of(nb0['cells'][8])
    if blend.count(ANCHOR_OLD) != 1:
        raise SystemExit('REFUSING: the blend anchor appears %d times' % blend.count(ANCHOR_OLD))
    if S6_DATASET in meta0['dataset_sources']:
        raise SystemExit('REFUSING: the S6 dataset is already attached')

    for w in weights:
        tag = 's6w%d' % round(w * 100)
        nb = copy.deepcopy(nb0)
        for cell in nb['cells']:
            set_text(cell, text_of(cell))
        # 1) two new cells BEFORE the blend cell
        prep = (PREP.replace('__CKPT__', S6_CKPT).replace('__SHA__', S6_SHA)
                .replace('__ARM__', S6_ARM).replace('__BID__', S6_BUNDLE_ID)
                .replace('__ASHA__', ANATOMY_SHA))
        nb['cells'].insert(8, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                               'source': prep.splitlines(keepends=True)})
        nb['cells'].insert(9, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                               'source': RUN.splitlines(keepends=True)})
        # 2) the blend cell (now index 10) gains exactly one block
        ins = (BLEND_INS.replace('__W__', str(w)).replace('__S6W__', str(w))
               .replace('__COMP__', str(round(1.0 - w, 10))))
        b = text_of(nb['cells'][10])
        nb_new = b.replace(ANCHOR_OLD, ANCHOR_OLD + '\n' + ins, 1)
        if nb_new == b:
            raise SystemExit('REFUSING[%s]: the blend insertion did nothing' % tag)
        set_text(nb['cells'][10], nb_new)
        # audit: original 9 cells unchanged except the blend cell, which differs by exactly the insertion
        for i in range(8):
            if text_of(nb['cells'][i]) != text_of(nb0['cells'][i]):
                raise SystemExit('REFUSING[%s]: cell %d changed' % (tag, i))
        changed = text_of(nb['cells'][10])
        if changed != b.replace(ANCHOR_OLD, ANCHOR_OLD + '\n' + ins, 1):
            raise SystemExit('REFUSING[%s]: the blend cell is not exactly anchor+insertion' % tag)
        if changed.replace('\n' + ins, '') != b:
            raise SystemExit('REFUSING[%s]: removing the insertion does not restore the blend cell' % tag)

        slug = 'rsna-knee-0958-s6-' + tag
        title = 'RSNA Knee 0958 s6 ' + tag
        if re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', title.lower())).strip('-') != slug:
            raise SystemExit('REFUSING[%s]: title does not slugify to the id' % tag)
        meta = copy.deepcopy(meta0)
        meta.pop('id_no', None)
        meta['id'] = 'ppxl16/' + slug
        meta['title'] = title
        meta['code_file'] = slug + '.ipynb'
        meta['is_private'] = True
        meta['dataset_sources'] = list(meta0['dataset_sources']) + [S6_DATASET]
        for k, v in meta0.items():
            if k in ('id', 'title', 'code_file', 'is_private', 'id_no', 'dataset_sources'):
                continue
            if json.dumps(meta.get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
                raise SystemExit('REFUSING[%s]: environment field %s changed' % (tag, k))
        out = json.dumps(nb, ensure_ascii=True)
        if json.loads(out) != nb or not out.isascii():
            raise SystemExit('REFUSING[%s]: ASCII round-trip failed' % tag)
        dst = ROOT / 'kernels' / ('kernel-0958-s6-' + tag)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / meta['code_file']).write_text(out, encoding='utf-8')
        (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print('S6 at %s : %s  (%d B, sha %s, cells 9->%d)' % (w, dst.name, len(out),
              hashlib.sha256(out.encode()).hexdigest()[:16], len(nb['cells'])))
    print('audit: cells 0-7 unchanged, two cells inserted before the blend, blend gains exactly one block')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
