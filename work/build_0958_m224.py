"""Add the anatomical mirror to the 224-crop Raptor arm (the "one free flip" that our base lacks).

What the gap is (FACTS C196): our base's Raptor stations are
  * cell 2, the 384-crop arm  -> ALREADY averages plain + anatomical-mirror views (`gpu_windows(..., mirror=True)`
                                 and `infer_pair_probs(model, plain, mirrored, device)`);
  * cell 4, the 224-crop arm  -> **plain only**: `_mirror_tri`, `mirror=True` and `infer_pair_probs` appear
                                 ZERO times.
The published 0.956 lineage fixes exactly that and documents the result verbatim
(`varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up`):

    "nartaa's 384-pixel CoAtNet already averages two views … The 224-pixel checkpoint … was published
     with one view only. This notebook gives it the mirror too, with nartaa's own mirror code."
    displayed score unchanged at 0.956, but rank 357 -> 289, "from bronze back into silver"

A within-band rank gain is worth real money right now: our 0.958 band has ~1155 teams and we sit around
position 207, with the silver cut about 70 places ahead.

How the port is made safe: the diff between our 224 script and that notebook's is EXACTLY
  (a) the mirror block (`SLOT_SIZES`, `SAGITTAL_SLOTS`, `_slot_index`, `_mirror_tri`, the `mirror=`
      parameter on `gpu_windows`, and `0.5*(plain + mirrored)` in the inference call), and
  (b) our own `raptor224_raw.csv` dump, which the blend cell needs and theirs does not write.
So the build takes THEIR script and puts OUR three-line raw dump back, anchored on the identical
`[gpu] done` print line.  Everything else in the base is untouched, and the audit below proves it.

Usage: python work/build_0958_m224.py
"""
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/kernel-0958-blend'
FLIP = ROOT / 'scratch/flip'
ANCHOR = '    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)'
DUMP = [
    '    _raw = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB)   # [blend] raw probabilities; a failed study stays 0.5',
    '    _raw.insert(0, "StudyInstanceUID", test_ids)',
    '    _raw.to_csv("/kaggle/working/raptor224_raw.csv", index=False)',
]
MIRROR_MARKERS = ('_mirror_tri', 'mirror=True', 'SLOT_SIZES')


def text_of(c):
    return c['source'] if isinstance(c['source'], str) else ''.join(c['source'])


def set_text(c, s):
    c['source'] = s if isinstance(c['source'], str) else s.splitlines(keepends=True)


def main() -> int:
    base_nb = next(BASE.glob('*.ipynb'))
    meta0 = json.loads((BASE / 'kernel-metadata.json').read_text(encoding='utf-8'))
    nb0 = json.loads(base_nb.read_bytes().decode('utf-8-sig'))
    flip_nb = json.loads(next(FLIP.glob('*.ipynb')).read_bytes().decode('utf-8-sig'))
    if len(nb0['cells']) != 9:
        raise SystemExit('REFUSING: the base has %d cells, expected 9' % len(nb0['cells']))
    src224 = text_of(flip_nb['cells'][9])
    if not src224.startswith('%%writefile /tmp/raptor224_run.py'):
        raise SystemExit('REFUSING: the donor cell 9 is not the 224 writefile cell')
    for m in MIRROR_MARKERS:
        if m not in src224:
            raise SystemExit('REFUSING: the donor script is missing the mirror marker %r' % m)
    if 'raptor224_raw' in src224:
        raise SystemExit('REFUSING: the donor already dumps raw -- the diff assumption is wrong')
    if src224.count(ANCHOR) != 1:
        raise SystemExit('REFUSING: the [gpu] done anchor appears %d times' % src224.count(ANCHOR))

    new224 = src224.replace(ANCHOR, ANCHOR + '\n' + '\n'.join(DUMP), 1)
    if new224.count('raptor224_raw.csv') != 1:
        raise SystemExit('REFUSING: the raw dump did not land exactly once')

    nb = copy.deepcopy(nb0)
    set_text(nb['cells'][4], new224)
    # audit: only cell 4 changed, and it is exactly donor + our dump
    changed = [i for i in range(9) if text_of(nb['cells'][i]) != text_of(nb0['cells'][i])]
    if changed != [4]:
        raise SystemExit('REFUSING: cells changed = %s, expected only [4]' % changed)
    if text_of(nb['cells'][4]).replace('\n' + '\n'.join(DUMP), '', 1) != src224:
        raise SystemExit('REFUSING: removing the dump does not restore the donor script')
    old224 = text_of(nb0['cells'][4])
    if 'mirror=True' in old224 or '_mirror_tri' in old224:
        raise SystemExit('REFUSING: the base 224 arm already had the mirror -- nothing to add')

    slug = 'rsna-knee-0958-m224'
    title = 'RSNA Knee 0958 m224'
    if re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', title.lower())).strip('-') != slug:
        raise SystemExit('REFUSING: title does not slugify to the id')
    meta = copy.deepcopy(meta0)
    meta.pop('id_no', None)
    meta['id'] = 'ppxl16/' + slug
    meta['title'] = title
    meta['code_file'] = slug + '.ipynb'
    meta['is_private'] = True
    for k, v in meta0.items():
        if k in ('id', 'title', 'code_file', 'is_private', 'id_no'):
            continue
        if json.dumps(meta.get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
            raise SystemExit('REFUSING: environment field %s changed' % k)
    out = json.dumps(nb, ensure_ascii=True)
    if json.loads(out) != nb or not out.isascii():
        raise SystemExit('REFUSING: ASCII round-trip failed')
    try:
        out.encode('utf-8').decode('gbk')
    except UnicodeDecodeError as exc:
        raise SystemExit('REFUSING: not GBK-decodable: %s' % exc)
    dst = ROOT / 'kernels' / 'kernel-0958-m224'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / meta['code_file']).write_text(out, encoding='utf-8')
    (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('224 arm now mirrors: %s  (%d B, sha %s)' % (dst.name, len(out),
                                                       hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('  old 224 script: %d chars, mirror markers=%s'
          % (len(old224), {m: old224.count(m) for m in MIRROR_MARKERS}))
    print('  new 224 script: %d chars, mirror markers=%s'
          % (len(new224), {m: new224.count(m) for m in MIRROR_MARKERS}))
    print('audit: only cell 4 changed, and it is the donor script plus our three raw-dump lines')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
