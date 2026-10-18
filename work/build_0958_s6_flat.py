"""Build the UN-FITTED twin of the 0.958 artifact: flat per-finding weights + the S6 arm.

Why this exists (FACTS C184/C186/C194):
  * the 0.958 notebook's `W_RAPTOR_C` was changed on 2026-10-16 -- five findings reweighted
    (Lateral OA 0.2, PF OA 0.6, Synovitis 0.8, Baker's 0.35) -- and that is the ONLY difference from
    the 0.9576 version.  Those five numbers are fitted on the PUBLIC half, whose standard error is
    ~0.003 (larger than the whole spread of the top 400 teams), so the +0.0004 they bought cannot be
    confirmed and may not transfer to the private split, which is what decides the medal;
  * so this kernel is the same artifact with the weights put back to flat 0.5 (i.e. no per-finding
    override at all) while keeping the S6 arm at 0.40.  It is the "least fitted" artifact we can
    produce from public parts, and it exists so that at the deadline we can choose between a
    public-tuned and an un-tuned twin with identical displayed scores rather than only holding one.

Change from `kernel-0958-s6-s6w40`: exactly ONE line (W_RAPTOR_C), audited per cell.

Usage: python work/build_0958_s6_flat.py
"""
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/kernel-0958-s6-s6w40'
PAT = re.compile(r'(?m)^W_RAPTOR_C = \{.*')
FLAT = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.5, 'PF OA': 0.5}   # [flat] no per-finding override: the un-fitted recipe"


def text_of(cell):
    return cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])


def set_text(cell, s):
    cell['source'] = s if isinstance(cell['source'], str) else s.splitlines(keepends=True)


def main() -> int:
    src_nb = next(BASE.glob('*.ipynb'))
    meta0 = json.loads((BASE / 'kernel-metadata.json').read_text(encoding='utf-8'))
    orig = json.loads(src_nb.read_bytes().decode('utf-8-sig'))
    hits = [i for i, c in enumerate(orig['cells']) if PAT.search(text_of(c))]
    if len(hits) != 1:
        raise SystemExit('REFUSING: the W_RAPTOR_C assignment appears in %d cells, expected 1' % len(hits))
    ci = hits[0]
    old_text = text_of(orig['cells'][ci])
    new_text = PAT.sub(FLAT, old_text, count=1)
    if new_text == old_text:
        raise SystemExit('REFUSING: the substitution changed nothing')
    nb = copy.deepcopy(orig)
    set_text(nb['cells'][ci], new_text)
    touched = 0
    for k, (co, cn) in enumerate(zip(orig['cells'], nb['cells'])):
        if text_of(co) != text_of(cn):
            if k != ci or text_of(cn) != PAT.sub(FLAT, text_of(co), count=1):
                raise SystemExit('REFUSING: cell %d is not exactly the W_RAPTOR_C swap' % k)
            touched += 1
    if touched != 1:
        raise SystemExit('REFUSING: %d cells changed, expected 1' % touched)

    slug = 'rsna-knee-0958-s6-flat'
    title = 'RSNA Knee 0958 s6 flat'
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
    dst = ROOT / 'kernels' / 'kernel-0958-s6-flat'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / meta['code_file']).write_text(out, encoding='utf-8')
    (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('flat weights + S6@0.40 -> %s  (%d B, sha %s)' % (dst.name, len(out),
                                                           hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('old line:', PAT.search(old_text).group(0)[:110])
    print('new line:', PAT.search(new_text).group(0)[:110])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
