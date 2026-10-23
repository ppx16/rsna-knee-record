"""Stack band32 onto heliosli's 0.957 recipe (the most architecturally distant public leg).

Why this pairing (C177/C178/C180):
  * `rsna-knee-09573-r7ds` is the public 0.957 notebook with only the dataset-slug gate fix;
  * the band32 leg is a DINOv2 ViT + per-finding slot attention, self-reported val macro AUC
    0.8728-0.8880 over 10 members -- a DIFFERENT architecture family from every arm in the base
    (CoAtNet x2 + compact ConvNeXt), which is exactly the property C178 identified as the one that
    can pay: an extra leg helps only when it is new relative to the base, and hurts when it is a
    weaker member of a family already present (the r9all leg, a CoAtNet, at w>=0.25);
  * the leg is already proven to run end to end with the member-count fix: `band32_leg_report.json`
    said `blended`, coverage 1.0, n_expected 10, child_returncode 0, at 5.43 s/study steady state.

The cells are lifted VERBATIM from `kernels/kernel-r7-c224-band32-w08` (writefile cell with the leg
source, helpers, runner).  heliosli's notebook provides every global the runner needs.

Usage: python work/build_09573_band32.py [<w> ...]
"""
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the base can be any of our 0.957-family bases: override with KRSNA_BASE_DIR / KRSNA_TAG
BASE = Path(os.environ['KRSNA_BASE_DIR']) if os.environ.get('KRSNA_BASE_DIR') else ROOT / 'kernels/kernel-09573-r7ds'
TAG = os.environ.get('KRSNA_TAG', '09573')
DONOR = ROOT / 'kernels/kernel-r7-c224-band32-w08'
LEG_DATASET = 'ricecakar/rsna-knee-band32-ours'
LEG_MODEL = 'metaresearch/dinov2/PyTorch/small/1'
MD = """## Stage 4: ricecakar's public band32 leg, rank-blended at the very end

Lifted verbatim from our verified `rsna-knee-r7-c224-band32-w08` kernel (its leg report said
`blended`, coverage 1.0, 10 declared members).  This is the one added leg whose architecture family
differs from every arm already in the base."""


def main(argv: list[str]) -> int:
    weights = [float(a) for a in argv[1:]] or [0.08]
    src_nb = next(BASE.glob('*.ipynb'))
    src_meta = json.loads((BASE / 'kernel-metadata.json').read_text(encoding='utf-8'))
    nb0 = json.loads(src_nb.read_bytes().decode('utf-8-sig'))
    if len(nb0['cells']) != 9:
        raise SystemExit('REFUSING: the base notebook has %d cells, expected 9' % len(nb0['cells']))
    dnb = json.loads(next(DONOR.glob('*.ipynb')).read_bytes().decode('utf-8'))
    writefile, helpers, runner = (copy.deepcopy(dnb['cells'][i]) for i in (10, 11, 12))
    wf = ''.join(writefile['source'])
    if not wf.startswith('%%writefile /kaggle/working/_band32_leg.py'):
        raise SystemExit('REFUSING: the donor cell 10 is not the leg-source writefile cell')
    htxt, rtxt = ''.join(helpers['source']), ''.join(runner['source'])
    for needle in ('_leg_rank01', '_leg_blend'):
        if needle not in htxt:
            raise SystemExit('REFUSING: the helper cell is missing %r' % needle)
    for needle in ('_band32_leg.py', 'RSNA_N_EXPECTED', 'band32_leg_report.json'):
        if needle not in rtxt:
            raise SystemExit('REFUSING: the runner is missing %r' % needle)
    if LEG_DATASET in src_meta['dataset_sources']:
        raise SystemExit('REFUSING: the leg dataset is already attached')

    for w in weights:
        tag = 'w%02d' % round(w * 100)
        r = copy.deepcopy(runner)
        ls = r['source'] if isinstance(r['source'], list) else [r['source']]
        old_w = re.search(r'LEG_W = [0-9.]+', ''.join(ls)).group(0)
        new_w = 'LEG_W = %s' % w
        if new_w != old_w:
            ls = [l.replace(old_w, new_w) for l in ls]
            r['source'] = ls if isinstance(runner['source'], list) else ''.join(ls)
        if ''.join(r['source']).count(new_w) != 1:
            raise SystemExit('REFUSING[%s]: LEG_W does not appear exactly once' % tag)

        nb = copy.deepcopy(nb0)
        for kind, body in (('markdown', MD), ('code', writefile), ('code', helpers), ('code', r)):
            cell = {'cell_type': kind, 'metadata': {}}
            cell['source'] = body if kind == 'markdown' else body['source']
            if kind == 'code':
                cell['execution_count'] = None
                cell['outputs'] = []
            nb['cells'].append(cell)
        if [json.dumps(c) for c in nb['cells'][:9]] != [json.dumps(c) for c in nb0['cells']]:
            raise SystemExit('REFUSING[%s]: an original cell changed' % tag)

        slug = 'rsna-knee-%s-band32-%s' % (TAG, tag)
        title = 'RSNA Knee %s band32 %s' % (TAG, tag)
        if re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', title.lower())).strip('-') != slug:
            raise SystemExit('REFUSING[%s]: title does not slugify to the id' % tag)
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
                raise SystemExit('REFUSING[%s]: environment field %s changed' % (tag, k))
        # the leg source carries a character the Windows GBK pusher cannot encode, so serialise the
        # whole notebook as pure ASCII (the JSON payload Jupyter parses is identical -- asserted).
        out = json.dumps(nb, ensure_ascii=True)
        if json.loads(out) != nb:
            raise SystemExit('REFUSING[%s]: the ASCII-escaped notebook does not round-trip' % tag)
        if not out.isascii():
            raise SystemExit('REFUSING[%s]: the notebook is still not pure ASCII' % tag)
        try:
            out.encode('utf-8').decode('gbk')
        except UnicodeDecodeError as exc:
            raise SystemExit('REFUSING[%s]: not GBK-decodable: %s' % (tag, exc))
        dst = ROOT / 'kernels' / ('kernel-%s-band32-%s' % (TAG, tag))
        dst.mkdir(parents=True, exist_ok=True)
        (dst / meta['code_file']).write_text(out, encoding='utf-8')
        (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print('%s: r7ds(9 cells) + writefile + helpers + runner(LEG_W=%s)  ->  %s  (%d B, sha %s)'
              % (tag, w, dst.name, len(out), hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('audit: the 9 original cells are unchanged; only the leg dataset/model + 4 cells were added')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
