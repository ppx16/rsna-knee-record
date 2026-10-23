"""The three-way stack: heliosli's 0.957 recipe + BOTH small-weight legs, for tomorrow's slots.

base  = `kernels/kernel-09573-r9all-w10`  (heliosli 0.957 recipe + the r9all leg at w=0.10, already
        run and verified -- its leg report said blended, coverage 1.0, child_returncode 0)
donor = `kernels/kernel-r7-c224-band32-w08`  (writefile cell with the band32 leg source, helpers,
        runner -- already run and verified once)

Both legs are the "small weight" setting that measured best (FACTS C178: on the r7 base the r9all
leg is a small POSITIVE at w=0.10 and clearly negative at 0.25/0.45).  Each stage is fail-safe: a
failure restores whatever the previous stage wrote.

Usage: python work/build_09573_r9all_band32.py [<w> ...]
"""
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/kernel-09573-r9all-w10'
DONOR = ROOT / 'kernels/kernel-r7-c224-band32-w08'
LEG_DATASET = 'ricecakar/rsna-knee-band32-ours'
LEG_MODEL = 'metaresearch/dinov2/PyTorch/small/1'
MD = """## Stage 5: ricecakar's public band32 leg on top of stage 4

Both added legs sit at the small weight that measured best for this family of models."""


def main(argv: list[str]) -> int:
    weights = [float(a) for a in argv[1:]] or [0.08]
    src_nb = next(BASE.glob('*.ipynb'))
    src_meta = json.loads((BASE / 'kernel-metadata.json').read_text(encoding='utf-8'))
    nb0 = json.loads(src_nb.read_bytes().decode('utf-8-sig'))
    if len(nb0['cells']) != 12:
        raise SystemExit('REFUSING: the base has %d cells, expected 12 (9 + r9all stage)' % len(nb0['cells']))
    dnb = json.loads(next(DONOR.glob('*.ipynb')).read_bytes().decode('utf-8'))
    writefile, helpers, runner = (copy.deepcopy(dnb['cells'][i]) for i in (10, 11, 12))
    if not ''.join(writefile['source']).startswith('%%writefile /kaggle/working/_band32_leg.py'):
        raise SystemExit('REFUSING: donor cell 10 is not the band32 writefile cell')
    if LEG_DATASET in src_meta['dataset_sources']:
        raise SystemExit('REFUSING: the band32 dataset is already attached')

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
        if [json.dumps(c) for c in nb['cells'][:12]] != [json.dumps(c) for c in nb0['cells']]:
            raise SystemExit('REFUSING[%s]: an original cell changed' % tag)
        slug = 'rsna-knee-09573-legs2-' + tag
        title = 'RSNA Knee 09573 legs2 ' + tag
        if re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', title.lower())).strip('-') != slug:
            raise SystemExit('REFUSING[%s]: title does not slugify to the id' % tag)
        meta = copy.deepcopy(src_meta)
        meta.pop('id_no', None)
        meta['id'] = 'ppxl16/' + slug
        meta['title'] = title
        meta['code_file'] = slug + '.ipynb'
        meta['is_private'] = True
        meta['dataset_sources'] = list(src_meta['dataset_sources']) + [LEG_DATASET]
        meta['model_sources'] = [LEG_MODEL] + list(src_meta.get('model_sources') or [])
        for k, v in src_meta.items():
            if k in ('id', 'title', 'code_file', 'is_private', 'id_no', 'dataset_sources', 'model_sources'):
                continue
            if json.dumps(meta.get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
                raise SystemExit('REFUSING[%s]: environment field %s changed' % (tag, k))
        out = json.dumps(nb, ensure_ascii=True)
        if json.loads(out) != nb or not out.isascii():
            raise SystemExit('REFUSING[%s]: ASCII round-trip failed' % tag)
        try:
            out.encode('utf-8').decode('gbk')
        except UnicodeDecodeError as exc:
            raise SystemExit('REFUSING[%s]: not GBK-decodable: %s' % (tag, exc))
        dst = ROOT / 'kernels' / ('kernel-09573-legs2-' + tag)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / meta['code_file']).write_text(out, encoding='utf-8')
        (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print('%s: r9all base(12 cells) + band32 writefile/helpers/runner(LEG_W=%s)  ->  %s  (%d B, sha %s)'
              % (tag, w, dst.name, len(out), hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('audit: the 12 original cells are unchanged; only the band32 dataset/model + 4 cells were added')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
