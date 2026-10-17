"""Stack the two changes that each measured positive: heliosli's 0.957 recipe + the r9all leg at w=0.10.

Evidence being stacked (both from the live board, not from theory):
  * C180: `rsna-knee-09573-r7ds` = the public 0.957 notebook with only the dataset-slug gate fix.
  * C173/C178: on the r7 base, the r9all leg at w=0.10 became the CREDITED submission
    (`LastSubmissionDate` moved to 57185321), i.e. at that weight the leg is a small POSITIVE,
    while w=0.25 and w=0.45 were clearly negative (0.955 / 0.953).  So the leg's only useful
    setting is a small one.

The runner and helpers are lifted VERBATIM from `kernels/kernel-r7-c224-r9all-w10` (already run and
verified: `r9all_leg_report.json` said `blended`, coverage 1.0, child_returncode 0).  heliosli's
notebook provides every global the runner needs (`T0`, `WORK`, `test_ids`, `labels`).

Usage: python work/build_09573_r9all.py [<w> ...]
"""
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the base can be any of our 0.957/0.958-family bases: override with KRSNA_BASE_DIR / KRSNA_TAG
BASE = Path(os.environ['KRSNA_BASE_DIR']) if os.environ.get('KRSNA_BASE_DIR') else ROOT / 'kernels/kernel-09573-r7ds'
TAG = os.environ.get('KRSNA_TAG', '09573')
DONOR = ROOT / 'kernels/kernel-r7-c224-r9all-w10'  # helpers + runner, already verified once
LEG_DATASET = 'bliverrigpdoge/rsna-knee-leg-r9all'
MD = """## Stage 4: bliverrigpdoge's public r9all leg, rank-blended at the very end

Lifted verbatim from our verified `rsna-knee-r7-c224-r9all-w10` kernel.  On the r7 base this leg at
this weight measured the same displayed score as the base, while 0.25 and 0.45 were clearly worse
(0.955 / 0.953) -- so a SMALL share of it is the only setting worth testing, and this kernel asks
whether it changes anything on the newer base."""


def main(argv: list[str]) -> int:
    weights = [float(a) for a in argv[1:]] or [0.10]
    src_nb = next(BASE.glob('*.ipynb'))
    src_meta = json.loads((BASE / 'kernel-metadata.json').read_text(encoding='utf-8'))
    raw = src_nb.read_bytes().decode('utf-8-sig')
    nb0 = json.loads(raw)
    if len(nb0['cells']) != 9:
        raise SystemExit('REFUSING: the base notebook has %d cells, expected 9' % len(nb0['cells']))
    dnb = json.loads(next(DONOR.glob('*.ipynb')).read_bytes().decode('utf-8'))
    helpers = copy.deepcopy(dnb['cells'][10])
    runner = copy.deepcopy(dnb['cells'][11])
    htxt = ''.join(helpers['source'])
    rtxt = ''.join(runner['source'])
    for needle in ('_leg_rank01', '_leg_blend'):
        if needle not in htxt:
            raise SystemExit('REFUSING: the helper cell is missing %r' % needle)
    for needle in ('LEG_W = 0.1', 'r9all_leg_report.json', 'submission.csv'):
        if needle not in rtxt:
            raise SystemExit('REFUSING: the runner is missing %r' % needle)
    if LEG_DATASET in src_meta['dataset_sources']:
        raise SystemExit('REFUSING: the leg dataset is already attached')

    for w in weights:
        tag = 'w%02d' % round(w * 100)
        nb = copy.deepcopy(nb0)
        r = runner
        old_w = re.search(r'LEG_W = [0-9.]+', ''.join(r['source'])).group(0)
        new_w = 'LEG_W = %s' % w
        r = copy.deepcopy(runner)
        if new_w != old_w:
            r['source'] = [l.replace(old_w, new_w) for l in (r['source'] if isinstance(r['source'], list) else [r['source']])]
            if isinstance(runner['source'], str):
                r['source'] = ''.join(r['source'])
            if ''.join(r['source']).count(new_w) != 1 or old_w in ''.join(r['source']):
                raise SystemExit('REFUSING[%s]: the leg weight was not swapped exactly once' % tag)
        if ''.join(r['source']).count(new_w) != 1:
            raise SystemExit('REFUSING[%s]: LEG_W does not appear exactly once' % tag)
        for kind, body in (('markdown', MD), ('code', copy.deepcopy(helpers)), ('code', r)):
            cell = {'cell_type': kind, 'metadata': {}, 'source': body['source'] if kind == 'code' else body}
            if kind == 'code':
                cell.setdefault('execution_count', None)
                cell.setdefault('outputs', [])
            nb['cells'].append(cell)
        if [json.dumps(c) for c in nb['cells'][:9]] != [json.dumps(c) for c in nb0['cells']]:
            raise SystemExit('REFUSING[%s]: an original cell changed' % tag)

        slug = 'rsna-knee-%s-r9all-%s' % (TAG, tag)
        title = 'RSNA Knee %s r9all %s' % (TAG, tag)
        if re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', title.lower())).strip('-') != slug:
            raise SystemExit('REFUSING[%s]: title does not slugify to the id' % tag)
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
                raise SystemExit('REFUSING[%s]: environment field %s changed' % (tag, k))
        out = json.dumps(nb, ensure_ascii=False)
        try:
            out.encode('utf-8').decode('gbk')
        except UnicodeDecodeError as exc:
            raise SystemExit('REFUSING[%s]: not GBK-decodable: %s' % (tag, exc))
        dst = ROOT / 'kernels' / ('kernel-%s-r9all-%s' % (TAG, tag))
        dst.mkdir(parents=True, exist_ok=True)
        (dst / meta['code_file']).write_text(out, encoding='utf-8')
        (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print('%s: r7ds(9 cells) + helpers + runner(LEG_W=%s)  ->  %s  (%d B, sha %s)'
              % (tag, w, dst.name, len(out), hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('audit: the 9 original cells are unchanged; only the leg dataset + 3 cells were added')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
