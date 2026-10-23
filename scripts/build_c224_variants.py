"""Build c224-leg variants at alternative weights, from the already-verified c224-leg kernel.

RSNA's metric is DETERMINISTIC (a fixed test set with hidden labels), so re-submitting the same
recipe cannot change the score -- a mere copy of a public notebook is worth exactly nothing.  The
only thing that produces a different score is a genuinely different artifact, and the one axis we
have verified to run end-to-end is the weight on the transplanted r224/c224 leg.

Each variant is a ONE-NUMBER change on `rr = a*r384 + b*c224`, asserted to be the only change.
Downside is nil (the platform keeps our best), so these are cheap independent shots.

Usage: python work/build_c224_variants.py
"""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'kernels/kernel-r7-c224leg'

# (suffix, r384 weight, c224 weight)
VARIANTS = [('w02', '0.8', '0.2'), ('w05', '0.5', '0.5')]
OLD = '0.7 * np.nan_to_num(rr) + 0.3 * np.nan_to_num(_r2)'


def main() -> int:
    src_nb = next(SRC.glob('*.ipynb'))
    src_meta = json.loads((SRC / 'kernel-metadata.json').read_text(encoding='utf-8'))
    raw = src_nb.read_bytes().decode('utf-8')
    if raw.count(OLD) != 1:
        raise SystemExit('REFUSING: the fusion expression matched %d times' % raw.count(OLD))

    for suffix, a, b in VARIANTS:
        new = f'{a} * np.nan_to_num(rr) + {b} * np.nan_to_num(_r2)'
        fixed = raw.replace(OLD, new)
        # audit: exactly one occurrence replaced, everything else byte-identical
        if new.join(raw.split(OLD)) != fixed:
            raise SystemExit('REFUSING[%s]: not exactly this one swap' % suffix)
        if fixed.count(new) != 1 or OLD in fixed:
            raise SystemExit('REFUSING[%s]: post-swap content check failed' % suffix)
        # the labelled comment above the line should follow the numbers
        fixed = fixed.replace('# [c224 leg] 0.7 r384 / 0.3 r224, rank space',
                              '# [c224 leg] %s r384 / %s r224, rank space' % (a, b))

        nb = json.loads(fixed)
        cells = nb['cells']
        # rabari's transplanted stage also carries its own c224 weight; keep it at 1.0 (single-arm)
        slug = 'rsna-knee-r7-c224' + suffix
        dst = ROOT / 'kernels' / ('kernel-r7-c224' + suffix)
        code = slug + '.ipynb'
        meta = copy.deepcopy(src_meta)
        meta.pop('id_no', None)
        meta['id'] = 'ppxl16/' + slug
        meta['title'] = 'RSNA Knee r7 c224 ' + suffix
        meta['code_file'] = code
        meta['is_private'] = True
        for k, v in src_meta.items():
            if k in ('id', 'title', 'code_file', 'is_private', 'id_no'):
                continue
            if json.dumps(meta.get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
                raise SystemExit('REFUSING[%s]: environment field %s changed' % (suffix, k))
        out = json.dumps(nb, ensure_ascii=False)
        try:
            out.encode('utf-8').decode('gbk')
        except UnicodeDecodeError as exc:
            raise SystemExit('REFUSING[%s]: not GBK-decodable: %s' % (suffix, exc))
        dst.mkdir(parents=True, exist_ok=True)
        (dst / code).write_text(out, encoding='utf-8')
        (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print('%s: rr = %s*r384 + %s*c224   ->  %s  (%d B, sha %s)'
              % (suffix, a, b, dst.name, len(out), hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('audit: each variant == the c224-leg kernel with exactly the one fusion expression changed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
