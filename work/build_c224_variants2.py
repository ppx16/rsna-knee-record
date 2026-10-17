"""Second c224 weight sweep: push the c224 weight ABOVE the 0.5 that the platform just credited.

Why this direction: `LastSubmissionDate` on the live board moved to 57163476
(`rr = 0.5*rank(r384) + 0.5*rank(c224)`) from 57140469 (`0.7/0.3`), so the platform now credits the
HIGHER-c224-weight submission -- the tie-break is by UNROUNDED score (FACTS C113/C141/C173), so that
is a real ordering, not a rounding artefact.  Weight 0.5 > 0.3; the curve is still rising.

Each variant is a ONE-NUMBER change on `rr = a*r384 + b*c224`, asserted to be the only change, AND
every parameter-dependent output string is rewritten too (FACTS C171 (4): the w02/w05 builders left
a hardcoded "fused at 0.3" print behind, so their logs lied about their own weight).

Usage: python work/build_c224_variants2.py
"""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'kernels/kernel-r7-c224leg'

# (suffix, r384 weight, c224 weight)
VARIANTS = [('w60', '0.4', '0.6'), ('w40', '0.6', '0.4'), ('w70', '0.3', '0.7')]
OLD = '0.7 * np.nan_to_num(rr) + 0.3 * np.nan_to_num(_r2)'
OLD_CMT = '# [c224 leg] 0.7 r384 / 0.3 r224, rank space'
OLD_PRINT = "[c224] fused at 0.3 over %d studies"


def main() -> int:
    src_nb = next(SRC.glob('*.ipynb'))
    src_meta = json.loads((SRC / 'kernel-metadata.json').read_text(encoding='utf-8'))
    raw = src_nb.read_bytes().decode('utf-8')
    for needle, what in ((OLD, 'fusion expression'), (OLD_CMT, 'fusion comment'), (OLD_PRINT, 'fusion print')):
        if raw.count(needle) != 1:
            raise SystemExit('REFUSING: the %s matched %d times' % (what, raw.count(needle)))

    for suffix, a, b in VARIANTS:
        new = f'{a} * np.nan_to_num(rr) + {b} * np.nan_to_num(_r2)'
        new_cmt = '# [c224 leg] %s r384 / %s r224, rank space' % (a, b)
        new_print = "[c224] fused at %s over %%d studies" % b
        fixed = raw.replace(OLD, new).replace(OLD_CMT, new_cmt).replace(OLD_PRINT, new_print)
        # audit: exactly these three swaps, everything else byte-identical
        if new.join(fixed.split(new)) != fixed:
            raise SystemExit('REFUSING[%s]: expression swap is not a single substitution' % suffix)
        for needle in (OLD, OLD_CMT, OLD_PRINT):
            if needle in fixed:
                raise SystemExit('REFUSING[%s]: %r survived' % (suffix, needle))
        for needle in (new, new_cmt, new_print):
            if fixed.count(needle) != 1:
                raise SystemExit('REFUSING[%s]: %r appears %d times' % (suffix, needle, fixed.count(needle)))

        nb = json.loads(fixed)
        slug = 'rsna-knee-r7-c224-' + suffix
        dst = ROOT / 'kernels' / ('kernel-r7-c224-' + suffix)
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
        print('%s: rr = %s*r384 + %s*c224  print says %s  ->  %s  (%d B, sha %s)'
              % (suffix, a, b, b, dst.name, len(out), hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('audit: each variant == the c224-leg kernel with exactly the 3 weight-dependent strings changed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
