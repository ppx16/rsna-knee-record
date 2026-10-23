"""A224 variants on the 0.958 base (the current reproducible public ceiling).

The 0.958 notebook fixes `A224 = 0.35` -- the share of the Raptor half given to nartaa's 224-crop
checkpoint.  On the OLDER r7 base we swept c224 shares 0.2/0.3/0.5/0.8 and every one displayed
0.956, i.e. no measurable effect (C189); this base has never been swept, and the displayed score is
quantised, so a real but small shift could tip 0.958 -> 0.959.

Implementation notes learned the hard way:
  * a cell's `source` is either a LIST OF LINES or ONE STRING (A206), and here it is one string, so the
    substitution is done on the joined text with a line-anchored regex -- not per list element;
  * the literal "A224 = 0.35" also occurs inside a comment ("..., A224 = 0.35 fixed before any run;"),
    so a bare substring replace would silently edit the prose too.  The regex is anchored to a line
    that STARTS with the assignment.

Usage: python work/build_0958_a224.py [<a224> ...]
"""
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/kernel-0958-blend'
PAT = re.compile(r'(?m)^A224 = 0\.35[^\n]*$')


def text_of(cell):
    return cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])


def set_text(cell, s):
    cell['source'] = s if isinstance(cell['source'], str) else s.splitlines(keepends=True)


def main(argv: list[str]) -> int:
    weights = [float(a) for a in argv[1:]] or [0.50]
    src_nb = next(BASE.glob('*.ipynb'))
    meta0 = json.loads((BASE / 'kernel-metadata.json').read_text(encoding='utf-8'))
    orig = json.loads(src_nb.read_bytes().decode('utf-8-sig'))
    hits = [i for i, c in enumerate(orig['cells']) if PAT.search(text_of(c))]
    if len(hits) != 1:
        raise SystemExit('REFUSING: the A224 assignment line appears in %d cells, expected 1' % len(hits))
    ci = hits[0]

    for a in weights:
        tag = 'a%d' % round(a * 100)
        nb = copy.deepcopy(orig)
        old_text = text_of(nb['cells'][ci])
        new_text = PAT.sub('A224 = %s' % a, old_text, count=1)
        if new_text == old_text:
            raise SystemExit('REFUSING[%s]: the substitution changed nothing' % tag)
        set_text(nb['cells'][ci], new_text)
        # audit: exactly that one cell changed, and it is exactly the anchored substitution
        touched = 0
        for k, (co, cn) in enumerate(zip(orig['cells'], nb['cells'])):
            ao, an = text_of(co), text_of(cn)
            if ao != an:
                if k != ci or an != PAT.sub('A224 = %s' % a, ao, count=1):
                    raise SystemExit('REFUSING[%s]: cell %d is not exactly the A224 swap' % (tag, k))
                touched += 1
        if touched != 1:
            raise SystemExit('REFUSING[%s]: %d cells changed, expected 1' % (tag, touched))
        # the prose may mention 0.35 but must not have been rewritten
        if 'A224 = %s' % a not in new_text or new_text.count('A224 = %s' % a) != 1:
            raise SystemExit('REFUSING[%s]: the new value does not appear exactly once in the cell' % tag)

        slug = 'rsna-knee-0958-blend-' + tag
        title = 'RSNA Knee 0958 blend ' + tag
        if re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', title.lower())).strip('-') != slug:
            raise SystemExit('REFUSING[%s]: title does not slugify to the id' % tag)
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
                raise SystemExit('REFUSING[%s]: environment field %s changed' % (tag, k))
        out = json.dumps(nb, ensure_ascii=True)
        if json.loads(out) != nb or not out.isascii():
            raise SystemExit('REFUSING[%s]: ASCII round-trip failed' % tag)
        try:
            out.encode('utf-8').decode('gbk')
        except UnicodeDecodeError as exc:
            raise SystemExit('REFUSING[%s]: not GBK-decodable: %s' % (tag, exc))
        dst = ROOT / 'kernels' / ('kernel-0958-blend-' + tag)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / meta['code_file']).write_text(out, encoding='utf-8')
        (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print('A224 -> %s : %s  (%d B, sha %s)' % (a, dst.name, len(out),
                                                   hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('audit: exactly one cell changed, and only by the line-anchored A224 substitution')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
