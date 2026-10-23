"""Build heliosli's 0.957 notebook against a dataset slug that PASSES the submit gate.

Why (FACTS C149/C151/C179):
  * `heliosli/0-9573-latest-public-dataset-score-in-one-click` is credited **0.957** and is fully
    public, but a byte-for-byte fork of it is REFUSED:
        "Submission not allowed: Your Notebook uses non-versioned datasets
         [/datasets/heliosli/rsna-knee-r6-frozen-fe055406]"
    -- the same gate C149 found, on the same author's newly uploaded bundle.
  * The notebook's own comment says what that dataset IS:
        C_SLUG = 'rsna-knee-r6-frozen-fe055406'   # its version 2 = the R7 bundle
                                                  # (manifest sha256 c4bb0f73... = r7cnxfsss's)
    i.e. the attached dataset is byte-identical to `heliosli/rsna-knee-compact-r7cnxfsss`, which we
    ALREADY know the gate accepts (C151: a byte-for-byte fork of the r7 notebook, attached to that
    slug, submitted as 57127191 and scored 0.956).
  * So the cheapest correct fix is the C149(2) one: swap the slug, keep everything else byte-identical.

⇒ The only genuinely new content in this notebook relative to our r7 recipe is the r224 stage
(`%%writefile /tmp/raptor224_run.py`, "0.8 centre crop of the 384-pixel field resized to 224") and
`A224 = 0.35` in the blend -- i.e. heliosli independently adopted the r224 axis we measured in
C173/C174, with his own geometry.  The optional A224 variants push that axis further.

Usage: python work/build_09573_r7ds.py [<a224> ...]      # no args -> the author's 0.35 unchanged
"""
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'scratch/r8new'
OLD_SLUG = 'heliosli/rsna-knee-r6-frozen-fe055406'
NEW_SLUG = 'heliosli/rsna-knee-compact-r7cnxfsss'
# the notebook refers to the dataset by its BARE slug (`C_SLUG = '<slug>'`), the metadata by owner/slug
OLD_BARE = OLD_SLUG.split('/')[-1]
NEW_BARE = NEW_SLUG.split('/')[-1]
OLD_A = 'A224 = 0.35'
AUTHOR_A = 0.35


def slugify(title: str) -> str:
    return re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', title.lower())).strip('-')


def lines_of(cell: dict) -> list:
    """A cell's `source` is either a LIST OF LINES or a SINGLE STRING (Jupyter allows both)."""
    return cell['source'] if isinstance(cell['source'], list) else [cell['source']]


def set_lines(cell: dict, lines: list) -> None:
    cell['source'] = lines if isinstance(cell['source'], list) else ''.join(lines)


def main(argv: list[str]) -> int:
    variants = ([('', AUTHOR_A)] if len(argv) == 1
                else [(('a%d' % round(float(a) * 100)), float(a)) for a in argv[1:]])
    src_nb = next(SRC.glob('*.ipynb'))
    raw = src_nb.read_bytes().decode('utf-8-sig')
    meta0 = json.loads((SRC / 'kernel-metadata.json').read_text(encoding='utf-8'))
    orig = json.loads(raw)
    if meta0['dataset_sources'].count(OLD_SLUG) != 1:
        raise SystemExit('REFUSING: the metadata does not reference %r exactly once' % OLD_SLUG)
    if raw.count(OLD_BARE) != 1:
        raise SystemExit('REFUSING: the notebook text contains %r %d times' % (OLD_BARE, raw.count(OLD_BARE)))

    for suffix, a in variants:
        nb = copy.deepcopy(orig)
        edits = []                                    # (cell index, line index, before, after)
        for ci, cell in enumerate(nb['cells']):
            ls = lines_of(cell)
            # 1) the dataset slug -- exactly one line, and only there
            for li, l in enumerate(ls):
                if OLD_BARE in l:
                    new = l.replace(OLD_BARE, NEW_BARE)
                    if new != l:
                        ls[li] = new
                        edits.append((ci, li, l, new))
            # 2) A224, only when the caller asked for a different value
            if a != AUTHOR_A:
                for li, l in enumerate(ls):
                    if OLD_A in l:
                        new = l.replace(OLD_A, 'A224 = %s' % a)
                        edits.append((ci, li, l, new))
                        ls[li] = new
            set_lines(cell, ls)
        expect = 1 + (0 if a == AUTHOR_A else 1)
        if len(edits) != expect:
            raise SystemExit('REFUSING[%s]: expected %d changed lines, found %d' % (suffix, expect, len(edits)))
        if [e[3] for e in edits].count(edits[0][3]) > 1 and a == AUTHOR_A:
            raise SystemExit('REFUSING[%s]: the slug swap was not unique' % suffix)

        # audit, line by line, against the ORIGINAL: every cell identical except those exact lines
        changed = {}
        for ci, (co, cn) in enumerate(zip(orig['cells'], nb['cells'])):
            lo, ln = lines_of(co), lines_of(cn)
            if lo != ln:
                if len(lo) != len(ln):
                    raise SystemExit('REFUSING[%s]: cell %d changed length' % (suffix, ci))
                for li, (x, y) in enumerate(zip(lo, ln)):
                    if x != y:
                        changed[(ci, li)] = (x, y)
        want = {(e[0], e[1]): (e[2], e[3]) for e in edits}
        if changed != want:
            raise SystemExit('REFUSING[%s]: unexpected line changes %s' % (suffix, sorted(set(changed) ^ set(want))))

        slug = 'rsna-knee-09573-r7ds' + (('-' + suffix) if suffix else '')
        title = 'RSNA Knee 09573 r7ds' + ((' ' + suffix) if suffix else '')
        if slugify(title) != slug:
            raise SystemExit('REFUSING[%s]: title %r does not slugify to %r' % (suffix, title, slugify(title)))
        meta = copy.deepcopy(meta0)
        meta.pop('id_no', None)
        meta['id'] = 'ppxl16/' + slug
        meta['title'] = title
        meta['code_file'] = slug + '.ipynb'
        meta['is_private'] = True
        meta['dataset_sources'] = [NEW_SLUG if s == OLD_SLUG else s for s in meta0['dataset_sources']]
        for k, v in meta0.items():
            if k in ('id', 'title', 'code_file', 'is_private', 'id_no', 'dataset_sources'):
                continue
            if json.dumps(meta.get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
                raise SystemExit('REFUSING[%s]: environment field %s changed' % (suffix, k))
        out = json.dumps(nb, ensure_ascii=False)
        try:
            out.encode('utf-8').decode('gbk')
        except UnicodeDecodeError as exc:
            raise SystemExit('REFUSING[%s]: not GBK-decodable: %s' % (suffix, exc))
        dst = ROOT / 'kernels' / ('kernel-09573-r7ds' + (('-' + suffix) if suffix else ''))
        dst.mkdir(parents=True, exist_ok=True)
        (dst / meta['code_file']).write_text(out, encoding='utf-8')
        (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print('%s: dataset %s -> %s ; A224 = %s  ->  %s  (%d B, sha %s)'
              % (suffix or 'base', OLD_BARE, NEW_BARE, a, dst.name, len(out),
                 hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('audit: the ONLY edits are the C_SLUG line (1) and, for variants, the A224 line (1)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
