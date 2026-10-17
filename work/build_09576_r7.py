"""Build heliosli's 0.9576 notebook against the R7 bundle slug -- the "R7 + flip TTA" probe.

What the 0.9576 notebook actually changed vs the 0.9573 one (line-level diff, FACTS C182):
    BUNDLE_ID   '8c7b666e27dc0e8f' -> 'fe055406fad6a12f'   (= the R6 frozen bundle)
    FLIP_TTA    False              -> True                  ("package E: v4c's flip TTA on R6")
Everything else is comments.  The attached dataset is the same slug
(`rsna-knee-r6-frozen-fe055406`), whose *version 2* is the R7 bundle (bundle 8c7b666e) and whose
*version 1* is the R6 bundle (bundle fe055406) -- so the two notebooks resolve to different bundles
only if the kernel's dataset version PINNING survives, which a plain metadata fork does not
guarantee (`dataset_sources` carries no version).

This builder therefore produces the variant that is reproducible WITHOUT any version pinning:
    dataset slug -> `heliosli/rsna-knee-compact-r7cnxfsss` (known to pass the submit gate, C151)
    BUNDLE_ID    -> '8c7b666e27dc0e8f'  (what that slug actually contains -- C152 verified 8/8 sha256)
    FLIP_TTA     -> True                (unchanged from 0.9576)
so it isolates exactly ONE variable against our `rsna-knee-09573-r7ds`: flip TTA on the R7 bundle.

Usage: python work/build_09576_r7.py
"""
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'scratch/r9576'
OLD_DS = 'heliosli/rsna-knee-r6-frozen-fe055406'
NEW_DS = 'heliosli/rsna-knee-compact-r7cnxfsss'
OLD_BARE = OLD_DS.split('/')[-1]
NEW_BARE = NEW_DS.split('/')[-1]
OLD_BID = "BUNDLE_ID = 'fe055406fad6a12f'"
NEW_BID = "BUNDLE_ID = '8c7b666e27dc0e8f'"


def lines_of(c):
    return c['source'] if isinstance(c['source'], list) else [c['source']]


def set_lines(c, ls):
    c['source'] = ls if isinstance(c['source'], list) else ''.join(ls)


def main() -> int:
    src_nb = next(SRC.glob('*.ipynb'))
    raw = src_nb.read_bytes().decode('utf-8-sig')
    meta0 = json.loads((SRC / 'kernel-metadata.json').read_text(encoding='utf-8'))
    orig = json.loads(raw)
    if meta0['dataset_sources'].count(OLD_DS) != 1:
        raise SystemExit('REFUSING: metadata does not reference %r exactly once' % OLD_DS)
    if raw.count(OLD_BARE) != 1 or raw.count(OLD_BID) != 1:
        raise SystemExit('REFUSING: notebook text occurrences: bare=%d bid=%d'
                         % (raw.count(OLD_BARE), raw.count(OLD_BID)))

    nb = copy.deepcopy(orig)
    edits = []
    for ci, cell in enumerate(nb['cells']):
        ls = lines_of(cell)
        for li, l in enumerate(ls):
            new = l
            if OLD_BARE in new:
                new = new.replace(OLD_BARE, NEW_BARE)
                edits.append((ci, li, l, new))
            if OLD_BID in new:
                n2 = new.replace(OLD_BID, NEW_BID)
                if n2 != new:
                    edits.append((ci, li, new, n2))
                    new = n2
            ls[li] = new
        set_lines(cell, ls)
    if len(edits) != 2:
        raise SystemExit('REFUSING: expected exactly 2 changed lines, found %d' % len(edits))
    # ⚠️ A cell's source may be ONE string, so both substitutions can land in the same "line".
    # Audit per cell by exact text: new == old with exactly those two substitutions applied.
    touched = 0
    for ci, (co, cn) in enumerate(zip(orig['cells'], nb['cells'])):
        a = ''.join(lines_of(co))
        b = ''.join(lines_of(cn))
        expect = a.replace(OLD_BARE, NEW_BARE).replace(OLD_BID, NEW_BID)
        if b != expect:
            raise SystemExit('REFUSING: cell %d is not exactly the two substitutions' % ci)
        if b != a:
            touched += 1
    if touched != 1:
        raise SystemExit('REFUSING: %d cells changed, expected 1 (BUNDLE_ID and C_SLUG live together)' % touched)

    slug = 'rsna-knee-09576-r7'
    title = 'RSNA Knee 09576 r7'
    if re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', title.lower())).strip('-') != slug:
        raise SystemExit('REFUSING: title does not slugify to the id')
    meta = copy.deepcopy(meta0)
    meta.pop('id_no', None)
    meta['id'] = 'ppxl16/' + slug
    meta['title'] = title
    meta['code_file'] = slug + '.ipynb'
    meta['is_private'] = True
    meta['dataset_sources'] = [NEW_DS if s == OLD_DS else s for s in meta0['dataset_sources']]
    for k, v in meta0.items():
        if k in ('id', 'title', 'code_file', 'is_private', 'id_no', 'dataset_sources'):
            continue
        if json.dumps(meta.get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
            raise SystemExit('REFUSING: environment field %s changed' % k)
    # the notebook carries characters the Windows GBK pusher cannot encode -> serialise pure ASCII
    # (the JSON payload Jupyter parses is identical; asserted below).
    out = json.dumps(nb, ensure_ascii=True)
    if json.loads(out) != nb or not out.isascii():
        raise SystemExit('REFUSING: ASCII round-trip failed')
    try:
        out.encode('utf-8').decode('gbk')
    except UnicodeDecodeError as exc:
        raise SystemExit('REFUSING: not GBK-decodable: %s' % exc)
    dst = ROOT / 'kernels' / 'kernel-09576-r7'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / meta['code_file']).write_text(out, encoding='utf-8')
    (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('09576-r7: dataset %s -> %s ; %s -> %s ; FLIP_TTA unchanged (True)  ->  %s  (%d B, sha %s)'
          % (OLD_BARE, NEW_BARE, OLD_BID.split("'")[1], NEW_BID.split("'")[1], dst.name, len(out),
             hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('audit: exactly 2 lines changed (C_SLUG + BUNDLE_ID); everything else byte-identical')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
