"""Point the 0.9586 fork at the single-version R6 bundle, because the gate refuses the pinned-v1 slug.

`heliosli/this-is-best-public-dataset-new-version-0-9586` is credited **0.9586** -- the newest public
artifact and a full thousandth above our 0.958.  It adds goodpjw2008's 2.5D ConvNeXt MIL reader to the
same base we run, at his own per-finding weights (0.003-0.035), fail-safe.

The one thing that cannot be forked is its bundle pinning (FACTS C182): the notebook expects
`BUNDLE_ID = 'fe055406fad6a12f'` -- version **1** of `heliosli/rsna-knee-r6-frozen-fe055406`, i.e. the
R6 bundle -- while `dataset_sources` can only name the slug, and a fork receives the LATEST version
(v2 = the R7 bundle, `8c7b666e`), which trips `assert manifest['bundle_id'] == BUNDLE_ID`.

So this rewrites exactly one string -- the slug in `C_SLUG` -- to `rsna-knee-r6-frozen-fe055406-w1`,
the **single-version public copy of the R6 bundle** that our 0.958 base already uses.  The metadata's
`dataset_sources` entry is rewritten to match, and everything else (including `kernel_sources`, which
carries goodpjw's notebook) is left exactly as inherited.

Audited: exactly one cell changes, and only by the line-anchored slug substitution.
Usage: python work/patch_09586_slug.py
"""
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/kernel-09586'
OLD_BARE = 'rsna-knee-r6-frozen-fe055406'
NEW_BARE = 'rsna-knee-r6-frozen-fe055406-w1'
OLD_DS = 'heliosli/' + OLD_BARE
NEW_DS = 'heliosli/' + NEW_BARE
PAT = re.compile(r"(?m)^C_SLUG = '" + re.escape(OLD_BARE) + r"'[^\n]*$")


def text_of(c):
    return c['source'] if isinstance(c['source'], str) else ''.join(c['source'])


def set_text(c, s):
    c['source'] = s if isinstance(c['source'], str) else s.splitlines(keepends=True)


def main() -> int:
    nb_path = next(BASE.glob('*.ipynb'))
    meta_path = BASE / 'kernel-metadata.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    nb = json.loads(nb_path.read_bytes().decode('utf-8-sig'))

    hits = [i for i, c in enumerate(nb['cells']) if PAT.search(text_of(c))]
    if len(hits) != 1:
        raise SystemExit('REFUSING: the C_SLUG line matched %d cells, expected 1' % len(hits))
    ci = hits[0]
    old_text = text_of(nb['cells'][ci])
    new_line = "C_SLUG = '%s'   # [gate] single-version copy of the same R6 bundle (v1 of the original slug)" % NEW_BARE
    new_text = PAT.sub(new_line, old_text, count=1)
    if new_text == old_text:
        raise SystemExit('REFUSING: the substitution changed nothing')
    if meta['dataset_sources'].count(OLD_DS) != 1:
        raise SystemExit('REFUSING: metadata does not name %r exactly once' % OLD_DS)

    nb2 = copy.deepcopy(nb)
    set_text(nb2['cells'][ci], new_text)
    changed = [i for i, (a, b) in enumerate(zip(nb['cells'], nb2['cells'])) if text_of(a) != text_of(b)]
    if changed != [ci]:
        raise SystemExit('REFUSING: cells changed = %s, expected only [%d]' % (changed, ci))

    meta['dataset_sources'] = [NEW_DS if s == OLD_DS else s for s in meta['dataset_sources']]
    if OLD_DS in meta['dataset_sources'] or NEW_DS not in meta['dataset_sources']:
        raise SystemExit('REFUSING: the metadata swap did not land')
    if meta['dataset_sources'].count(NEW_DS) != 1:
        raise SystemExit('REFUSING: the new slug appears %d times in metadata' % meta['dataset_sources'].count(NEW_DS))

    out = json.dumps(nb2, ensure_ascii=True)
    if json.loads(out) != nb2 or not out.isascii():
        raise SystemExit('REFUSING: ASCII round-trip failed')
    try:
        out.encode('utf-8').decode('gbk')
    except UnicodeDecodeError as exc:
        raise SystemExit('REFUSING: not GBK-decodable: %s' % exc)
    nb_path.write_text(out, encoding='utf-8')
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('patched %s: %s -> %s' % (nb_path.name, OLD_BARE, NEW_BARE))
    print('  cell %d: %d -> %d chars' % (ci, len(old_text), len(new_text)))
    print('  dataset_sources: %s' % meta['dataset_sources'])
    print('  kernel_sources (unchanged): %s' % meta.get('kernel_sources'))
    print('  sha256 %s' % hashlib.sha256(out.encode()).hexdigest()[:16])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
