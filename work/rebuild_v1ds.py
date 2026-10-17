"""Rebuild the fork against rsna-knee-compact-v1 instead of rsna-knee-compact-r6cnxfsss.

WHY.  Kaggle's code-competition submit gate refused our fork with

    Submission not allowed: Your Notebook uses non-versioned datasets
    [/datasets/heliosli/rsna-knee-compact-r6cnxfsss] (see Dataset Settings).

A four-way probe (work/probe_submit_gate.py, 2026-10-13) showed it is NOT the dataset that is
banned -- it is whether *this kernel version* carries a pinned version of it:

    le4ves/...dual-blend-raptor-convnext-s6   (r6cnxfsss) -> 400 non-versioned
    heliosli/blend-mine-in-...                (r6cnxfsss) -> 403 permission  (dataset gate PASSED)
    goodpjw2008/...coatnet-2-5d-convnext-mil (compact-v1) -> 200
    ppxl16/rsna-knee-blendgold-restored       (compact-v1) -> 200

So a fork loses the pin and cannot be submitted, while kernels attached to `compact-v1` pass.

WHY THIS IS NOT A SCIENTIFIC VARIABLE.  FACTS C145 proved the two datasets are byte identical:
their manifest.json files have the SAME md5 (2057920e89de63acdbd6e9f7bb3e1a18), the same
bundle_id 209a32c8b50a6b1c, and identical sha256 for all six payload files.  Swapping which of
two byte-identical mounts the notebook reads therefore changes no data at all.

WHAT CHANGES.  Only the dataset slug string, everywhere it appears.  Everything else in the
notebook is asserted byte-identical to the 0.956 fork.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'kernels/kernel-raptor-s6-fork'
DST = ROOT / 'kernels/kernel-raptor-s6-v1ds'

OLD_SLUG = 'heliosli/rsna-knee-compact-r6cnxfsss'
NEW_SLUG = 'heliosli/rsna-knee-compact-v1'
OLD_NAME = 'rsna-knee-compact-r6cnxfsss'
NEW_NAME = 'rsna-knee-compact-v1'

NEW_ID = 'ppxl16/rsna-knee-raptor-s6-v1ds'
NEW_TITLE = 'RSNA Knee Raptor S6 v1ds'
NEW_CODE = 'rsna-knee-raptor-s6-v1ds.ipynb'


def _find_all(hay: str, needle: str):
    out, i = [], hay.find(needle)
    while i != -1:
        out.append(i)
        i = hay.find(needle, i + 1)
    return out


def main() -> int:
    src_nb = SRC / 'rsna-knee-raptor-s6-fork.ipynb'
    raw = src_nb.read_bytes()
    print('source notebook : %s bytes  sha256 %s'
          % (len(raw), hashlib.sha256(raw).hexdigest()[:24]))

    text = raw.decode('utf-8')
    n_old_name = text.count(OLD_NAME)
    if n_old_name == 0:
        raise SystemExit('REFUSING: the source notebook does not mention %s' % OLD_NAME)
    fixed = text.replace(OLD_NAME, NEW_NAME)
    print('replaced %d occurrence(s) of %r -> %r' % (n_old_name, OLD_NAME, NEW_NAME))
    for i, pos in enumerate(_find_all(text, OLD_NAME), 1):
        ctx = text[max(0, pos - 60):pos + len(OLD_NAME) + 40].replace('\n', '\\n')
        print('   [%d] ...%s...' % (i, ctx))

    # AUDIT (linear, constructive): fixed must equal text with every OLD_NAME occurrence
    # replaced by NEW_NAME and nothing else touched.  `NEW_NAME.join(text.split(OLD_NAME))`
    # is exactly that statement, and `OLD_NAME not in fixed` proves none was missed.
    # (A plain reverse-replace cannot be used as the check: NEW_NAME *also* occurs in the
    #  notebook's credits paragraph, so replacing it back would corrupt that text.  And a
    #  SequenceMatcher diff on a 516 KB string is O(n^2) -- it hung for five minutes.)
    pieces = text.split(OLD_NAME)
    if len(pieces) != n_old_name + 1:
        raise SystemExit('REFUSING: split produced %d pieces for %d occurrences'
                         % (len(pieces), n_old_name))
    if NEW_NAME.join(pieces) != fixed:
        raise SystemExit('REFUSING: fixed is not exactly text with the slug swapped')
    if OLD_NAME in fixed:
        raise SystemExit('REFUSING: the old slug survived the swap')
    print('audit: fixed == NEW_NAME.join(text.split(OLD_NAME)) and no old slug remains')
    print('       => the only change is the %d slug occurrence(s); nothing else was touched'
          % n_old_name)

    out_nb = fixed.encode('utf-8')
    try:
        out_nb.decode('gbk')
    except UnicodeDecodeError as exc:
        raise SystemExit('REFUSING: not GBK-decodable: %s' % exc)

    # the two byte-identical datasets: assert the claim we are relying on, from the saved manifests
    import hashlib as _h
    m1 = ROOT / 'scratch/cmp/v1_manifest.json'
    m0 = ROOT / 'scratch/cmp/r6_manifest.json'
    if m0.is_file() and m1.is_file():
        h0 = _h.md5(m0.read_bytes()).hexdigest()
        h1 = _h.md5(m1.read_bytes()).hexdigest()
        print('manifest md5: r6cnxfsss=%s  v1=%s  IDENTICAL=%s' % (h0, h1, h0 == h1))
        if h0 != h1:
            raise SystemExit('REFUSING: the two manifests are NOT identical -- C145 no longer holds')
    else:
        print('!! manifest pair not on disk; relying on recorded FACTS C145 (md5 2057920e...)')

    DST.mkdir(parents=True, exist_ok=True)
    (DST / NEW_CODE).write_bytes(out_nb)

    meta = json.loads((SRC / 'kernel-metadata.json').read_text(encoding='utf-8'))
    env_before = {k: json.dumps(meta.get(k), sort_keys=True)
                  for k in ('kernel_sources', 'model_sources', 'competition_sources', 'docker_image',
                            'machine_shape', 'enable_gpu', 'enable_tpu', 'enable_internet',
                            'language', 'kernel_type', 'keywords')}
    ds = list(meta.get('dataset_sources') or [])
    if OLD_SLUG not in ds:
        raise SystemExit('REFUSING: %s is not in dataset_sources %s' % (OLD_SLUG, ds))
    meta['dataset_sources'] = [NEW_SLUG if d == OLD_SLUG else d for d in ds]
    if len(meta['dataset_sources']) != len(ds):
        raise SystemExit('REFUSING: dataset_sources length changed')

    meta.pop('id_no', None)
    meta['id'] = NEW_ID
    meta['title'] = NEW_TITLE
    meta['code_file'] = NEW_CODE
    meta['is_private'] = True

    for k, v in env_before.items():
        after = json.dumps(meta.get(k), sort_keys=True)
        if after != v:
            raise SystemExit('REFUSING: environment field %s changed (%s -> %s)' % (k, v, after))
        print('  inherit %-20s %s' % (k, v[:100]))
    print('  dataset_sources      %s' % meta['dataset_sources'])

    (DST / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    # final audit: the rebuilt notebook is the source plus the swap, by construction
    print('\nwrote %s' % DST)
    print('  notebook %d bytes  sha256 %s' % (len(out_nb), hashlib.sha256(out_nb).hexdigest()))
    print('  audit: rebuilt notebook == source with the slug swapped, nothing else')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
