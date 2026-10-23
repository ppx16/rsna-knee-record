"""Fork heliosli's public 0.956 notebook into our account, inheriting its ENVIRONMENT verbatim.

Why this exists (C139/A170/A183): when we fork a public notebook we must not hand-copy the
environment fields.  This builder reads the upstream `kernel-metadata.json` and copies it whole,
overriding ONLY the four fields that describe *whose kernel it is*.  Every field that decides what
the kernel can SEE at runtime (dataset_sources, kernel_sources, model_sources, competition_sources,
docker_image, machine_shape, enable_gpu/tpu/internet) is asserted unchanged and non-empty.

The notebook body is copied byte-for-byte except for ONE character: upstream contains U+8DEF (a
mangled middle dot) in a credits paragraph, and `kaggle kernels push` on Windows reads the code file
with the GBK locale codec, which that byte sequence breaks.  Replacing it with an ASCII hyphen makes
the file pure ASCII.  It is markdown prose, so nothing executable changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NB = ROOT / 'scratch/top/blend-mine-in-private-lb-luck-to-all-upvoters/blend-mine-in-private-lb-luck-to-all-upvoters.ipynb'
SRC_META = ROOT / 'scratch/top/blend-mine-in-private-lb-luck-to-all-upvoters/kernel-metadata.json'
DST = ROOT / 'kernels/kernel-raptor-s6-fork'

NEW_ID = 'ppxl16/rsna-knee-raptor-s6-fork'
NEW_TITLE = 'RSNA Knee Raptor S6 fork'
NEW_CODE = 'rsna-knee-raptor-s6-fork.ipynb'

# every field that decides what the kernel can see/use at runtime
ENV_FIELDS = ('dataset_sources', 'kernel_sources', 'model_sources', 'competition_sources',
              'docker_image', 'machine_shape', 'enable_gpu', 'enable_tpu', 'enable_internet',
              'language', 'kernel_type', 'keywords')


def main() -> int:
    upstream_raw = SRC_NB.read_bytes()
    up_sha = hashlib.sha256(upstream_raw).hexdigest()
    print('upstream notebook  : %s' % SRC_NB.name)
    print('  bytes            : %d' % len(upstream_raw))
    print('  sha256           : %s' % up_sha)

    up_meta = json.loads(SRC_META.read_text(encoding='utf-8'))
    print('upstream id        : %s' % up_meta.get('id'))
    print('upstream title     : %r' % up_meta.get('title'))

    # ---- the one byte-level change, and a hard check that it is the ONLY one ----
    text = upstream_raw.decode('utf-8')
    n_bad = sum(1 for ch in text if ord(ch) > 127)
    if n_bad != 1 or '\u8def' not in text:
        raise SystemExit('REFUSING: expected exactly one non-ASCII char (U+8DEF), found %d' % n_bad)
    fixed = text.replace('\u8def', '-')
    if any(ord(ch) > 127 for ch in fixed):
        raise SystemExit('REFUSING: file still contains non-ASCII after the substitution')

    DST.mkdir(parents=True, exist_ok=True)
    nb_bytes = fixed.encode('utf-8')
    try:
        nb_bytes.decode('gbk')          # the exact codec `kaggle kernels push` uses here
    except UnicodeDecodeError as exc:
        raise SystemExit('REFUSING: still not GBK-decodable: %s' % exc)
    (DST / NEW_CODE).write_bytes(nb_bytes)
    print('fork notebook      : %s bytes (upstream %d, delta %+d)'
          % (len(nb_bytes), len(upstream_raw), len(nb_bytes) - len(upstream_raw)))
    print('  sha256           : %s' % hashlib.sha256(nb_bytes).hexdigest())

    # only the substitution may differ: re-apply it to the original and compare
    ref = upstream_raw.decode('utf-8').replace('\u8def', '-').encode('utf-8')
    if ref != nb_bytes:
        raise SystemExit('REFUSING: fork differs from upstream by more than the one substitution')
    print('  diff vs upstream : EXACTLY the one U+8DEF substitution, nothing else')

    # ---- metadata: inherit whole, override only identity ----
    meta = dict(up_meta)
    # Every environment field must be PRESENT upstream (that is what "inherit, do not hand-copy"
    # means).  Only these three must also be non-empty: an empty kernel_sources / model_sources is
    # a legitimate value ("this kernel needs none"), not a missing field.
    MUST_BE_NONEMPTY = ('dataset_sources', 'competition_sources', 'docker_image')
    for f in ENV_FIELDS:
        if f not in up_meta:
            raise SystemExit('REFUSING: upstream has no %s field; cannot inherit it' % f)
        if f in MUST_BE_NONEMPTY and up_meta[f] in (None, '', []):
            raise SystemExit('REFUSING: upstream %s is empty (%r); cannot inherit it'
                             % (f, up_meta[f]))
    env_before = {f: json.dumps(up_meta.get(f), sort_keys=True) for f in ENV_FIELDS}

    meta.pop('id_no', None)
    meta['id'] = NEW_ID
    meta['title'] = NEW_TITLE
    meta['code_file'] = NEW_CODE
    meta['is_private'] = True

    for f in ENV_FIELDS:
        after = json.dumps(meta.get(f), sort_keys=True)
        if after != env_before[f]:
            raise SystemExit('REFUSING: %s changed during the fork (%s -> %s)'
                             % (f, env_before[f], after))
        print('  inherit %-20s %s' % (f, env_before[f][:110]))

    (DST / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('\nwrote %s' % DST)
    print('  id    : %s' % meta['id'])
    print('  title : %r' % meta['title'])
    print('  gpu   : %s   shape: %s   internet: %s'
          % (meta['enable_gpu'], meta.get('machine_shape'), meta.get('enable_internet')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
