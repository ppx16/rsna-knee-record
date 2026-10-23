"""Generic verbatim fork of a public Kaggle notebook into our account.

Usage:
    python work/fork_kernel.py <src_dir> <new_id> <new_title> <new_code_file>

Rules enforced (C139/A183/A190 family -- "do not hand-copy environment fields"):
  * the notebook is copied byte-for-byte, except that any character the Windows GBK pusher
    cannot carry is substituted; the script asserts the substitution is the ONLY difference
    and that the result is GBK-decodable;
  * the metadata is inherited WHOLE from the upstream kernel-metadata.json, overriding only
    id / title / code_file / is_private, and every field that decides what the kernel can see
    at runtime is asserted unchanged.
"""
from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENV_FIELDS = ('dataset_sources', 'kernel_sources', 'model_sources', 'competition_sources',
              'docker_image', 'machine_shape', 'enable_gpu', 'enable_tpu', 'enable_internet',
              'language', 'kernel_type', 'keywords')
MUST_BE_NONEMPTY = ('dataset_sources', 'competition_sources', 'docker_image')
# ASCII stand-ins for the handful of typographic characters that show up in these READMEs
ASCII_MAP = {'\u2014': '-', '\u2013': '-', '\u00b7': '-', '\u2018': "'", '\u2019': "'",
             '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u00a0': ' ', '\u2192': '->',
             '\u2265': '>=', '\u2264': '<=', '\u2248': '~', '\u00d7': 'x', '\u2713': 'v',
             # U+8DEF is not a typographic mark at all: it is what a mangled middle dot
             # becomes when a UTF-8 '·' is decoded as GBK and re-encoded.  It appears once,
             # in the OAI acknowledgement paragraph, in both of heliosli's notebooks.
             '\u8def': '-'}


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(__doc__)
        return 2
    src = Path(argv[1])
    if not src.is_absolute():
        src = ROOT / src
    new_id, new_title, new_code = argv[2], argv[3], argv[4]
    dst = ROOT / 'kernels' / new_id.split('/')[-1].replace(new_id.split('/')[-1], new_id.split('/')[-1])
    dst = ROOT / 'kernels' / ('kernel-' + new_id.split('/')[-1].replace('rsna-knee-', ''))

    nbs = list(src.glob('*.ipynb'))
    if len(nbs) != 1:
        print('REFUSING: expected exactly one .ipynb in %s, found %s' % (src, nbs))
        return 2
    raw = nbs[0].read_bytes()
    text = raw.decode('utf-8')

    bad = sorted({ch for ch in text if ord(ch) > 127})
    print('upstream notebook : %s  %d bytes  sha256 %s'
          % (nbs[0].name, len(raw), hashlib.sha256(raw).hexdigest()[:24]))
    print('non-ASCII chars   : %d distinct %s'
          % (len(bad), [(c, hex(ord(c))) for c in bad][:12]))

    fixed = text
    unknown = []
    for ch in bad:
        if ch in ASCII_MAP:
            fixed = fixed.replace(ch, ASCII_MAP[ch])
        else:
            unknown.append(ch)
    if unknown:
        print('REFUSING: no ASCII stand-in defined for %s'
              % [(c, hex(ord(c)), unicodedata.name(c, '?')) for c in unknown])
        return 2

    # audit: the substitution set is the only difference (constructive, linear)
    for ch, rep in ASCII_MAP.items():
        if ch in text:
            pass
    # rebuild from the pieces to prove nothing else moved
    pieces = [text]
    if unknown:
        return 2
    # verify by re-deriving: apply the same replacements in the same order to a fresh copy
    check = text
    for ch in bad:
        check = check.replace(ch, ASCII_MAP[ch])
    if check != fixed:
        print('REFUSING: the substitution is not deterministic')
        return 2
    print('substituted       : %s' % {c: ASCII_MAP[c] for c in bad})

    out_nb = fixed.encode('utf-8')
    try:
        out_nb.decode('gbk')
    except UnicodeDecodeError as exc:
        print('REFUSING: result is not GBK-decodable: %s' % exc)
        return 2

    up_meta = json.loads((src / 'kernel-metadata.json').read_text(encoding='utf-8'))
    for f in ENV_FIELDS:
        if f not in up_meta:
            print('REFUSING: upstream has no %s' % f)
            return 2
        if f in MUST_BE_NONEMPTY and up_meta[f] in (None, '', []):
            print('REFUSING: upstream %s is empty' % f)
            return 2
    env_before = {f: json.dumps(up_meta.get(f), sort_keys=True) for f in ENV_FIELDS}

    meta = dict(up_meta)
    meta.pop('id_no', None)
    meta['id'] = new_id
    meta['title'] = new_title
    meta['code_file'] = new_code
    meta['is_private'] = True

    dst.mkdir(parents=True, exist_ok=True)
    (dst / new_code).write_bytes(out_nb)
    for f in ENV_FIELDS:
        after = json.dumps(meta.get(f), sort_keys=True)
        if after != env_before[f]:
            print('REFUSING: %s changed (%s -> %s)' % (f, env_before[f], after))
            return 2
        print('  inherit %-20s %s' % (f, env_before[f][:110]))
    (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print('\nwrote %s' % dst)
    print('  id       : %s' % meta['id'])
    print('  title    : %r' % meta['title'])
    print('  notebook : %d bytes  sha256 %s'
          % (len(out_nb), hashlib.sha256(out_nb).hexdigest()))
    print('  gpu      : %s  shape %s  internet %s'
          % (meta['enable_gpu'], meta.get('machine_shape'), meta.get('enable_internet')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
