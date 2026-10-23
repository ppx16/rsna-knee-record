"""Verify a downloaded bundle against the sha256 map in its own manifest.json.

This is the check FACTS C145/A194 demand: "same architecture" may be read off file sizes, but
"same weights" can only be read off hashes.  Here we confirm that (a) the archive really contains
what the manifest promises, and (b) r7's weights differ from r6/v1's on every fold model.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'downloads/r7bundle/rsna-knee-compact-r7cnxfsss.zip'
DEST = ROOT / 'downloads/r7bundle/extracted'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        print('archive members:')
        for i in z.infolist():
            print('   %-28s %12d' % (i.filename, i.file_size))
        z.extractall(DEST)

    man = json.loads((DEST / 'manifest.json').read_text(encoding='utf-8'))
    print('\nbundle_id   : %s' % man['bundle_id'])
    print('tag / arm   : %s / %s' % (man['tag'], man['arm']))
    print('built       : %s' % man['built'])
    print('gold_in_training %s   official_labels_read_by_trainer %s'
          % (man['gold_in_training'], man['official_labels_read_by_trainer']))

    ok = bad = 0
    print('\nsha256 verification against the manifest:')
    for name, want in sorted(man['sha256'].items()):
        p = DEST / name
        if not p.is_file():
            print('   MISSING  %s' % name)
            bad += 1
            continue
        got = sha256(p)
        if got == want:
            print('   OK       %-26s %s' % (name, got[:20]))
            ok += 1
        else:
            print('   MISMATCH %-26s want %s got %s' % (name, want[:20], got[:20]))
            bad += 1

    # and the decisive comparison: do r7's fold weights differ from r6/v1's?
    old = json.loads((ROOT / 'scratch/cmp/v1_manifest.json').read_text(encoding='utf-8'))
    print('\nr7 vs r6/v1 weight identity:')
    for name in sorted(set(man['sha256']) | set(old['sha256'])):
        a = man['sha256'].get(name, '<absent>')
        b = old['sha256'].get(name, '<absent>')
        same = 'IDENTICAL' if a == b else 'DIFFERENT'
        print('   %-26s %s' % (name, same))

    print('\n%d ok, %d bad' % (ok, bad))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
