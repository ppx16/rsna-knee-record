"""Download a SPECIFIC version of a Kaggle dataset.

Why (FACTS C182): heliosli's 0.9576 kernel is pinned to **version 1** of
`heliosli/rsna-knee-r6-frozen-fe055406` (the R6 bundle, bundle_id fe055406fad6a12f), while a fork of
it receives the LATEST version (v2 = the R7 bundle, 8c7b666e27dc0e8f) and dies on
`assert manifest['bundle_id'] == BUNDLE_ID: ('8c7b666e27dc0e8f', 'fe055406fad6a12f')`.
The CLI has no version flag, but the public REST endpoint accepts `datasetVersionNumber`.

Usage: python work/get_dataset_version.py <owner/slug> <version> <outdir>
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def creds() -> tuple[str, str]:
    p = Path.home() / '.kaggle' / 'kaggle.json'
    d = json.loads(p.read_text(encoding='utf-8'))
    return d['username'], d['key']


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    slug, version, outdir = argv[1], argv[2], Path(argv[3])
    user, key = creds()
    url = 'https://www.kaggle.com/api/v1/datasets/download/%s' % slug
    r = requests.get(url, params={'datasetVersionNumber': version}, auth=(user, key),
                     stream=True, timeout=1800)
    print('GET %s?datasetVersionNumber=%s -> %s' % (url, version, r.status_code))
    if r.status_code != 200:
        print(r.text[:400])
        return 1
    r.raw.decode_content = True
    outdir.mkdir(parents=True, exist_ok=True)
    zp = outdir / 'v%s.zip' % version
    n = 0
    with zp.open('wb') as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
            n += len(chunk)
    print('downloaded %d bytes -> %s' % (n, zp))
    with zipfile.ZipFile(zp) as z:
        z.extractall(outdir)
    for p in sorted(outdir.iterdir()):
        if p.is_file():
            print('   %-46s %d' % (p.name, p.stat().st_size))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
