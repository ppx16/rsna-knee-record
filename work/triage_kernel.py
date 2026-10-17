"""Triage a public notebook in one command: is it submittable, and is its model actually new?

Written for the one event that can still move us: a new public release beating our 0.956.
heliosli went r6 -> r7 inside a day, so the next release may need to be captured quickly, and
three separate traps have already cost us time in this competition:

  * EMPTY SLUG  -- `dataset_sources` containing "" means a private dataset, so the notebook is not
    reproducible.  C119/C132, and it has now fired on 4 different notebooks.
  * RENAME      -- a dataset whose name changes but whose bytes do not.  `rsna-knee-compact-r6cnxfsss`
    is byte-identical to `rsna-knee-compact-v1` (C145); only the manifest's bundle_id and sha256 tell
    these apart (C148/A194).
  * TITLE       -- the score in a title is not a score.  Read `bestPublicScore` (C142/C143).

Usage:  python work/triage_kernel.py <owner/slug> [<owner/slug> ...]
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
COMP_ID = 154281

# bundle_id -> (what it is, what it scored inside the Raptor blend)
KNOWN_BUNDLES = {
    '209a32c8b50a6b1c': ('cnx_s6 (a.k.a. compact-v1 / r6cnxfsss)', '0.954'),
    '8c7b666e27dc0e8f': ('cnx_r7 (retrained, + fluid/state stackers)', '0.956'),
}
KNOWN_C_SLUGS = {'rsna-knee-compact-v1': 'same bytes as r6cnxfsss',
                 'rsna-knee-compact-r6cnxfsss': 'same bytes as compact-v1',
                 'rsna-knee-compact-r7cnxfsss': 'NEW generation (different weights)'}


def api_session():
    s = requests.Session()
    s.get('https://www.kaggle.com/competitions', timeout=60)
    return s, {'x-xsrf-token': s.cookies.get('XSRF-TOKEN', '')}


def scores_by_url() -> dict:
    """scriptUrl -> bestPublicScore, over the whole competition (works without login)."""
    s, hdr = api_session()
    out, page = {}, 1
    while page < 40:
        body = {'kernelFilterCriteria': {'search': '', 'listRequest': {
                    'competitionId': COMP_ID, 'sortBy': 'SCORE_DESCENDING', 'pageSize': 100,
                    'group': 'EVERYONE', 'page': page}},
                'detailFilterCriteria': {'deletedAccessBehavior': 'RETURN_NOTHING',
                                         'unauthorizedAccessBehavior': 'RETURN_NOTHING',
                                         'excludeResultsFilesOutputs': True},
                'readMask': 'pinnedKernels'}
        try:
            r = s.post('https://www.kaggle.com/api/i/kernels.KernelsService/ListKernels',
                       json=body, headers=hdr, timeout=90)
            if r.status_code != 200:
                break
            got = r.json().get('kernels', [])
        except Exception:  # noqa: BLE001
            break
        if not got:
            break
        for k in got:
            out[k.get('scriptUrl', '')] = k.get('bestPublicScore')
        page += 1
    return out


def pull(ref: str, dest: Path) -> bool:
    for attempt in range(4):
        r = subprocess.run(['kaggle', 'kernels', 'pull', ref, '-p', str(dest), '-m'],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode == 0:
            return True
        if 'SSLError' not in (r.stdout + r.stderr) and 'Max retries' not in (r.stdout + r.stderr):
            return False
        time.sleep(8 * (attempt + 1))
    return False


def fetch_manifest(slug: str, dest: Path):
    """manifest.json of a dataset, or None.  Never raises: absence is a normal answer."""
    if not slug:
        return None
    for attempt in range(3):
        r = subprocess.run(['kaggle', 'datasets', 'download', '-d', slug, '-f', 'manifest.json',
                            '-p', str(dest), '--force'],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        p = dest / 'manifest.json'
        if p.is_file():
            try:
                d = json.loads(p.read_text(encoding='utf-8'))
                p.unlink()
                return d
            except Exception:  # noqa: BLE001
                p.unlink(missing_ok=True)
        if 'SSLError' not in (r.stdout + r.stderr):
            return None
        time.sleep(6 * (attempt + 1))
    return None


def triage(ref: str, scores: dict, tmp: Path) -> dict:
    d = tmp / ref.replace('/', '__')
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    out = {'ref': ref, 'pulled': pull(ref, d)}
    if not out['pulled']:
        return out

    meta_path = d / 'kernel-metadata.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.is_file() else {}
    nbs = list(d.glob('*.ipynb'))
    out['title'] = meta.get('title')
    out['datasets'] = meta.get('dataset_sources') or []
    out['empty_slugs'] = [s for s in out['datasets'] if not str(s).strip()]
    out['score'] = scores.get('/code/' + ref)

    text = ''
    if nbs:
        raw = nbs[0].read_bytes()
        out['notebook_bytes'] = len(raw)
        import hashlib
        out['notebook_sha'] = hashlib.sha256(raw).hexdigest()[:16]
        try:
            nb = json.loads(raw.decode('utf-8'))
            text = '\n'.join(''.join(c.get('source', [])) for c in nb.get('cells', []))
        except Exception:  # noqa: BLE001
            pass

    def one(pat):
        m = re.search(pat, text)
        return m.group(1) if m else None

    out['weights'] = one(r'W_RAPTOR,\s*W_OURS\s*=\s*([0-9., ]+)')
    out['c_slug'] = one(r"C_SLUG\s*=\s*'([^']+)'")
    out['bundle_id'] = one(r"BUNDLE_ID\s*=\s*'([0-9a-f]+)'")
    out['deadline_h'] = one(r'DEADLINE_HOURS\s*=\s*([0-9.]+)')
    out['c_slug_note'] = KNOWN_C_SLUGS.get(out['c_slug'] or '', 'unknown slug')
    out['bundle_note'] = KNOWN_BUNDLES.get(out['bundle_id'] or '', ('unknown bundle', '?'))

    # confirm the bundle id against the dataset's OWN manifest (do not trust the notebook's string)
    if out['c_slug']:
        man = fetch_manifest('heliosli/' + out['c_slug'], tmp) or \
              fetch_manifest(next((s for s in out['datasets'] if out['c_slug'] in str(s)), ''), tmp)
        if man:
            out['dataset_bundle_id'] = man.get('bundle_id')
            out['dataset_built'] = man.get('built')
            out['dataset_gold_in_training'] = man.get('gold_in_training')
            out['dataset_declared_matches_code'] = (man.get('bundle_id') == out['bundle_id'])
    return out


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    refs = argv[1:]
    print('reading the competition kernel list for real bestPublicScore ...')
    scores = scores_by_url()
    print('  %d kernels listed' % len(scores))
    tmp = Path(tempfile.mkdtemp(prefix='triage_'))
    try:
        for ref in refs:
            t = triage(ref, scores, tmp)
            print('=' * 84)
            print(ref)
            if not t.get('pulled'):
                print('  !! could not pull (transport or missing) -- NOT a verdict')
                continue
            print('  title            : %s' % t.get('title'))
            print('  real best score  : %s   <-- read THIS, never the title' % t.get('score'))
            print('  datasets         : %s' % json.dumps(t.get('datasets'), ensure_ascii=False))
            if t['empty_slugs']:
                print('  *** NOT REPRODUCIBLE: %d empty slug(s) = private dataset(s)' % len(t['empty_slugs']))
            else:
                print('  reproducible     : yes (no empty slug)')
            print('  notebook         : %s bytes sha %s' % (t.get('notebook_bytes'), t.get('notebook_sha')))
            print('  W_RAPTOR,W_OURS  : %s' % t.get('weights'))
            print('  C_SLUG           : %s  (%s)' % (t.get('c_slug'), t.get('c_slug_note')))
            print('  BUNDLE_ID (code) : %s  -> %s' % (t.get('bundle_id'), t.get('bundle_note')))
            if 'dataset_bundle_id' in t:
                print('  BUNDLE_ID (data) : %s  built %s  gold_in_training %s  matches code: %s'
                      % (t['dataset_bundle_id'], t.get('dataset_built'),
                         t.get('dataset_gold_in_training'), t.get('dataset_declared_matches_code')))
            print('  DEADLINE_HOURS   : %s' % t.get('deadline_h'))
            known = t.get('bundle_id') in KNOWN_BUNDLES
            print('  VERDICT          : %s'
                  % ('NEW MODEL — not one of our two known bundles, worth a run'
                     if not known else
                     'known bundle (%s, scores %s in the Raptor blend) — only worth it if our score is below that'
                     % (KNOWN_BUNDLES[t['bundle_id']][0], KNOWN_BUNDLES[t['bundle_id']][1])))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
