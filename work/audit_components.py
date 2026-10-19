"""Systematic component audit: what do the other 0.956-0.958 notebooks have that we might not?

The 224-mirror gap (FACTS C196) was found by reading ONE notebook's markdown, not by guessing.  So this
pulls the remaining high-scoring notebooks and prints, for each, the lines of prose that describe what it
ADDS -- filtered to vocabulary that maps onto a concrete component (mirror, flip, TTA, crop, views,
slices, checkpoint, weight, stage).  Nothing is executed and nothing is submitted.

Usage: python work/audit_components.py [--pull]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch' / 'audit'

TARGETS = [
    'kaiwalyaatulraut/rsna-knee',                                    # 0.958
    'degnonguidi/rnsa-best-public-score',                             # 0.957
    'laimerbotrolenk/rsna-four-model-ensemble',                       # 0.957
    'kiyoshiohno/rsna-knee-template',                                 # 0.957
    'yuvrajlamba01/rsna-knee-research-optimized',                     # 0.957
    'bliverrigpdoge/r7-raptor-knee-centred-convnext',                 # 0.957 (private weights)
]

KEYS = re.compile(r'mirror|flip|TTA|tta|centre|crop|view|slice|checkpoint|weight|lifts?|adds?|'
                  r'free|cheap|rank|blend|stacker|state|fluid|side', re.I)
SKIP = re.compile(r'^(#+ |\s*$|\s*[|*-]\s*$)')


def slug_dir(slug: str) -> Path:
    return OUT / slug.replace('/', '__')


def main(argv: list[str]) -> int:
    do_pull = '--pull' in argv
    OUT.mkdir(parents=True, exist_ok=True)
    for slug in TARGETS:
        d = slug_dir(slug)
        if do_pull or not d.is_dir() or not list(d.glob('*.ipynb')):
            d.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(['kaggle', 'kernels', 'pull', slug, '-p', str(d), '-m'],
                               capture_output=True, text=True, encoding='utf-8', errors='replace')
            if r.returncode != 0:
                print('PULL FAILED %-56s %s' % (slug, (r.stdout or r.stderr).strip()[:80]))
                continue
        nb_path = list(d.glob('*.ipynb'))
        if not nb_path:
            print('NO NOTEBOOK %s' % slug)
            continue
        nb = json.loads(Path(nb_path[0]).read_bytes().decode('utf-8-sig', errors='replace'))
        print('\n' + '=' * 100)
        print('== %s   cells=%d' % (slug, len(nb['cells'])))
        shown = 0
        for i, c in enumerate(nb['cells']):
            if c['cell_type'] != 'markdown':
                continue
            s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
            for line in s.splitlines():
                t = line.strip()
                if not t or SKIP.match(t) or len(t) < 18:
                    continue
                if KEYS.search(t):
                    print('   [md %d] %s' % (i, t[:190]))
                    shown += 1
                if shown >= 14:
                    break
            if shown >= 14:
                break
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
