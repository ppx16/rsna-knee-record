"""Archive the machine evidence for every artifact we submitted into evidence/.

Why: every number in FACTS.md is one line of prose I wrote. The receipts are what make those lines checkable
-- a blend receipt records the weights a run actually used, and a leg report records whether an arm ran or
fell back. They are a few kilobytes each and currently live only in scratch/, which is gitignored and would
be lost. Archiving them turns "the S6 arm was folded at 1.0" from an assertion into a file.

Copies receipts and leg reports only (never prediction CSVs, which are large and reproducible).
Writes evidence/<run>/ on each invocation; idempotent.

Usage: python work/archive_receipts.py
"""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'scratch'
DST = ROOT / 'evidence'

# which local output directory corresponds to which artifact, and what it proves
PROVENANCE = {
    'out_0958b': 'the 0.958 base (tick 1 lineage): blend weights + raptor/ours halves',
    'out_m224': '224 arm + anatomical mirror',
    'out_m224p': '224 arm plain/TTA rank split',
    'out_p384': '384 arm plain/TTA rank split (plus the 224 split)',
    'out_s6w60': 'S6 arm folded at 0.60',
    'out_w80': 'S6 arm folded at 0.80 (the weight at which the cost appears)',
    'out_flat': 'per-finding weights flattened to 0.5 (the un-fitted twin)',
    'out_s6solo': 'S6 alone: raptor 0.0, ours 1.0, no per-finding weights (tick 2)',
    'out_09586': 'heliosli 0.9586 fork: flat weights plus the 2.5D MIL reader',
    'out_w10': 'the r9all leg at 0.10',
    'out_b32w08': 'the band32 leg at 0.08',
    'out_w15b': 'the band32 leg at 0.15',
}


def main() -> int:
    DST.mkdir(exist_ok=True)
    index = {}
    for run, what in PROVENANCE.items():
        src = SRC / run
        if not src.is_dir():
            print('%-18s missing locally -- skipped' % run)
            continue
        files = []
        for pat in ('*receipt*.json', '*leg_report.json'):
            files += sorted(src.glob(pat))
        if not files:
            print('%-18s no receipts' % run)
            continue
        out = DST / run
        out.mkdir(exist_ok=True)
        kept = []
        for f in files:
            shutil.copy2(f, out / f.name)
            kept.append({'file': f.name, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()[:16],
                         'bytes': f.stat().st_size})
        index[run] = {'proves': what, 'files': kept}
        print('%-18s %d file(s) archived' % (run, len(kept)))

    man = DST / 'MANIFEST.json'
    man.write_text(json.dumps({'read_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                               'runs': index}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    total = sum(len(v['files']) for v in index.values())
    print('\n%d runs, %d files archived; manifest at %s' % (len(index), total, man.relative_to(ROOT)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
