"""Estimate whether the notebook's internal 8.3 h deadline binds on the real test set.

Why this matters: the notebook degrades GRACEFULLY -- a study the S6 stage did not reach before
the deadline takes *Raptor's* rank (Raptor alone is 0.949).  So if the deadline binds, the public
score sinks toward 0.949 and the fix is a faster machine_shape (FACTS C144, C151 area).

The Save run scores only the 3 studies in the sample test.csv, but it reports the per-study cost,
so the arithmetic is: seconds_per_study * n_test_studies / n_workers + the Raptor stage.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# the real hidden test set size: the competition's public sample is a stub, but the
# leaderboard/description implies roughly 1300 studies (the noise analysis notebook assumed 1300,
# and the sibling "is +0.001 real?" notebook used 1300 with a 30/70 public/private split).
N_TEST_CANDIDATES = [900, 1100, 1300, 1500]

for tag in ('v1ds', 'r7', 'out_fork'):
    p = ROOT / 'scratch/runs' / tag / 'submission_receipt.json'
    if not p.is_file():
        continue
    d = json.loads(p.read_text(encoding='utf-8'))
    print('=' * 84)
    print('%s   status=%s rows=%s scored=%s fallbacks=%s'
          % (tag, d.get('status'), d.get('rows'), d.get('scored'), d.get('fallback_studies')))
    print('  seconds(whole controller) %s' % d.get('seconds'))
    print('  per_study_mean           %s' % json.dumps(d.get('per_study_mean')))
    print('  worker exit codes        %s' % d.get('worker_exit_codes'))
    for w in (d.get('workers') or []):
        st = w.get('stats') or {}
        print('  worker %s: studies=%s scored=%s gpu_s=%s windows=%s decode_s=%s'
              % (w.get('worker', '?'), w.get('studies'), w.get('scored'),
                 st.get('gpu_s'), st.get('windows'), st.get('decode_s')))
        ps = w.get('per_study') or []
        print('    first per-study rows [uid, windows, decode_s, render_s, gpu_s]: %s' % ps[:3])

    # the arithmetic, with the loading time removed as well as possible
    ws = d.get('workers') or []
    n_workers = max(1, len(ws))
    gpu_total = sum(float((w.get('stats') or {}).get('gpu_s', 0.0)) for w in ws)
    scored = max(1, int(d.get('scored') or 1))
    per_study_gpu = gpu_total / scored
    print('  => measured: %d workers, %.1f s of GPU work over %d studies = %.2f s/study/worker'
          % (n_workers, gpu_total, scored, per_study_gpu))
    for n in N_TEST_CANDIDATES:
        est = per_study_gpu * n / n_workers
        print('     if the hidden test has %5d studies -> S6 stage %6.0f s = %5.2f h  %s'
              % (n, est, est / 3600.0,
                 'BINDS (>8.3 h)' if est > 8.3 * 3600 else 'fits'))
