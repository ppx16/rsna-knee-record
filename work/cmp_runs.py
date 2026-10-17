import hashlib, json, os

rows = [
    ('helios956', 'scratch/runs/out_helios956'),
    ('le4ves954', 'scratch/runs/out_le4ves954'),
    ('ours954', 'scratch/runs/out_ours954'),
]

for tag, d in rows:
    print('=' * 80)
    print(tag, d)
    for f in ('submission.csv', 'ours_sub.csv', 'raptor_sub.csv', 'raptor_raw.csv',
              'blend_receipt.json', 'submission_receipt.json'):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            print('  %-26s MISSING' % f)
            continue
        b = open(p, 'rb').read()
        print('  %-26s %7d  md5=%s' % (f, len(b), hashlib.md5(b).hexdigest()[:16]))

print()
print('=== blend_receipt.json contents ===')
for tag, d in rows:
    p = os.path.join(d, 'blend_receipt.json')
    if os.path.exists(p):
        print('---', tag)
        print(open(p, encoding='utf-8').read()[:900])

print()
print('=== submission.csv head ===')
for tag, d in rows:
    p = os.path.join(d, 'submission.csv')
    if os.path.exists(p):
        t = open(p, encoding='utf-8').read().splitlines()
        print('---', tag, 'lines', len(t))
        for l in t[:3]:
            print('   ', l[:160])
