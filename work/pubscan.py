import subprocess, json, sys, os, csv, io

def run(args):
    r = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return (r.stdout or '') + (r.stderr or '')

out = []
out.append('#' * 90)
out.append('# A. datasets matching "rsna knee", newest first')
out.append('#' * 90)
for q in ['rsna knee', 'rsna-knee', 'knee abnormality', 'raptor knee', 'knee mri']:
    t = run(['kaggle', 'datasets', 'list', '-s', q, '--sort-by', 'updated', '--page-size', '40', '--csv'])
    out.append('\n===== query: %s =====' % q)
    for line in t.splitlines():
        if not line.strip() or line.startswith('Warning') or 'outdated' in line:
            continue
        out.append(line.strip())

out.append('\n' + '#' * 90)
out.append('# B. datasets by voteCount (the ones people actually use)')
out.append('#' * 90)
for q in ['rsna knee']:
    t = run(['kaggle', 'datasets', 'list', '-s', q, '--sort-by', 'voteCount', '--page-size', '40', '--csv'])
    out.append('\n===== query: %s =====' % q)
    for line in t.splitlines():
        if not line.strip() or line.startswith('Warning') or 'outdated' in line:
            continue
        out.append(line.strip())

out.append('\n' + '#' * 90)
out.append('# C. kernels for the competition, by voteCount')
out.append('#' * 90)
for sb in ['voteCount', 'dateRun', 'scoreDescending']:
    t = run(['kaggle', 'kernels', 'list', '--competition', 'rsna-knee-abnormality-detection',
             '--sort-by', sb, '--page-size', '50', '--csv'])
    out.append('\n===== sort: %s =====' % sb)
    for line in t.splitlines():
        if not line.strip() or line.startswith('Warning') or 'outdated' in line:
            continue
        out.append(line.strip())

open('scratch/pubscan.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('wrote scratch/pubscan.txt', os.path.getsize('scratch/pubscan.txt'))
