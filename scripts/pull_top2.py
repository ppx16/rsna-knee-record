import subprocess, json, os, glob, shutil, hashlib

KERNELS = [
    'heliosli/0-957-latest-dataset-score-in-one-click',
    'heliosli/rsna-knee-blend-gold-old-dataset',
]

base = os.path.abspath('scratch/top2')
os.makedirs(base, exist_ok=True)
fh = open('scratch/top2_report.txt', 'w', encoding='utf-8')

for k in KERNELS:
    slug = k.split('/')[-1]
    d = os.path.join(base, slug)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    r = subprocess.run(['kaggle', 'kernels', 'pull', k, '-p', d, '-m'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    fh.write('\n' + '#' * 96 + '\n# %s (rc=%s)\n' % (k, r.returncode) + '#' * 96 + '\n')
    print('=' * 90)
    print(k, 'rc=', r.returncode, (r.stderr or '')[-200:])
    if r.returncode != 0:
        fh.write((r.stdout or '') + (r.stderr or '') + '\n')
        continue
    meta = json.load(open(os.path.join(d, 'kernel-metadata.json'), encoding='utf-8'))
    nbs = glob.glob(os.path.join(d, '*.ipynb'))
    sha = ''
    if nbs:
        b = open(nbs[0], 'rb').read()
        sha = hashlib.sha256(b).hexdigest()
        fh.write('notebook : %s  %d bytes  sha256 %s\n' % (os.path.basename(nbs[0]), len(b), sha))
    for key in ('title', 'dataset_sources', 'kernel_sources', 'model_sources', 'docker_image',
                'machine_shape', 'enable_gpu', 'enable_internet'):
        v = meta.get(key)
        fh.write('%-17s: %s\n' % (key, json.dumps(v, ensure_ascii=False)))
        print('  %-17s %s' % (key, json.dumps(v, ensure_ascii=False)[:150]))
    if nbs:
        j = json.load(open(nbs[0], encoding='utf-8'))
        txt = '\n'.join(''.join(c.get('source', [])) for c in j.get('cells', []))
        for kw in ('W_RAPTOR, W_OURS', 'C_SLUG', 'BUNDLE_ID', 'DEADLINE_HOURS', 'compact_train.py',
                   'r6cnxfsss', 'compact-v1', 'blend-gold'):
            hits = [l.strip()[:170] for l in txt.splitlines() if kw in l]
            if hits:
                fh.write('  [%s] %s\n' % (kw, hits[0]))
        fh.write('--- markdown ---\n')
        for c in j.get('cells', []):
            if c.get('cell_type') == 'markdown':
                fh.write(''.join(c.get('source', [])) + '\n')

fh.close()
print('\nwrote scratch/top2_report.txt')
