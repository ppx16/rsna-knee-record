import subprocess, json, os, glob, shutil

KERNELS = [
    'heliosli/blend-mine-in-private-lb-luck-to-all-upvoters',
    'bliverrigpdoge/velci-kcd25-20',
    'matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed',
    'evgendvorkin/rsna-versia-7',
]

base = os.path.abspath('scratch/top')
os.makedirs(base, exist_ok=True)
fh = open('scratch/top_report.txt', 'w', encoding='utf-8')

for k in KERNELS:
    slug = k.split('/')[-1]
    d = os.path.join(base, slug)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    r = subprocess.run(['kaggle', 'kernels', 'pull', k, '-p', d, '-m'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    fh.write('\n' + '#' * 100 + '\n# %s (rc=%s)\n' % (k, r.returncode) + '#' * 100 + '\n')
    if r.returncode != 0:
        fh.write(r.stdout + r.stderr + '\n')
        continue
    mp = os.path.join(d, 'kernel-metadata.json')
    meta = json.load(open(mp, encoding='utf-8')) if os.path.exists(mp) else {}
    fh.write('title    : %s\n' % meta.get('title'))
    fh.write('datasets : %s\n' % json.dumps(meta.get('dataset_sources'), ensure_ascii=False))
    fh.write('ksrc     : %s\n' % json.dumps(meta.get('kernel_sources'), ensure_ascii=False))
    fh.write('models   : %s\n' % json.dumps(meta.get('model_sources'), ensure_ascii=False))
    fh.write('docker   : %s\n' % meta.get('docker_image'))
    fh.write('shape    : %s\n' % meta.get('machine_shape'))
    nbs = glob.glob(os.path.join(d, '*.ipynb'))
    fh.write('notebook : %s  (%d bytes)\n' % ([os.path.basename(x) for x in nbs], os.path.getsize(nbs[0]) if nbs else 0))
    if nbs:
        j = json.load(open(nbs[0], encoding='utf-8'))
        for i, c in enumerate(j.get('cells', [])):
            if c.get('cell_type') == 'markdown':
                fh.write('\n--- md cell %d ---\n' % i)
                fh.write(''.join(c.get('source', [])) + '\n')
fh.close()
print('wrote scratch/top_report.txt', os.path.getsize('scratch/top_report.txt'))
