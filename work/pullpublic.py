import subprocess, json, os, glob, shutil

KERNELS = [
    'varshithakolupuri/rsna-knee-real-scores-of-every-public-notebook',
    'dmitriykolmikov/rsna-knee-is-0-001-on-public-lb-real',
    'romantamrazov/rsna-knee-dinosaur-the-last',
    'goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954',
    'sujanmajhisuzan/rsna-knee-apex-grandmaster-v10-tri-engine-probit',
    'shuheitak/rsna-knee-probit954-fork',
    'goodpjw2008/rsna-knee-coatnet-s6-2-5d-mil-lb-0-954',
]

base = os.path.abspath('scratch/public')
os.makedirs(base, exist_ok=True)
report = []

for k in KERNELS:
    slug = k.split('/')[-1]
    d = os.path.join(base, slug)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    r = subprocess.run(['kaggle', 'kernels', 'pull', k, '-p', d, '-m'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    meta = {}
    mp = os.path.join(d, 'kernel-metadata.json')
    if os.path.exists(mp):
        try:
            meta = json.load(open(mp, encoding='utf-8'))
        except Exception as e:
            meta = {'ERR': repr(e)}
    nb = [p for p in glob.glob(os.path.join(d, '*.ipynb'))]
    md_cells = []
    if nb:
        try:
            j = json.load(open(nb[0], encoding='utf-8'))
            for c in j.get('cells', []):
                if c.get('cell_type') == 'markdown':
                    md_cells.append(''.join(c.get('source', [])))
        except Exception as e:
            md_cells = ['ERR ' + repr(e)]
    report.append((k, r.returncode, meta, md_cells, nb))

with open('scratch/publicscan_report.txt', 'w', encoding='utf-8') as fh:
    for k, rc, meta, md, nb in report:
        fh.write('\n' + '#' * 100 + '\n')
        fh.write('# %s   (rc=%s)  notebook=%s\n' % (k, rc, os.path.basename(nb[0]) if nb else 'NONE'))
        fh.write('#' * 100 + '\n')
        fh.write('title       : %s\n' % meta.get('title'))
        fh.write('datasets    : %s\n' % (meta.get('dataset_sources')))
        fh.write('kernel_src  : %s\n' % (meta.get('kernel_sources')))
        fh.write('models      : %s\n' % (meta.get('model_sources')))
        fh.write('comp        : %s\n' % (meta.get('competition_sources')))
        fh.write('--- markdown ---\n')
        for m in md:
            fh.write(m + '\n')
print('wrote scratch/publicscan_report.txt')
