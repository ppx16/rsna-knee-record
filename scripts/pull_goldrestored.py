import subprocess, json, os, glob, shutil, hashlib

K = 'heliosli/rsna-knee-blend-gold-dataset-restored'
d = os.path.abspath('scratch/top/blend-gold-dataset-restored')
if os.path.isdir(d):
    shutil.rmtree(d, ignore_errors=True)
r = subprocess.run(['kaggle', 'kernels', 'pull', K, '-p', d, '-m'],
                   capture_output=True, text=True, encoding='utf-8', errors='replace')
print('rc', r.returncode, r.stdout[-300:], r.stderr[-300:])
mp = os.path.join(d, 'kernel-metadata.json')
meta = json.load(open(mp, encoding='utf-8'))
print('datasets :', json.dumps(meta.get('dataset_sources'), ensure_ascii=False))
print('docker   :', meta.get('docker_image'))
print('shape    :', meta.get('machine_shape'), 'gpu', meta.get('enable_gpu'))
nbs = glob.glob(os.path.join(d, '*.ipynb'))
print('nb', nbs)
for p in nbs:
    b = open(p, 'rb').read()
    print(p, len(b), hashlib.sha256(b).hexdigest()[:20])
