import subprocess, os, glob

TARGETS = [
    ('heliosli/blend-mine-in-private-lb-luck-to-all-upvoters', 'out_helios956'),
    ('le4ves/rsna-knee-0-954-dual-blend-raptor-convnext-s6', 'out_le4ves954'),
    ('ppxl16/rsna-knee-blendgold-restored', 'out_ours954'),
]

os.makedirs('scratch/runs', exist_ok=True)
for ref, tag in TARGETS:
    d = os.path.abspath('scratch/runs/' + tag)
    r = subprocess.run(['kaggle', 'kernels', 'output', ref, '-p', d],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    print('=' * 90)
    print(ref, 'rc=', r.returncode)
    print((r.stdout or '')[-1200:])
    print((r.stderr or '')[-600:])
    if os.path.isdir(d):
        for p in sorted(glob.glob(d + '/**/*', recursive=True)):
            if os.path.isfile(p):
                print('   %10d  %s' % (os.path.getsize(p), os.path.relpath(p, d)))
