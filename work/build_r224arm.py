"""Build the r224-arm variant of our r7 recipe: add nartaa's SECOND checkpoint as a second Raptor arm.

The arm dict is copied verbatim from rabari9999's dual-engine notebook (FACTS C163) -- NOT guessed:
    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt",
     "arch": "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", "res": 224, "w": 1.0}
and that notebook fuses it into the CoAtNet leg at weight 0.3 in rank space:
    W_C224 = 0.3 ; coat = ((1-W_C224)*coat + W_C224*legs["c224"]).rank(pct=True)
Our ARMS mechanism is the same thing (weighted rank-mean of the arms), so 0.7/0.3 reproduces it.

Everything else is asserted unchanged: the S6 stage, the 0.5/0.5 stage-3 blend, every environment
field, and the notebook bytes outside this one block.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'kernels/kernel-r7-onclick'
DST = ROOT / 'kernels/kernel-r7-r224arm'
NEW_ID = 'ppxl16/rsna-knee-r7-r224arm'
NEW_TITLE = 'RSNA Knee r7 r224arm'
NEW_CODE = 'rsna-knee-r7-r224arm.ipynb'

OLD_BLOCK = (
    'ARMS = [   # our 5-fold CoAtNet (public recipe reproduced on our folds/labels), equal-weight rank-mean\n'
    '    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", '
    '"arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},\n'
    ']'
)
NEW_BLOCK = (
    'ARMS = [   # two CoAtNet arms: the 384 checkpoint and nartaa\'s r224 checkpoint (multi-scale);\n'
    '    # per-arm weights are normalised downstream, so 0.7/0.3 reproduces rabari9999\'s fusion.\n'
    '    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", '
    '"arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.7},\n'
    '    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt", '
    '"arch": "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", "res": 224, "w": 0.3},\n'
    ']'
)


def main() -> int:
    src_nb = next(SRC.glob('*.ipynb'))
    raw = src_nb.read_bytes()
    text = raw.decode('utf-8')
    print('base : %s  %d bytes  sha %s' % (src_nb.name, len(raw), hashlib.sha256(raw).hexdigest()[:16]))

    # The notebook is JSON: inside it the source lines are ESCAPED ("\n", "\""), so a block written
    # with real newlines can never match the file bytes.  Build the escaped form with json.dumps.
    esc_old = json.dumps(OLD_BLOCK)[1:-1]
    esc_new = json.dumps(NEW_BLOCK)[1:-1]

    n = text.count(esc_old)
    if n != 1:
        raise SystemExit('REFUSING: the ARMS block matched %d times (expected exactly 1)' % n)
    fixed = text.replace(esc_old, esc_new)

    # audit, linearly: fixed must be text with exactly that one block swapped
    if esc_new.join(text.split(esc_old)) != fixed:
        raise SystemExit('REFUSING: not exactly this one block swap')
    if esc_old in fixed or 'c08_r224_swa' not in fixed:
        raise SystemExit('REFUSING: post-swap content check failed')
    if fixed.count('c08_r224_swa') != 1:
        raise SystemExit('REFUSING: r224 arm appears %d times' % fixed.count('c08_r224_swa'))
    # everything downstream of ARMS must be untouched
    tail_old = text[text.index(esc_old) + len(esc_old):]
    tail_new = fixed[fixed.index(esc_new) + len(esc_new):]
    if tail_old != tail_new:
        raise SystemExit('REFUSING: something after ARMS changed')
    print('swapped 1 ARMS block; %d bytes -> %d bytes; tail after ARMS identical'
          % (len(raw), len(fixed.encode('utf-8'))))

    out_nb = fixed.encode('utf-8')
    try:
        out_nb.decode('gbk')
    except UnicodeDecodeError as exc:
        raise SystemExit('REFUSING: not GBK-decodable: %s' % exc)

    meta = json.loads((SRC / 'kernel-metadata.json').read_text(encoding='utf-8'))
    env = ('dataset_sources', 'kernel_sources', 'model_sources', 'competition_sources',
           'docker_image', 'machine_shape', 'enable_gpu', 'enable_tpu', 'enable_internet',
           'language', 'kernel_type', 'keywords')
    before = {k: json.dumps(meta.get(k), sort_keys=True) for k in env}
    meta.pop('id_no', None)
    meta['id'] = NEW_ID
    meta['title'] = NEW_TITLE
    meta['code_file'] = NEW_CODE
    meta['is_private'] = True
    for k in env:
        if json.dumps(meta.get(k), sort_keys=True) != before[k]:
            raise SystemExit('REFUSING: environment field %s changed' % k)
    print('environment inherited unchanged (%d fields)' % len(env))

    DST.mkdir(parents=True, exist_ok=True)
    (DST / NEW_CODE).write_bytes(out_nb)
    (DST / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('wrote %s\n  notebook %d bytes  sha %s' % (DST, len(out_nb), hashlib.sha256(out_nb).hexdigest()))
    print('  NOT PUSHED yet')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
