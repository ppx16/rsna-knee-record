"""Split the 224 arm's share between its plain and its mirror-averaged view (costs no forward pass).

Where this comes from (FACTS C197): `laimerbotrolenk/rsna-four-model-ensemble` (0.957) states the rule
verbatim --

    "TTA is the arithmetic mean of the plain and mirrored **probabilities**, then ranked per finding.
     Mixing these ranks is not equivalent to changing the probability average to 75/25."
    their V33 weights: Raptor-384 TTA 32.5% + Raptor-224 **plain** 8.75% + Raptor-224 **TTA** 8.75%

and, decisively:

    "The plain 224 forward already runs in the supplied code. Saving it separately adds
     **no extra forward passes or checkpoints**."

So after C196 gave our 224 arm the mirror, the plain view it already computes is being thrown away.
Rank-mixing the two is a different signal from averaging their probabilities, and it is free.

Four surgical patches, all asserted:
  A. the 224 script keeps a second array for the plain probabilities;
  B. the TTA branch stores the plain result instead of discarding it (one extra assignment, no extra
     forward pass -- `infer_probs(model, xw, d)` was already evaluated, just inlined);
  C. the 224 script dumps `raptor224_plain.csv` next to the raw dump;
  D. the blend cell loads it if it is there and, only then, averages the 224 TTA rank with the 224 plain
     rank BEFORE the author's `(1-A224)*r384 + A224*r224` is applied -- which is exactly "split the 224
     share in two" while leaving the author's expression untouched.

Fail-safe: if the plain file is missing, malformed or has the wrong rows, the blend is byte-identical to
the kernel it was built from (the m224 twin).

Usage: python work/build_0958_m224p.py
"""
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/kernel-0958-m224'

A_INIT = '    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]'
A_NEW = A_INIT + '\n    arm_plain = [np.full((N, len(LAB)), np.nan, np.float32) for _ in range(A)]   # [s9b] the plain half, kept for a separate rank mix'

B_OLD = """                    else:   # [s9] average the plain and the anatomical-mirror view, as the 0.949 model does
                        xm = gpu_windows(vol, mask, K_EVAL, res, d, mirror=True)
                        arm_probs[a][i] = 0.5 * (infer_probs(model, xw, d) + infer_probs(model, xm, d))
                        del xm"""
B_NEW = """                    else:   # [s9] average the plain and the anatomical-mirror view, as the 0.949 model does
                        xm = gpu_windows(vol, mask, K_EVAL, res, d, mirror=True)
                        _p = infer_probs(model, xw, d)   # [s9b] the plain half -- already computed, now kept
                        arm_probs[a][i] = 0.5 * (_p + infer_probs(model, xm, d))
                        arm_plain[a][i] = _p
                        del xm"""

C_OLD = """    _raw.to_csv("/kaggle/working/raptor224_raw.csv", index=False)"""
C_NEW = C_OLD + """
    _pl = pd.DataFrame(np.nan_to_num(arm_plain[0], nan=0.5).astype(np.float32), columns=LAB)   # [s9b]
    _pl.insert(0, "StudyInstanceUID", test_ids)
    _pl.to_csv("/kaggle/working/raptor224_plain.csv", index=False)"""
D1_OLD = "rsub224 = pd.read_csv(WORK / 'raptor224_sub.csv', dtype={'StudyInstanceUID': str})"
D1_NEW = D1_OLD + """
# [s9b] the 224 plain view, if the stage kept it: rank-mix it with the TTA view at half the 224 share.
plain224 = None
try:
    _pl = pd.read_csv(WORK / 'raptor224_plain.csv', dtype={'StudyInstanceUID': str})
    if set(labels) <= set(_pl.columns) and sorted(_pl['StudyInstanceUID']) == sorted(test_ids):
        plain224 = _pl.set_index('StudyInstanceUID').loc[test_ids, labels]
        print('[s9b] 224 plain view loaded', flush=True)
    else:
        print('[s9b] 224 plain columns/rows mismatch -> not used', flush=True)
except Exception as _e:
    print('[s9b] 224 plain not available -> not used: %r' % (_e,), flush=True)"""

D2_OLD = '    rr = np.where(r384_fail, r224, np.where(r224_fail, r384, (1.0 - A224) * r384 + A224 * r224))   # [d957] NaN only where both failed'
D2_NEW = """    if plain224 is not None:   # [s9b] split the 224 share: TTA and plain are different signals
        _r224p = pct_on(plain224[c].values, ~r224_fail)
        _mp = np.isfinite(r224) & np.isfinite(_r224p)
        r224 = np.where(_mp, 0.5 * np.nan_to_num(r224) + 0.5 * np.nan_to_num(_r224p), r224)
""" + D2_OLD


def text_of(c):
    return c['source'] if isinstance(c['source'], str) else ''.join(c['source'])


def set_text(c, s):
    c['source'] = s if isinstance(c['source'], str) else s.splitlines(keepends=True)


def sub_once(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit('REFUSING: %s matched %d times (expected 1)' % (what, n))
    out = text.replace(old, new, 1)
    if new not in out:
        raise SystemExit('REFUSING: %s did not land' % what)
    return out


def main() -> int:
    base_nb = next(BASE.glob('*.ipynb'))
    meta0 = json.loads((BASE / 'kernel-metadata.json').read_text(encoding='utf-8'))
    nb0 = json.loads(base_nb.read_bytes().decode('utf-8-sig'))
    if len(nb0['cells']) != 9:
        raise SystemExit('REFUSING: the base has %d cells, expected 9' % len(nb0['cells']))

    s224 = text_of(nb0['cells'][4])
    old224 = s224
    s224 = sub_once(s224, A_INIT, A_NEW, 'patch A (plain array)')
    s224 = sub_once(s224, B_OLD, B_NEW, 'patch B (keep the plain probs)')
    s224 = sub_once(s224, C_OLD, C_NEW, 'patch C (dump the plain csv)')

    sblend = text_of(nb0['cells'][8])
    oldblend = sblend
    sblend = sub_once(sblend, D1_OLD, D1_NEW, 'patch D1 (load the plain csv)')
    sblend = sub_once(sblend, D2_OLD, D2_NEW, 'patch D2 (split the 224 share)')

    nb = copy.deepcopy(nb0)
    set_text(nb['cells'][4], s224)
    set_text(nb['cells'][8], sblend)
    changed = [i for i in range(9) if text_of(nb['cells'][i]) != text_of(nb0['cells'][i])]
    if changed != [4, 8]:
        raise SystemExit('REFUSING: cells changed = %s, expected [4, 8]' % changed)

    # the 224 script and the blend cell must still be valid Python on their own
    compile(s224.split('\n', 1)[1], 'raptor224_run', 'exec')
    compile(sblend, 'blend', 'exec')

    slug = 'rsna-knee-0958-m224p'
    title = 'RSNA Knee 0958 m224p'
    if re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', title.lower())).strip('-') != slug:
        raise SystemExit('REFUSING: title does not slugify to the id')
    meta = copy.deepcopy(meta0)
    meta.pop('id_no', None)
    meta['id'] = 'ppxl16/' + slug
    meta['title'] = title
    meta['code_file'] = slug + '.ipynb'
    meta['is_private'] = True
    for k, v in meta0.items():
        if k in ('id', 'title', 'code_file', 'is_private', 'id_no'):
            continue
        if json.dumps(meta.get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
            raise SystemExit('REFUSING: environment field %s changed' % k)
    out = json.dumps(nb, ensure_ascii=True)
    if json.loads(out) != nb or not out.isascii():
        raise SystemExit('REFUSING: ASCII round-trip failed')
    try:
        out.encode('utf-8').decode('gbk')
    except UnicodeDecodeError as exc:
        raise SystemExit('REFUSING: not GBK-decodable: %s' % exc)
    dst = ROOT / 'kernels' / 'kernel-0958-m224p'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / meta['code_file']).write_text(out, encoding='utf-8')
    (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('224 plain+TTA split: %s  (%d B, sha %s)' % (dst.name, len(out),
                                                       hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('  224 script  %d -> %d chars' % (len(old224), len(s224)))
    print('  blend cell  %d -> %d chars' % (len(oldblend), len(sblend)))
    print('audit: exactly cells 4 and 8 changed; both still compile; env fields untouched')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
