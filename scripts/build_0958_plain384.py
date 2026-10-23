"""Split the 384 arm's share too: its plain half is computed and discarded exactly like the 224 one was.

FACTS C197 established the rule from `laimerbotrolenk/rsna-four-model-ensemble` (0.957): TTA is the mean
of the plain and mirrored PROBABILITIES, and rank-mixing plain with that mean is a different signal, so
their V33 gives the 224 arm two separate slots (plain 8.75 % + TTA 8.75 %).

Their table also shows the 384 arm carrying 32.5 % of the final -- and our 384 arm has the same waste:

    def infer_pair_probs(model, plain, mirrored, device):
        p = infer_probs(model, plain, device)
        q = infer_probs(model, mirrored, device)
        return 0.5 * (p + q)          # <- p is thrown away

So the same split applies to the arm that carries four times the 224 arm's weight (0.65 of the Raptor
half vs 0.35).  Nothing is re-trained and no extra forward pass is added: `p` is already computed.

Patches (each asserted to match exactly once):
  A. `infer_pair_probs` returns `(average, plain)`;
  B. the single call site stores both;
  C. the 384 script keeps an `arm_plain` array and dumps `raptor_plain.csv`;
  D. the blend loads it if present and averages the 384 TTA rank with the 384 plain rank BEFORE the
     author's `(1-A224)*r384 + A224*r224`, i.e. it splits the 384 share in half without touching the
     author's expression.

Fail-safe: a missing/invalid plain file leaves the blend byte-identical to the m224p kernel.
Usage: python work/build_0958_plain384.py
"""
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/kernel-0958-m224p'

FN_OLD = """    p = infer_probs(model, plain, device)
    q = infer_probs(model, mirrored, device)
    return 0.5 * (p + q)"""
FN_NEW = """    p = infer_probs(model, plain, device)
    q = infer_probs(model, mirrored, device)
    return 0.5 * (p + q), p          # [s9c] the plain half is already computed -- hand it back"""

CALL_OLD = '                    arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)'
CALL_NEW = """                    _avg, _pl = infer_pair_probs(model, xw, xm, d)   # [s9c]
                    arm_probs[a][i] = _avg
                    arm_plain[a][i] = _pl"""

ARR_OLD = '    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]'
ARR_NEW = ARR_OLD + '\n    arm_plain = [np.full((N, len(LAB)), np.nan, np.float32) for _ in range(A)]   # [s9c] the 384 plain half'

DUMP_OLD = '    _raw.to_csv("/kaggle/working/raptor_raw.csv", index=False)'
DUMP_NEW = DUMP_OLD + """
    _pl = pd.DataFrame(np.nan_to_num(arm_plain[0], nan=0.5).astype(np.float32), columns=LAB)   # [s9c]
    _pl.insert(0, "StudyInstanceUID", test_ids)
    _pl.to_csv("/kaggle/working/raptor_plain.csv", index=False)"""

LOAD_OLD = "rsub = pd.read_csv(WORK / 'raptor_sub.csv', dtype={'StudyInstanceUID': str})"
LOAD_NEW = LOAD_OLD + """
# [s9c] the 384 plain view, if the stage kept it: rank-mix it with the TTA view at half the 384 share.
plain384 = None
try:
    _pl = pd.read_csv(WORK / 'raptor_plain.csv', dtype={'StudyInstanceUID': str})
    if set(labels) <= set(_pl.columns) and sorted(_pl['StudyInstanceUID']) == sorted(test_ids):
        plain384 = _pl.set_index('StudyInstanceUID').loc[test_ids, labels]
        print('[s9c] 384 plain view loaded', flush=True)
    else:
        print('[s9c] 384 plain columns/rows mismatch -> not used', flush=True)
except Exception as _e:
    print('[s9c] 384 plain not available -> not used: %r' % (_e,), flush=True)"""

MIX_OLD = '    rr = np.where(r384_fail, r224, np.where(r224_fail, r384, (1.0 - A224) * r384 + A224 * r224))   # [d957] NaN only where both failed'
MIX_NEW = """    if plain384 is not None:   # [s9c] split the 384 share: TTA and plain are different signals
        _r384p = pct_on(plain384[c].values, ~r384_fail)
        _m4 = np.isfinite(r384) & np.isfinite(_r384p)
        r384 = np.where(_m4, 0.5 * np.nan_to_num(r384) + 0.5 * np.nan_to_num(_r384p), r384)
""" + MIX_OLD


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

    s384 = text_of(nb0['cells'][2])
    old384 = s384
    s384 = sub_once(s384, FN_OLD, FN_NEW, 'A (return the plain half)')
    s384 = sub_once(s384, CALL_OLD, CALL_NEW, 'B (store it at the call site)')
    s384 = sub_once(s384, ARR_OLD, ARR_NEW, 'C1 (arm_plain array)')
    s384 = sub_once(s384, DUMP_OLD, DUMP_NEW, 'C2 (dump raptor_plain.csv)')

    sblend = text_of(nb0['cells'][8])
    oldblend = sblend
    sblend = sub_once(sblend, LOAD_OLD, LOAD_NEW, 'D1 (load the 384 plain csv)')
    sblend = sub_once(sblend, MIX_OLD, MIX_NEW, 'D2 (split the 384 share)')

    nb = copy.deepcopy(nb0)
    set_text(nb['cells'][2], s384)
    set_text(nb['cells'][8], sblend)
    changed = [i for i in range(9) if text_of(nb['cells'][i]) != text_of(nb0['cells'][i])]
    if changed != [2, 8]:
        raise SystemExit('REFUSING: cells changed = %s, expected [2, 8]' % changed)

    compile(s384.split('\n', 1)[1], 'raptor_run', 'exec')
    compile(sblend, 'blend', 'exec')

    slug = 'rsna-knee-0958-p384'
    title = 'RSNA Knee 0958 p384'
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
    dst = ROOT / 'kernels' / 'kernel-0958-p384'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / meta['code_file']).write_text(out, encoding='utf-8')
    (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('384 plain+TTA split: %s  (%d B, sha %s)' % (dst.name, len(out),
                                                      hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('  384 script %d -> %d chars ; blend %d -> %d chars'
          % (len(old384), len(s384), len(oldblend), len(sblend)))
    print('audit: exactly cells 2 and 8 changed; both still compile; env fields untouched')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
