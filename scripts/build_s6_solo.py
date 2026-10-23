"""Build an S6-ONLY artifact: nartaa's all-data ConvNeXt and nothing else.

Why this exists (FACTS C211): our tick 2 is a 0.944 artifact -- our own non-OAI stack -- and it exists
purely as insurance in case the organisers rule the OAI-lineage Raptor models unusable.  But nartaa's S6
all-data model is documented as **0.950 alone** and explicitly built with "No OAI data was used"
(our own submission 57291788 says so in its description).  We have never submitted S6 on its own, so the
insurance floor has been sitting 0.006 below what it could be.

The build takes the existing S6 kernel and changes exactly two things, both audited:
  * the S6 share of the compact half: 0.40 -> 1.00  (so the compact half IS S6);
  * the blend: W_RAPTOR 0.50 -> 0.00 and W_RAPTOR_C -> {} (so the Raptor half contributes nothing).
Everything else -- the S6 stage, its anchor checkpoint, the fail-safes -- is untouched.  If the S6 stage
fails, the artifact keeps the 0.958 submission rather than degrading, which is the existing behaviour.

Usage: python work/build_s6_solo.py
"""
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/kernel-0958-s6-s6w40'

W_OLD = 'W_RAPTOR, W_OURS = 0.5, 0.5'
W_NEW = 'W_RAPTOR, W_OURS = 0.0, 1.0   # [solo] S6 alone: the Raptor half contributes nothing'
C_OLD = re.compile(r"(?m)^W_RAPTOR_C = \{.*$")
C_NEW = "W_RAPTOR_C = {}   # [solo] no per-finding Raptor weight: this artifact is S6 only"
S6_OLD = re.compile(r"(?m)^(\s*)ours = \(0\.6 \* ours\.rank\(pct=True\) \+ 0\.4 \* _s6f\.rank\(pct=True\)\)$")
S6_NEW_TMPL = r'\g<1>ours = _s6f.rank(pct=True)   # [solo] the compact half IS S6, no R6 share'
MSG_OLD = "[s6] folded into the compact half at 0.4 over %d studies"
MSG_NEW = "[s6] folded into the compact half at 1.0 (S6 only) over %d studies"


def text_of(c):
    return c['source'] if isinstance(c['source'], str) else ''.join(c['source'])


def set_text(c, s):
    c['source'] = s if isinstance(c['source'], str) else s.splitlines(keepends=True)


def main() -> int:
    nb_path = next(BASE.glob('*.ipynb'))
    meta0 = json.loads((BASE / 'kernel-metadata.json').read_text(encoding='utf-8'))
    nb0 = json.loads(nb_path.read_bytes().decode('utf-8-sig'))

    # locate the cells
    w_cells = [i for i, c in enumerate(nb0['cells']) if W_OLD in text_of(c)]
    c_cells = [i for i, c in enumerate(nb0['cells']) if C_OLD.search(text_of(c))]
    s6_cells = [i for i, c in enumerate(nb0['cells']) if S6_OLD.search(text_of(c))]
    if len(w_cells) != 1 or len(c_cells) != 1 or len(s6_cells) != 1:
        raise SystemExit('REFUSING: anchor counts W=%s C=%s S6=%s (each must be 1)'
                         % (w_cells, c_cells, s6_cells))
    wi, ci, si = w_cells[0], c_cells[0], s6_cells[0]
    print('anchors: W_RAPTOR in cell %d, W_RAPTOR_C in cell %d, S6 weight in cell %d' % (wi, ci, si))

    nb = copy.deepcopy(nb0)
    sw = text_of(nb['cells'][si])
    sw2 = S6_OLD.sub(S6_NEW_TMPL, sw, count=1)
    if sw2 == sw:
        raise SystemExit('REFUSING: the S6 weight substitution changed nothing')
    set_text(nb['cells'][si], sw2)

    bw = text_of(nb['cells'][wi])
    if bw.count(W_OLD) != 1:
        raise SystemExit('REFUSING: W_RAPTOR line count %d' % bw.count(W_OLD))
    bw2 = bw.replace(W_OLD, W_NEW, 1)
    bc = C_OLD.sub(C_NEW, bw2, count=1)
    if bc.count(C_NEW) != 1:
        raise SystemExit('REFUSING: the W_RAPTOR_C substitution did not land')
    # the log line must tell the truth about the weight, or the run cannot be verified from its output
    if bc.count(MSG_OLD) != 1:
        raise SystemExit('REFUSING: the fold message anchor count %d' % bc.count(MSG_OLD))
    bc = bc.replace(MSG_OLD, MSG_NEW, 1)
    set_text(nb['cells'][wi], bc)

    changed = [i for i in range(len(nb0['cells'])) if text_of(nb['cells'][i]) != text_of(nb0['cells'][i])]
    if sorted(set(changed)) != sorted({si, wi}):
        raise SystemExit('REFUSING: cells changed = %s, expected exactly %s' % (changed, sorted({si, wi})))
    # the fold message must still exist so the run can be verified, and must now say 1.0
    if MSG_NEW not in text_of(nb['cells'][wi]):
        raise SystemExit('REFUSING: the [s6] fold log line does not report the new weight')
    for i in (si, wi):
        try:
            compile(text_of(nb['cells'][i]), 'cell%d' % i, 'exec')
        except SyntaxError as exc:
            raise SystemExit('REFUSING: cell %d no longer compiles: %s' % (i, exc))

    slug = 'rsna-knee-s6-solo'
    title = 'RSNA Knee s6 solo'
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
    dst = ROOT / 'kernels' / 'kernel-s6-solo'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / meta['code_file']).write_text(out, encoding='utf-8')
    (dst / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('S6-only artifact -> %s  (%d B, sha %s)' % (dst.name, len(out),
                                                      hashlib.sha256(out.encode()).hexdigest()[:16]))
    print('  S6 weight 0.4 -> 1.0 ; W_RAPTOR 0.5 -> 0.0 ; W_RAPTOR_C -> {} ')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
