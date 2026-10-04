"""Build `kernel-0943cf448`: our 0.943 stack (`0943cf`) plus the author's PUBLIC
one-fold M448 ConvNeXt member, blended per-label on RANKS.

WHY THIS ARM EXISTS
    The public frontier is capped at 0.943 and two independent notebooks show the
    0.946 variants get there by blending a PRIVATE ConvNeXt/M448 member at ~10%.
    That member's own `package_receipt.json` says the public dataset is
        "one-fold fallback when the private five-fold M448 output is unavailable"
    -- i.e. it is published precisely so this blend can be attempted. Nothing in the
    0943 stack mounts it, so we have never been running the member at all.

WHY THE BASE IS `0943cf` AND NOT `0943cfm`
    `0943cfm` adds MCL to `_RAD_EXCLUDE` (the Rad head stops contributing to the MCL
    label). The member predicts all 12 labels, MCL included. Blending it in would
    partly undo that correction, and the two effects are not separable on this board.
    Keep them in different arms.

WHY RANKS
    The metric is macro ROC-AUC over 12 binary findings: only the ordering within a
    label column counts. Blending rank(pct) is therefore the score-appropriate
    operation and matches the frontier's own "rank-of-mean-probabilities" reduction.
    Blending raw sigmoid probabilities would let the member's calibration move the
    ordering, which is not what we want from a member.

SAFETY RAIL
    The member is fail-CLOSED: any exception leaves the base submission.csv untouched,
    so the arm degrades to `0943cf` rather than to a broken file. Every guard the
    member ships with (SHA-256, fold-0 primary, cfg match, schema drift) is kept.

Usage:
    python scripts/build_m448_arm.py                 # write files
    python scripts/build_m448_arm.py --push          # write + push to Kaggle
"""
from __future__ import annotations

import json
import pathlib
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels" / "kernel-0943cf"
DST = ROOT / "kernels" / "kernel-0943cf448"

SLUG = "rsna-knee-0943cf448"
TITLE = "RSNA knee 0943cf448"
MEMBER_DATASET = "mattiaangeli/rsna-knee-cnx-m448-f0-public"
MEMBER_SHA = "3b7f50c7bd16d353b8d79bf6089530a376009b6c3480e1bd97b778aa1f4ecab8"
MEMBER_WEIGHT = 0.10          # the weight the frontier uses for the 5-fold original

CELL_SOURCE = f'''# ---------------------------------------------------------------------------
# M448 member: the author's PUBLIC one-fold fallback for the private member that
# carries the 0.943 -> 0.946 variants. Blended on RANKS, per label.
#
# FAIL-CLOSED: on any error the base submission.csv is left exactly as written, so
# this arm degrades to `0943cf` and never to a broken file.
# ---------------------------------------------------------------------------
import glob as _g, sys as _sys, time as _t, traceback as _tb

M448_W = {MEMBER_WEIGHT}
M448_SHA = "{MEMBER_SHA}"
_SUB = "/kaggle/working/submission.csv"

try:
    _hits = [p for p in _g.glob("/kaggle/input/**/cnx_montage448_f0.pt", recursive=True)]
    if len(_hits) != 1:
        raise RuntimeError("expected exactly one cnx_montage448_f0.pt, found %r" % _hits)
    _mdir = os.path.dirname(_hits[0])
    print("[m448] member dir:", _mdir, flush=True)
    _sys.path.insert(0, _mdir)
    import btkr_cnx_montage448_optional as _m448

    _base = pd.read_csv(_SUB, dtype={{"StudyInstanceUID": str}})
    if _base.columns.tolist() != ["StudyInstanceUID", *_m448.LABELS]:
        raise RuntimeError("submission schema drift: %r" % _base.columns.tolist())
    if len(_base) == 0:
        raise RuntimeError("empty submission")
    print("[m448] base rows:", len(_base), flush=True)

    _out = "/kaggle/working/_m448_member.csv"
    _t0 = _t.time()
    _m448.run(_hits[0], _out, expected_checkpoint_sha256=M448_SHA,
              workers=4, micro=4)
    _dt = _t.time() - _t0
    print("[m448] member inference: %.1fs for %d studies (%.2f s/study)"
          % (_dt, len(_base), _dt / max(len(_base), 1)), flush=True)

    _mem = pd.read_csv(_out, dtype={{"StudyInstanceUID": str}})
    _mem = _mem.set_index("StudyInstanceUID").loc[_base.StudyInstanceUID].reset_index()
    if _mem.StudyInstanceUID.tolist() != _base.StudyInstanceUID.tolist():
        raise RuntimeError("member/base study alignment failed")
    if not np.isfinite(_mem[list(_m448.LABELS)].to_numpy()).all():
        raise RuntimeError("member produced non-finite values")

    _blended = _base.copy()
    for _c in _m448.LABELS:
        _b = _base[_c].rank(pct=True)
        _m = _mem[_c].rank(pct=True)
        _blended[_c] = (1.0 - M448_W) * _b + M448_W * _m
    if _blended.columns.tolist() != _base.columns.tolist():
        raise RuntimeError("blend changed the schema")
    if _blended.StudyInstanceUID.tolist() != _base.StudyInstanceUID.tolist():
        raise RuntimeError("blend changed the study order")
    # per-label rank correlation between base and member: the member is only worth
    # its weight if it disagrees with the base in a structured way
    _rho = {{c: float(_base[c].corr(_mem[c], method="spearman")) for c in _m448.LABELS}}
    print("[m448] spearman(base, member) per label:", flush=True)
    for _c in _m448.LABELS:
        print("        %-18s %+.4f" % (_c, _rho[_c]), flush=True)
    print("[m448] mean rho = %.4f" % (sum(_rho.values()) / len(_rho)), flush=True)

    _blended.to_csv(_SUB, index=False)
    _chk = pd.read_csv(_SUB, dtype={{"StudyInstanceUID": str}})
    assert _chk.columns.tolist() == _base.columns.tolist()
    assert len(_chk) == len(_base)
    assert np.isfinite(_chk[list(_m448.LABELS)].to_numpy()).all()
    print("[m448] BLEND APPLIED at w=%.2f -> %s" % (M448_W, _SUB), flush=True)
except Exception as _e:
    print("[m448] MEMBER FAILED, base submission left untouched:", flush=True)
    _tb.print_exc()
'''


def main() -> int:
    nb_path = SRC / "rsna-knee-restructured-version-3.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    meta_path = SRC / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    n_before = len(nb["cells"])
    # refuse to double-append
    for c in nb["cells"]:
        s = c.get("source", "")
        if isinstance(s, list):
            s = "".join(s)
        if "btkr_cnx_montage448_optional" in s:
            print("!! member cell already present -- refusing to append twice")
            return 2

    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "m448member",
        "metadata": {},
        "outputs": [],
        "source": CELL_SOURCE,
    })
    assert len(nb["cells"]) == n_before + 1

    meta["id"] = "ppxl16/" + SLUG
    meta["title"] = TITLE
    meta["enable_gpu"] = True
    meta["machine_shape"] = "NvidiaTeslaT4"
    ds = list(meta.get("dataset_sources", []))
    if MEMBER_DATASET not in ds:
        ds.append(MEMBER_DATASET)
    meta["dataset_sources"] = ds

    DST.mkdir(parents=True, exist_ok=True)
    (DST / "rsna-knee-restructured-version-3.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    (DST / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DST}")
    print(f"  cells {n_before} -> {len(nb['cells'])}")
    print(f"  dataset_sources {len(meta['dataset_sources'])}: "
          f"{MEMBER_DATASET in meta['dataset_sources']}")
    print(f"  member weight = {MEMBER_WEIGHT}")

    if "--push" not in sys.argv:
        print("\nnot pushed (pass --push).")
        return 0
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        api.kernels_push(str(DST))
        print("pushed -> ppxl16/%s" % SLUG)
    except Exception as e:
        print("push failed: %s: %s" % (type(e).__name__, str(e)[:500]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
