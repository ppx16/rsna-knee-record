"""Census every `[phase]` record in every log we have, so stage SLOPES can be fitted.

WHY
    `docs/EFFICIENCY-PLAN.md` Step 1 assumed the per-study cost could only be obtained by running the
    pipeline at several study counts. It cannot: the notebook emits
        [phase] {"stage": ..., "status": "START"|"COMPLETE", "elapsed_seconds": ...}
    and different runs cover different numbers of studies, so (N, seconds) points for the SAME named
    stage already exist on disk. `arm_timing.py --phases` reads one log; this reads all of them.

    This matters because the answer decides the whole efficiency track: `RuntimeSeconds/32400` is the
    only term we can move (the AUC term is pinned by the ensemble), and 1 h = 0.0510 AUC-equivalent.

⚠️ TWO TRAPS THIS TOOL EXISTS TO AVOID
    1. `[  123.4s] label` stamps are NOT usable for this: on a 337 s submission run they stop at 111 s,
       because the stages that actually loop over studies (Raptor and the three CoAt members) print no
       timestamped progress. Only `[phase]` records give per-stage boundaries.
    2. Most phase records do NOT carry a study count (`residual_coat` reports only `ok: true`). So N is
       inferred from the log's own prose (`test g1: N studies`, `OOF: ... N training studies`,
       `sizing for T train + N test studies`) and attached to every phase record in that log. A log whose
       N cannot be inferred is reported as `N=?` and EXCLUDED from fits rather than silently treated as 3.

Usage:
    python scripts/phase_census.py                  # table of every (stage, N, seconds) point
    python scripts/phase_census.py --fits           # plus a least-squares slope per stage
    python scripts/phase_census.py --roots state downloads outputs
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parent.parent

# N inference, most specific first. Each is (regex, group index).
N_PATTERNS = [
    (re.compile(r"OOF:\s*A5 cohort\s*=\s*[^,]+,\s*(\d+)\s+studies"), 1),
    (re.compile(r"OOF:\s*fold\s+\d+\s*->\s*(\d+)\s+training studies"), 1),
    (re.compile(r"test g1:\s*(\d+)\s+studies"), 1),
    (re.compile(r"sizing for\s+\d+\s+train\s*\+\s*(\d+)\s+test studies"), 1),
    (re.compile(r"(\d+)\s+studies,\s*\d+\s+slot-series"), 1),
]


def payload(text: str) -> str:
    for marker in ('[{"stream_name"', "[{"):
        i = text.find(marker)
        if i >= 0:
            return text[i:]
    return text


def read_log(path: pathlib.Path):
    """-> (list_of_phase_dicts, inferred_N_or_None, full_text)"""
    try:
        text = io.open(path, encoding="utf-8-sig", errors="replace").read()
    except Exception:
        return [], None, ""
    p = payload(text)
    recs = None
    try:
        recs = json.loads(p)
    except Exception:
        recs = None
    body_parts = []
    phases = []
    if isinstance(recs, list):
        for r in recs:
            if not isinstance(r, dict):
                continue
            d = r.get("data") or ""
            body_parts.append(d)
            for ln in d.splitlines():
                ln = ln.strip()
                if ln.startswith("[phase]"):
                    try:
                        obj = json.loads(ln[len("[phase]"):].strip())
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        phases.append(obj)
    body = "\n".join(body_parts)
    n = None
    for rx, g in N_PATTERNS:
        m = rx.search(body)
        if m:
            n = int(m.group(g))
            break
    return phases, n, body


def pairwise(points):
    """Slope from the two extreme study counts, using the MEAN of the replicates at the low end.

    This is the estimator that actually fits the data we have, and the pooled least-squares fit is not.
    Every stage has ~10 replicates at N=3 -- genuine runs, but all at ONE study count -- plus (for a few
    stages) a single run at a much larger N. Pooling them lets the N=3 spread, which is machine variance
    of 50-93 s, masquerade as a study-count effect; that is how `dinov2` came back as -0.016 s/study
    ("gets faster per study"). Replicates estimate VARIANCE. The slope needs two DISTINCT N, so use the
    low-N mean and the high-N value and report how much the answer moves across the low-N range.
    """
    by: dict[int, list[float]] = {}
    for n, y in points:
        by.setdefault(int(n), []).append(float(y))
    ns = sorted(by)
    if len(ns) < 2:
        return None
    n_lo, n_hi = ns[0], ns[-1]
    lo_vals, hi_vals = by[n_lo], by[n_hi]
    out = {}
    for tag, y_lo in (("min", min(lo_vals)), ("mean", sum(lo_vals) / len(lo_vals)),
                      ("max", max(lo_vals))):
        y_hi = sum(hi_vals) / len(hi_vals)
        s = (y_hi - y_lo) / (n_hi - n_lo)
        out[tag] = (s, y_lo - s * n_lo)
    return n_lo, n_hi, len(lo_vals), out


def fit(points):
    """Least squares y = f + s*x. Returns (slope, fixed) or None."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2 or len({x for x, _ in pts}) < 2:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    if den == 0:
        return None
    s = sum((x - mx) * (y - my) for x, y in pts) / den
    return s, my - s * mx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=["state", "downloads", "outputs", "work", "scratch"])
    ap.add_argument("--fits", action="store_true")
    ap.add_argument("--pairwise", action="store_true",
                    help="two-point slope from the low-N mean vs the high-N value (the estimator that "
                         "matches this data; see pairwise())")
    a = ap.parse_args()

    files = []
    for r in a.roots:
        d = ROOT / r
        if not d.is_dir():
            continue
        for pat in ("*.json", "*.log"):
            files.extend(d.rglob(pat))
    files = sorted(set(files))

    # stage -> list of (N, seconds, source)
    census: dict[str, list[tuple[int, float, str]]] = {}
    nlogs = nphases = 0
    unknown = []
    for f in files:
        phases, n, _body = read_log(f)
        if not phases:
            continue
        nlogs += 1
        if n is None:
            unknown.append(f)
            n = -1
        start: dict[str, float] = {}
        for obj in phases:
            st = obj.get("stage")
            el = obj.get("elapsed_seconds")
            if not st or el is None:
                continue
            if obj.get("status") == "START":
                start[st] = float(el)
            elif obj.get("status") == "COMPLETE" and st in start:
                dur = float(el) - start[st]
                census.setdefault(st, []).append((n, dur, f.name))
                nphases += 1

    if not census:
        print("no [phase] records found under: %s" % ", ".join(a.roots))
        return 1

    # ⚠️ DROP SUB-SECOND "COMPLETIONS": they are SKIPPED stages, not measurements, and one of them
    # poisons the estimator. `a5oof_full` logs `dinov2 COMPLETE` at 0.2 s because that run had the
    # DINOv2 features precomputed, so dinov2 was a no-op there -- and since it is the LARGEST study
    # count on record (4407), a two-point slope against it reported dinov2 as 0.6-1.1 min at N=1322
    # instead of ~39 min. Absolute threshold on purpose: a relative one (say 1% of the stage max) would
    # also discard `a5`'s genuine 20.1 s point. No real stage here loads its weights in under a second.
    skipped = []
    for st in list(census):
        kept = []
        for n, dur, src in census[st]:
            if dur < 1.0:
                skipped.append((st, n, dur, src))
            else:
                kept.append((n, dur, src))
        if kept:
            census[st] = kept
        else:
            del census[st]

    print("%d log(s) with phase records, %d stage-completion(s)" % (nlogs, nphases))
    if skipped:
        print("⚠️ dropped %d sub-second completion(s) as SKIPPED stages, e.g. %s"
              % (len(skipped),
                 ", ".join("%s@N=%s(%.2fs)" % (s[0], s[1], s[2]) for s in skipped[:4])))
    if unknown:
        print("⚠️ %d log(s) with NO inferable study count (excluded from fits): %s"
              % (len(unknown), ", ".join(p.name for p in unknown[:6])))
    print()
    print("%-18s %8s %10s  %s" % ("stage", "N", "seconds", "source"))
    print("-" * 72)
    for st in sorted(census):
        for n, dur, src in sorted(census[st]):
            print("%-18s %8s %10.1f  %s" % (st, n if n >= 0 else "?", dur, src))

    if a.pairwise:
        print()
        print("%-18s %14s %10s %9s %9s  %s"
              % ("stage", "N lo->hi(rep)", "s/study", "low_s", "high_s", "@1322 (range) min"))
        print("-" * 88)
        for st in sorted(census):
            pts = [(n, d) for n, d, _s in census[st] if n >= 0]
            got = pairwise(pts)
            if not got:
                continue
            n_lo, n_hi, nrep, out = got
            sl = out["mean"][0]
            fixed = out["mean"][1]
            lo1322 = (min(o[1] + o[0] * 1322 for o in out.values())) / 60.0
            hi1322 = (max(o[1] + o[0] * 1322 for o in out.values())) / 60.0
            print("%-18s %14s %10.4f %9.1f %9.1f  %.1f - %.1f"
                  % (st, "%d->%d(%d)" % (n_lo, n_hi, nrep), sl, fixed,
                     out["mean"][1] + sl * n_hi, lo1322, hi1322))
        print()
        print("⚠️ `low_s`/`high_s` are the fitted stage cost at the two measured study counts, not the")
        print("   fixed term. The @1322 range spans the low-N replicate spread (min..max), so it is the")
        print("   honest uncertainty on the extrapolation.")
        return 0

    if a.fits:
        print()
        print("%-18s %5s %6s %10s %9s %9s %9s  %s"
              % ("stage", "pts", "N#", "s/study", "fixed_s", "@1322_s", "noise_s", "verdict"))
        print("-" * 96)
        for st in sorted(census):
            pts = [(n, d) for n, d, _s in census[st] if n >= 0]
            distinct = sorted({x for x, _ in pts})
            if len(distinct) < 2:
                continue
            got = fit(pts)
            if not got:
                continue
            s, f = got
            # ⚠️ THE GUARD THAT MATTERS. Replicates of one N measure VARIANCE, not slope. If the spread
            # within a single N is larger than the total effect the slope predicts across the whole N
            # range, the fit is noise wearing a slope's clothes -- this is how a pooled `dinov2` fit came
            # back as -0.016 s/study ("gets faster per study"). Report it instead of printing a number.
            within = max((max(d for n_, d in pts if n_ == n) - min(d for n_, d in pts if n_ == n))
                         for n in distinct if sum(1 for n_, _ in pts if n_ == n) > 1) \
                if any(sum(1 for n_, _ in pts if n_ == n) > 1 for n in distinct) else 0.0
            signal = abs(s) * (distinct[-1] - distinct[0])
            unreliable = within > signal
            print("%-18s %5d %6d %10.4f %9.1f %9.0f %9.1f  %s"
                  % (st, len(pts), len(distinct), s, f, f + s * 1322, within,
                     "UNRELIABLE (within-N spread > fitted signal)" if unreliable else "ok"))
        print()
        print("⚠️ A slope fitted across DIFFERENT notebooks/entry points is not a measurement of the")
        print("   submission pipeline. Prefer points from the same stage name and check the sources.")
        print("   A stage needs >=2 DISTINCT N; ten replicates of N=3 add no slope information.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
