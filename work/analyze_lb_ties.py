"""Analyse the full public leaderboard: tie-breaking, block structure, medal thresholds.

Reads state/lb_full/*.zip (downloaded via `kaggle competitions leaderboard --download`).
Answers: do equal scores rank by earliest submission time, and where would a
late 0.950 submission land?
"""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load() -> list[dict]:
    zp = sorted((ROOT / "state" / "lb_full").glob("*.zip"))[-1]
    z = zipfile.ZipFile(zp)
    raw = z.read(z.namelist()[0]).decode("utf-8-sig", errors="replace")
    rows = list(csv.DictReader(io.StringIO(raw)))
    out = []
    for x in rows:
        try:
            x["Rank"] = int(x["Rank"])
            x["Score"] = float(x["Score"])
        except (ValueError, KeyError):
            continue
        out.append(x)
    out.sort(key=lambda r: r["Rank"])
    return out


def main() -> None:
    rows = load()
    n = len(rows)
    print(f"source rows={n}  bronze(top10%) rank<={int(n * 0.10)}")

    scores = sorted({r["Score"] for r in rows}, reverse=True)
    print("\n=== block structure (top 25 blocks) ===")
    for s in scores[:25]:
        blk = [r for r in rows if abs(r["Score"] - s) < 1e-9]
        ts = [r["LastSubmissionDate"] for r in blk]
        inc = sum(1 for a, b in zip(ts, ts[1:]) if a <= b)
        mono = "TIME-ORDERED" if inc >= len(ts) - 3 else "NOT-time-ordered"
        print(
            f"  {s:.3f}  n={len(blk):5d}  ranks {blk[0]['Rank']:5d}..{blk[-1]['Rank']:5d}  {mono}"
        )

    print("\n=== the 0.950 block in detail ===")
    blk = [r for r in rows if abs(r["Score"] - 0.950) < 1e-9]
    print(f"  n={len(blk)} ranks {blk[0]['Rank']}..{blk[-1]['Rank']}")
    print("  earliest 4:")
    for r in blk[:4]:
        print(f"    {r['Rank']:5d} {r['LastSubmissionDate']}  {r['TeamName'][:30]}")
    print("  latest 6:")
    for r in blk[-6:]:
        print(f"    {r['Rank']:5d} {r['LastSubmissionDate']}  {r['TeamName'][:30]}")
    ts = [r["LastSubmissionDate"] for r in blk]
    inc = sum(1 for a, b in zip(ts, ts[1:]) if a <= b)
    print(f"  non-decreasing date pairs: {inc}/{len(ts) - 1}")

    print("\n=== where a NEW 0.950 submission would land ===")
    blk950 = [r for r in rows if abs(r["Score"] - 0.950) < 1e-9]
    n950 = len(blk950)
    print(f"  appended at the END of the 0.950 block => rank ~{blk950[-1]['Rank'] + 1}")
    print(f"  bronze cut now = {int(n * 0.10)}  => {'WOULD make bronze' if blk950[-1]['Rank'] + 1 <= int(n * 0.10) else 'would NOT make bronze'}")

    print("\n=== score needed to be safely inside bronze ===")
    cut = int(n * 0.10)
    for k in (int(cut * 0.5), cut - 50, cut, cut + 50):
        r = rows[k - 1]
        print(f"  rank {k:5d} -> {r['Score']:.3f}")

    print("\n=== our team ===")
    for r in rows:
        if "ppxl16" in r.get("TeamMemberUserNames", ""):
            print(
                f"  rank {r['Rank']} score {r['Score']:.3f} last {r['LastSubmissionDate']}"
                f" n_sub={r['SubmissionCount']} name={r['TeamName']!r}"
            )

    print("\n=== how many teams sit at >= each level ===")
    for th in (0.944, 0.945, 0.946, 0.947, 0.948, 0.949, 0.950, 0.951, 0.952, 0.955):
        c = sum(1 for r in rows if r["Score"] >= th - 1e-9)
        print(f"  >= {th:.3f}: {c:5d}")


if __name__ == "__main__":
    main()
