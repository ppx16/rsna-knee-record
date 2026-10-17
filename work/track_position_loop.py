"""Continuously track WHERE WE SIT INSIDE THE 0.950 BLOCK.

Why this and not the overall rank: bronze is rank <= 10% of N, and a huge band of teams all
DISPLAY 0.950. Ties inside that band are broken by the unrounded score (FACTS C113). So the
only number that decides our medal is our position inside our own band, and how fast it erodes.

Erosion model: our unrounded score is fixed once submitted, so our rank rises only when new teams
land ABOVE us. Meanwhile the cut rank rises by ~0.1 per new team. Margin therefore erodes at
(new-teams-above-us per new-team) - 0.1. This loop measures that empirically instead of guessing.

Each iteration:
  1. refresh the full public leaderboard (archived by bronze_forecast);
  2. run bronze_forecast, which appends (our_rank, block, margin) to state/block_position.jsonl;
  3. print the observed trend so far.

Usage: python work/track_position_loop.py [hours] [interval_minutes]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "state" / "block_position.jsonl"
COMP = "rsna-knee-abnormality-detection"


def refresh_lb() -> None:
    for _ in range(3):
        r = subprocess.run(["kaggle", "competitions", "leaderboard", "-c", COMP,
                            "--download", "-p", str(ROOT / "state" / "lb_full"), "--quiet"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=1200)
        if r.returncode == 0:
            return
        time.sleep(30)
    print("  !! leaderboard refresh failed (transport) -- keeping the previous file")


def run_forecast() -> str:
    r = subprocess.run([sys.executable, str(ROOT / "work" / "bronze_forecast.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=900)
    return (r.stdout or "") + (r.stderr or "")


def trend() -> str:
    if not TRACK.is_file():
        return "  (no history yet)"
    rows = []
    for line in TRACK.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if len(rows) < 2:
        return f"  {len(rows)} datapoint(s) so far -- need >=2 for a trend"

    # The record schema changed on 2026-10-11: bronze_forecast now writes margin_bronze /
    # margin_silver (and a per-band view) because we moved out of the cut's band entirely.
    # Read BOTH schemas, otherwise the loop dies on the first iteration it ever prints.
    def m_bronze(r: dict) -> int | None:
        for k in ("margin_bronze", "margin_ranks"):
            if k in r:
                return int(r[k])
        return None

    def m_silver(r: dict) -> int | None:
        return int(r["margin_silver"]) if "margin_silver" in r else None

    def describe(r: dict) -> str:
        pos = (f"{r['our_pos_in_block']}/{r['block_n']}"
               if "our_pos_in_block" in r and "block_n" in r else "-")
        b, s = m_bronze(r), m_silver(r)
        bits = [f"{r['at']}", f"N={r['n_teams']}", f"rank={r['our_rank']}",
                f"score={r.get('our_score')}", f"bandpos={pos}"]
        if b is not None:
            bits.append(f"bronze={b:+d}")
        if s is not None:
            bits.append(f"silver={s:+d}")
        return "  ".join(bits)

    out = [f"  {len(rows)} datapoints"]
    for r in rows[-6:]:
        out.append("    " + describe(r))
    first, last = rows[0], rows[-1]
    dn = last["n_teams"] - first["n_teams"]
    b0, b1 = m_bronze(first), m_bronze(last)
    if dn > 0 and b0 is not None and b1 is not None:
        dm = b1 - b0
        out.append(f"  observed: N +{dn} -> bronze margin {dm:+d} "
                   f"({dm / dn:+.3f} margin per new team)")
        if dm < 0 and b1 > 0:
            rate = -dm / dn
            out.append(f"  => bronze margin erodes {rate:.2f} per new team; "
                       f"{b1 / rate:.0f} more teams would consume the remaining buffer")
        else:
            out.append("  => bronze margin is holding or growing")
    s1 = m_silver(last)
    if s1 is not None:
        out.append(f"  silver margin now {s1:+d} "
                   f"({'INSIDE silver' if s1 >= 0 else 'outside silver'})")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    hours = float(argv[1]) if len(argv) > 1 else 6.0
    every = float(argv[2]) if len(argv) > 2 else 30.0
    t0 = time.time()
    print(f"tracking block position for {hours:.1f} h, every {every:.0f} min")
    while True:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        print(f"\n=== iteration at {stamp} "
              f"({(time.time() - t0) / 60.0:.0f} min elapsed) ===", flush=True)
        refresh_lb()
        out = run_forecast()
        for line in out.splitlines():
            if any(k in line for k in ("us:", "we are at", "bronze cut is rank",
                                       "buffer as a fraction", "appended to",
                                       "CUT IS INSIDE", "not found")):
                print("  " + line.strip(), flush=True)
        print(trend(), flush=True)
        if (time.time() - t0) / 3600.0 >= hours:
            print("\n=== tracking window finished ===", flush=True)
            return 0
        time.sleep(every * 60)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
