"""Where exactly does the bronze cut fall inside the 0.950 block, and where will it move?

FACTS C113 established that ties are ranked by the UNROUNDED score, so the displayed 0.950
is a band. That makes "is 0.950 enough for bronze?" a question about POSITION INSIDE THE
BAND, not about the band's value. This script answers it with the real board and projects
it forward as more teams join before the 2026-10-22 deadline.

Reads the newest state/lb_full/*.zip (download with:
    kaggle competitions leaderboard -c rsna-knee-abnormality-detection --download -p state/lb_full)
"""
from __future__ import annotations

import csv
import io
import json
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDAL_FRACTION = 0.10
TRACK = ROOT / "state" / "block_position.jsonl"
# filled by load(): our own rank/score, because the decisive metric is not the overall rank
# but WHERE WE SIT INSIDE THE 0.950 BLOCK (FACTS C129).
OUR: dict = {"rank": None, "score": None, "team": None}


def snapshot_time(zp) -> str:
    """The REAL time this board was read, taken from the CSV name INSIDE the zip.

    Never use the file's mtime for freshness: this script's own archiving step copies a stale
    snapshot to a new file, which gives STALE data a FRESH mtime.  That is how 2026-10-12's run
    read the 2026-10-11T14:18 board and reported rank 310 / "margin +265 INSIDE" while the same
    output printed rank 730 / margin -148 from the fresh JSON snapshot (FACTS A187).
    """
    z = zipfile.ZipFile(zp)
    names = [n for n in z.namelist() if n.endswith(".csv")]
    # The CLI names the member `...-publicleaderboard-2026-10-12T13:17:04.csv` -- the stamp uses
    # COLONS.  Accept both separators: an earlier version matched only underscores, silently
    # returned a sentinel for every board, and made max() pick the OLDEST snapshot.
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}[:_]\d{2}[:_]\d{2})", names[0]) if names else None
    return m.group(1).replace("_", ":") if m else "0000-00-00T00:00:00"


def fetch_board():
    """Download the board NOW and return (zip_path, stamp).

    FACTS A187/C140: this script used to read `state/lb_full/*.zip`, a directory nothing ever
    refreshed, while half of its output came from the fresh `state/lb_snapshot_*.json`.  One run
    therefore printed rank 730 AND rank 310, and the stale half is the one that said we were safe.
    There is now ONE source: a download made by this process, in the same place lb_snapshot.py
    uses.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    last = None
    for attempt in range(6):
        try:
            api.authenticate()
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(8)
    else:
        raise SystemExit("COULD NOT AUTHENTICATE (%r) -- transport failure, not a result" % (last,))
    live = ROOT / "work" / "lb_live"
    live.mkdir(parents=True, exist_ok=True)
    api.competition_leaderboard_download("rsna-knee-abnormality-detection", path=str(live), quiet=True)
    zp = live / "rsna-knee-abnormality-detection.zip"
    if not zp.is_file():
        raise SystemExit("download did not produce %s" % zp)
    return zp, snapshot_time(zp)


def newest_zip():
    """Newest ARCHIVED snapshot by the timestamp it actually contains (trend section only)."""
    zips = list((ROOT / "state" / "lb_full").glob("*.zip")) + \
           list((ROOT / "state" / "lb_archive").glob("*.zip"))
    if not zips:
        return None
    return max(zips, key=snapshot_time)


def load() -> list[tuple[int, float]]:
    zp, stamp = fetch_board()
    # Archive a dated copy so later comparisons are possible -- the CLI overwrites in place.
    # The archive lives in its OWN directory: a copy inside the live/downloaded directory would get
    # a fresh mtime and masquerade as a newer board on the next run.
    adir = ROOT / "state" / "lb_archive"
    adir.mkdir(parents=True, exist_ok=True)
    dated = adir / ("lb_%s_%s" % (stamp.replace(":", "").replace("-", ""), zp.name))
    if not dated.exists():
        dated.write_bytes(zp.read_bytes())
    z = zipfile.ZipFile(zp)
    raw = z.read(z.namelist()[0]).decode("utf-8-sig", errors="replace")
    out = []
    for r in csv.DictReader(io.StringIO(raw)):
        try:
            rank, score = int(r["Rank"]), float(r["Score"])
        except (ValueError, KeyError):
            continue
        out.append((rank, score))
        members = (r.get("TeamMemberUserNames") or "") + " " + (r.get("TeamName") or "")
        if "ppxl16" in members.replace(" ", ""):
            OUR.update(rank=rank, score=score, team=r.get("TeamName"))
    out.sort()
    print(f"source: board downloaded now, read at {stamp}  rows={len(out)}")
    if OUR["rank"]:
        print(f"us: rank {OUR['rank']}  score {OUR['score']}  team={OUR['team']!r}")
    else:
        print("us: NOT FOUND on the board (check the member-name match)")
    return out


def guard_against_the_json_snapshot() -> None:
    """Reconcile this run's own reading with the newest lb_snapshot_*.json.

    FACTS A187: ONE script printed two contradictory ranks because half of it read a stale zip and
    half read the JSON, and the output looked self-consistent.  Both now read the live board, so
    they may differ only by teams that arrived in between -- refuse on a real disagreement.
    """
    snaps = sorted((ROOT / "state").glob("lb_snapshot_*.json"),
                   key=lambda p: p.stat().st_mtime_ns)
    if not snaps:
        print("guard: no lb_snapshot_*.json yet -- nothing to reconcile against")
        return
    j = json.loads(snaps[-1].read_text(encoding="utf-8"))
    print(f"guard: newest JSON snapshot {snaps[-1].name}  our_rank={j.get('our_rank')}  "
          f"N={j.get('n_teams')}")
    bad = []
    if j.get("our_score") is not None and OUR["score"] is not None:
        if abs(float(j["our_score"]) - float(OUR["score"])) > 1e-9:
            bad.append("our_score json=%s vs live=%s" % (j["our_score"], OUR["score"]))
    if j.get("our_rank") is not None and OUR["rank"] is not None:
        if abs(int(j["our_rank"]) - int(OUR["rank"])) > 25:
            bad.append("our_rank json=%s vs live=%s (>25 apart)" % (j.get("our_rank"), OUR["rank"]))
    if bad:
        raise SystemExit(
            "REFUSING: the JSON snapshot and the live board disagree (%s).\n"
            "  Re-run work/lb_snapshot.py, then this; do not trust any number below this line."
            % "; ".join(bad))
    print("guard: JSON snapshot and live board agree -> readings below come from one board")




def blocks(rows: list[tuple[int, float]]) -> list[tuple[float, int, int, int]]:
    """(score, first_rank, last_rank, n) for each distinct displayed score, descending."""
    out = []
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and abs(rows[j + 1][1] - rows[i][1]) < 1e-9:
            j += 1
        out.append((rows[i][1], rows[i][0], rows[j][0], j - i + 1))
        i = j + 1
    return out


def main() -> None:
    rows = load()
    guard_against_the_json_snapshot()
    n = len(rows)
    cut = int(n * MEDAL_FRACTION)
    cut_score = rows[cut - 1][1]
    print(f"N={n}  bronze cut rank={cut}  score at cut={cut_score:.3f}")

    print("\n=== block structure near the cut ===")
    for s, lo, hi, cnt in blocks(rows):
        if hi < cut - 900 or lo > cut + 900:
            continue
        inside = "  <== CUT IS INSIDE THIS BLOCK" if lo <= cut <= hi else ""
        print(f"  {s:.3f}  n={cnt:5d}  ranks {lo:5d}..{hi:5d}{inside}")

    print("\n=== position of the cut inside its own block ===")
    our_row = None
    for s, lo, hi, cnt in blocks(rows):
        if lo <= cut <= hi:
            above = cut - lo + 1          # teams in this block at or above the cut
            pct = 100.0 * above / cnt
            print(f"  cut score {s:.3f}: block ranks {lo}..{hi} (n={cnt})")
            print(f"  => {above} of {cnt} teams in the block make bronze "
                  f"({pct:.0f}% of the block)")
            print(f"  => a submission whose unrounded score lands UNIFORMLY in this block "
                  f"medals with probability {pct:.0f}%")
            our_row = (s, lo, hi, cnt, cut, above, pct)
            break

    # --- WHERE WE SIT: the metric that actually decides our medal (FACTS C129) ---------
    # IMPORTANT: find OUR OWN block, not the block the CUT falls in. Once we score above the
    # cut's band the two are different, and using the cut's block produced nonsense like
    # "position -294/571 (-51%)".
    if OUR["rank"]:
        ours = None
        for s, lo, hi, cnt in blocks(rows):
            if lo <= OUR["rank"] <= hi:
                ours = (s, lo, hi, cnt)
                break
        silver_cut = int(n * 0.05)
        print("\n=== OUR POSITION (the decisive metric) ===")
        if ours is None:
            print("  !! could not locate our rank in any block")
        else:
            s, lo, hi, cnt = ours
            pos = OUR["rank"] - lo + 1
            print(f"  we are at rank {OUR['rank']} (score {OUR['score']}) in the {s:.3f} band "
                  f"{lo}..{hi} (n={cnt}) -> position {pos}/{cnt} "
                  f"({100.0 * pos / cnt:.0f}% down our own band)")
            for name, cutr in (("bronze (top 10%)", int(n * MEDAL_FRACTION)),
                               ("silver (top 5%)", silver_cut),
                               ("gold (top 10 teams or 0.2%)", max(10, int(n * 0.002)))):
                m = cutr - OUR["rank"]
                print(f"  {name:28s} rank {cutr:5d} -> margin {m:+5d} ranks "
                      f"({'INSIDE' if m >= 0 else 'outside'})")
            print(f"  our band gives us {max(0, (hi - OUR['rank']) + 1)} ranks of room before "
                  f"the band itself ends")
        TRACK.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_teams": n,
            "cut_rank": int(n * MEDAL_FRACTION), "cut_score": cut_score,
            "silver_cut_rank": silver_cut,
            "our_rank": OUR["rank"], "our_score": OUR["score"],
            "margin_bronze": int(n * MEDAL_FRACTION) - OUR["rank"],
            "margin_silver": silver_cut - OUR["rank"],
        }
        if ours is not None:
            s, lo, hi, cnt = ours
            rec.update({"our_block_score": s, "block_lo": lo, "block_hi": hi, "block_n": cnt,
                        "our_pos_in_block": OUR["rank"] - lo + 1})
        with TRACK.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  appended to {TRACK.name}")

    print("\n=== projection: what happens as more teams join? ===")
    print("  NOTE: the naive 'block grows in proportion to N' assumption is OPTIMISTIC.")
    print("  What matters is the block's growth relative to N; compare the archives below.")
    widths = {}
    for s, lo, hi, cnt in blocks(rows):
        widths[s] = (lo, hi, cnt)
    cur_cut_score = cut_score
    lo, hi, cnt = widths[cur_cut_score]
    for n_final in (5627, 6000, 6500, 7000, 7500, 8000):
        c = int(n_final * MEDAL_FRACTION)
        width = cnt * (n_final / n)
        new_lo = c - (1 - ((hi - cut) / cnt)) * width
        if c < new_lo:
            verdict = f"cut moves ABOVE this block -> would need {cur_cut_score + 0.001:.3f}"
        else:
            above = c - new_lo + 1
            pct = 100.0 * above / width
            verdict = f"{pct:.0f}% of the block makes bronze (if block/N is unchanged)"
        print(f"  N={n_final:5d}  cut rank={c:5d}  block≈{new_lo:.0f}..{new_lo + width:.0f}  -> {verdict}")

    # --- trend across archived snapshots: this is the part that is NOT an assumption ---
    # Deduplicate by the timestamp the board actually carries, then order by THAT. Sorting these
    # by name or mtime put copies-of-copies in the series and read the same board six times.
    seen: dict = {}
    for a in list((ROOT / "state" / "lb_full").glob("*.zip")) + \
             list((ROOT / "state" / "lb_archive").glob("*.zip")):
        try:
            seen.setdefault(snapshot_time(a), a)
        except Exception:  # noqa: BLE001
            continue
    arch = [seen[k] for k in sorted(seen)]
    if len(arch) >= 2:
        print("\n=== OBSERVED trend across archived snapshots (this is data, not a model) ===")
        hist = []
        for a in arch[-6:]:
            try:
                z2 = zipfile.ZipFile(a)
                r2 = z2.read(z2.namelist()[0]).decode("utf-8-sig", errors="replace")
                sc = {}
                tot = 0
                for row in csv.DictReader(io.StringIO(r2)):
                    try:
                        sc[int(row["Rank"])] = float(row["Score"])
                    except (ValueError, KeyError):
                        continue
                    tot += 1
                if not tot:
                    continue
                c2 = int(tot * MEDAL_FRACTION)
                csc = sc.get(c2)
                lo2 = min(k for k, v in sc.items() if abs(v - csc) < 1e-9) if csc else None
                hi2 = max(k for k, v in sc.items() if abs(v - csc) < 1e-9) if csc else None
                pct = 100.0 * (c2 - lo2 + 1) / (hi2 - lo2 + 1) if lo2 else None
                hist.append((a.name, tot, c2, csc, lo2, hi2, pct))
            except Exception:  # noqa: BLE001
                continue
        for name, tot, c2, csc, lo2, hi2, pct in hist:
            print(f"  {name[:52]:52s} N={tot:5d} cut={c2:5d} score={csc:.3f} "
                  f"block={lo2}..{hi2}  cut at {pct:.0f}% of block")
        if len(hist) >= 2:
            d_n = hist[-1][1] - hist[0][1]
            d_blk = (hist[-1][5] - hist[-1][4]) - (hist[0][5] - hist[0][4])
            print(f"  => over {len(hist)} snapshots: N {hist[0][1]}->{hist[-1][1]} (+{d_n}), "
                  f"the cut's block grew by {d_blk}")
            print(f"  => cut percentile {hist[0][6]:.0f}% -> {hist[-1][6]:.0f}% "
                  f"({'WORSENING for a displayed-0.950' if hist[-1][6] < hist[0][6] else 'improving'})")

    print("\n=== our best non-OAI score for reference ===")
    for want in (0.944, 0.950, 0.951):
        for s, lo, hi, cnt in blocks(rows):
            if abs(s - want) < 1e-9:
                print(f"  {want:.3f}: ranks {lo}..{hi}  -> "
                      f"{'INSIDE bronze' if lo <= cut else 'OUTSIDE bronze'}"
                      f" (needs rank <= {cut})")
                break
        else:
            print(f"  {want:.3f}: no team at this score")


if __name__ == "__main__":
    main()
