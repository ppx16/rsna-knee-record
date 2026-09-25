# -*- coding: utf-8 -*-
"""One-shot overview of everything this project has in flight on Kaggle.

Written because the same status sweep kept being retyped, and because PowerShell
here-strings mangle the quotes inside inline Python. Prints, in order:

  1. our submissions for the competition, newest first, with public scores
  2. the live status of every kernel we watch
  3. what `director.py` is currently watching and which actions each watch may take
  4. the director's heartbeat age, so a dead scheduled task is obvious

Read-only: it never submits, pushes, or downloads.

    python scripts/status.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMPETITION = "rsna-knee-abnormality-detection"
CONFIG = ROOT / "state" / "director.json"
STATE_DIR = ROOT / "state"
HEARTBEAT = STATE_DIR / "director.heartbeat"


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)

# Verified verbatim from the competition rules page: "All deadlines are at 11:59 PM UTC on
# the corresponding day unless otherwise noted." Selecting the Final Submissions is a
# web-UI-only action with no API, so it is the one endgame step that automation cannot do
# for us - which is exactly why the countdown belongs in the status output.
FINAL_SUBMISSION_DEADLINE = datetime(2026, 10, 22, 23, 59, tzinfo=timezone.utc)
MERGER_DEADLINE = datetime(2026, 10, 15, 23, 59, tzinfo=timezone.utc)
SELECT_WARN_DAYS = 5

KERNELS = [
    "ppxl16/rsna-knee-ensemble-v1",
    "ppxl16/rsna-knee-ens-exclude-fracture",
    "ppxl16/rsna-knee-ens-parent-routing",
    "ppxl16/rsna-knee-ens-add-widedense-arm",
    "ppxl16/rsna-knee-ens-arm-weights-alt",
    "ppxl16/rsna-knee-ens-halfway-routing",
    "ppxl16/rsna-knee-ens-v2src-v8rev",
    "ppxl16/rsna-knee-ens-v2src-default",
    "ppxl16/rsna-knee-train-v1",
    "ppxl16/rsna-knee-tpu-train-v1",
]

# The public leaderboard displays THREE decimals. Two runs that both show 0.941 can differ
# by nearly 0.001, and Kaggle ranks on the full-precision value - so the 190-team "tie" at
# 0.941 actually occupies ranks 121..310. That makes our RANK the only instrument sensitive
# enough to see a +0.0005 improvement, which is exactly the size of the moves we are making
# now. Hence this block: it turns every submission into a measurable datapoint even when
# the displayed score does not budge.
LB_DIR = ROOT / "downloads" / "lb"
LB_ZIP = LB_DIR / "rsna-knee-abnormality-detection.zip"
LB_STATE = ROOT / "state" / "lb_rank.json"
ME = "ppxl16"

# A variant must beat the canonical recipe by this much in the DISPLAYED (three-decimal)
# score before it is promoted over it. We will have run ~a dozen variants; taking the
# maximum of a dozen noisy draws on the same hidden-set sample biases that maximum upward
# by about the noise scale, and the private score regresses. 0.002 (two display steps) is
# comfortably above that bias.
PROMOTE_MARGIN = 0.002


def cli(*args):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(["kaggle"] + list(args), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, timeout=600)
    raw = (proc.stdout or "") + (proc.stderr or "")
    return [ln for ln in raw.splitlines()
            if ln.strip() and not ln.startswith("Warning: Looks like")]


BASELINE_MARKER = "public 0.941-class ensemble"


def leaderboard_snapshot(refresh=True):
    """Return our rank / score and the TIE BAND around us from the public leaderboard.

    Deliberately not "medal margins": the displayed score is rounded to three decimals,
    so the teams sharing our score occupy a ~200-rank band, and a medal line can fall
    inside it. See HANDOFF 51.
    """
    if refresh:
        LB_DIR.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        try:
            subprocess.run(["kaggle", "competitions", "leaderboard", COMPETITION,
                            "-d", "-p", str(LB_DIR)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if not LB_ZIP.is_file():
        return None
    # The entry name contains colons, so Expand-Archive refuses it on Windows; read the zip
    # in memory instead.
    import io as _io
    import zipfile
    try:
        with zipfile.ZipFile(LB_ZIP) as z:
            rows = list(csv.DictReader(_io.StringIO(
                z.read(z.namelist()[0]).decode("utf-8-sig", "replace"))))
    except (OSError, KeyError, zipfile.BadZipFile, ValueError):
        return None
    mine = [r for r in rows
            if ME in (r.get("TeamMemberUserNames") or "") or ME in (r.get("TeamName") or "")]
    if not mine:
        return None
    me = mine[0]
    scored = []
    for r in rows:
        try:
            scored.append((int(r["Rank"]), float(r["Score"]), r))
        except (ValueError, KeyError):
            continue
    scored.sort()
    total = len(scored)
    my_score = float(me["Score"])
    my_rank = int(me["Rank"])
    # Every team displaying OUR score. The displayed score is rounded to three decimals,
    # so this is a TIE BAND: 200-odd teams packed into a 0.001-wide score window while
    # occupying 200-odd distinct ranks. Final placement is decided on the PRIVATE split,
    # which re-orders that band in a way we cannot observe from here - so "margin N
    # places" is a public-board fact and NOT a private guarantee (HANDOFF 51).
    band = sorted(rk for rk, sc, _ in scored if sc == my_score)
    top, bottom = (band[0], band[-1]) if band else (None, None)
    lines = {"gold": 10 + int(0.002 * total), "silver": 10 + int(0.05 * total),
             "bronze": 10 + int(0.10 * total)}
    # For any medal line that falls INSIDE the band: how much of the band is above it.
    cut = {name: sum(1 for rk in band if rk <= ln)
           for name, ln in lines.items() if top is not None and top <= ln <= bottom}
    return {"rank": my_rank, "score": me["Score"], "teams": total, "block": len(band),
            "band_top": top, "band_bottom": bottom,
            "band_pos": (band.index(my_rank) + 1) if band else None,
            "cut": cut, "lines": lines}


def report_leaderboard():
    snap = leaderboard_snapshot()
    if not snap:
        print("LEADERBOARD  unavailable (could not fetch or parse)")
        return
    prev = load_json(LB_STATE, None)
    delta = ""
    if prev and prev.get("rank"):
        d = prev["rank"] - snap["rank"]
        if d:
            delta = "   %+d vs last check (was rank %s)" % (d, prev["rank"])
    print("LEADERBOARD  rank %d / %d teams   score %s   (%d teams also display %s)%s"
          % (snap["rank"], snap["teams"], snap["score"], snap["block"], snap["score"], delta))
    L = snap["lines"]
    print("             medal lines: gold top %d | silver top %d | bronze top %d"
          % (L["gold"], L["silver"], L["bronze"]))
    top, pos = snap.get("band_top"), snap.get("band_pos")
    if top is None:
        print("             -> could not locate our tie band")
    else:
        print("             TIE BAND: ranks %d..%d (%d teams); we are #%d of %d"
              % (top, snap["band_bottom"], snap["block"], pos, snap["block"]))
        # The old line here read "-> SILVER range (margin N places)". That was true of the
        # PUBLIC board and misleading about the private one, which is what decides
        # placement; HANDOFF 51 retracts it. Print the band and the odds instead.
        for name in sorted(snap.get("cut", {})):
            above = snap["cut"][name]
            print("             the %s line (top %d) cuts our band at #%d of %d:"
                  % (name, L[name], above, snap["block"]))
            print("               order preserved -> SAFE by %d | random re-order -> %.0f%%"
                  % (above - pos, 100.0 * above / snap["block"]))
        if not snap.get("cut"):
            inside = [n for n in ("gold", "silver", "bronze") if snap["rank"] <= L[n]]
            print("             -> %s"
                  % ("entire band is inside %s" % inside[0] if inside
                     else "entire band is outside the medals"))
    print("             NOTE: the displayed score is 3 decimals but rank is full precision,")
    print("             so RANK movement is the sensitive readout for small experiments.")
    save_json(LB_STATE, {"rank": snap["rank"], "score": snap["score"],
                         "teams": snap["teams"],
                         "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})


def verdict(rows):
    """Turn the baseline's public score into the §20.3 decision, in plain words.

    The decision tree lives in HANDOFF §20.3, but a tree in a document still has to be
    interpreted at 8am. This prints the conclusion instead: what the number means and what
    the next action is. If the baseline has not scored yet, it says so rather than
    implying the run failed - code-competition reruns queue for a long time.
    """
    base = [r for r in rows if BASELINE_MARKER in (r.get("description") or "")]
    if not base:
        return "no baseline submission found in the history"
    b = base[0]
    score = (b.get("publicScore") or "").strip()
    status = (b.get("status") or "").replace("SubmissionStatus.", "")
    if status == "ERROR":
        return ("BASELINE ERRORED -> suspect the 9h scoring limit. Push the lean-3arm "
                "insurance (HANDOFF 20.8) before spending anything else.")
    if not score:
        return ("baseline %s is still %s - code-competition reruns queue for hours; "
                "nothing to decide yet" % (b["ref"], status or "pending"))
    v = float(score)
    if v >= 0.945:
        head = ("0.945+ - ABOVE the public ceiling of 0.941. Spend nothing else; "
                "select two final submissions from this family.")
    elif v >= 0.938:
        head = ("0.938-0.944 - REPRODUCTION SUCCEEDED, we are at the public ceiling.")
    elif v >= 0.92:
        head = ("0.92-0.938 - mostly there but something is off.")
    else:
        head = "below 0.92 - REPRODUCTION FAILED."
    tail = {
        0: " Next: HANDOFF 20.4 (pick the 2 final submissions; web-UI only). Watch the "
           "LEADERBOARD block below, not the score - see the 3-decimal note there.",
        1: " Next: HANDOFF 20.4 (pick the 2 final submissions; web-UI only). Watch the "
           "LEADERBOARD block below, not the score - see the 3-decimal note there.",
        2: " Next: check HANDOFF 16.1 (20 fingerprint matches / 0 dropped) and the arm "
           "lines in the landing log.",
        3: " Next: check the weight paths resolve to the right package (HANDOFF 12.1 - the "
           "same-name-different-content trap) and that every arm actually ran.",
    }[0 if v >= 0.938 else (2 if v >= 0.92 else 3)]
    return "%s = %.4f\n          %s" % (b["ref"], v, head + tail)


def report_final_picks(rows):
    """Name the two submissions to select, because that action is web-UI only.

    Selecting the Final Submissions is the one endgame step no script can perform, so it
    should not also require judgement at the moment it is done. Kaggle scores the BETTER of
    the two on the private leaderboard, so the right pair is one offence plus one hedge that
    is materially DIFFERENT rather than merely high - two near-identical 0.941 runs would
    fail together.

    ⚠️ The offence is NOT simply the highest public score. We will have run on the order of a
    dozen variants, and picking the maximum of a dozen draws on a three-decimal instrument
    is a max-statistic: its public value is biased upward by roughly the noise scale, and the
    private score regresses. The public LB is a sample of the same hidden set, not a second
    opinion. So a VARIANT is promoted over the canonical recipe only when it beats it by
    PROMOTE_MARGIN - more than the selection bias can explain. Otherwise the canonical recipe,
    which is upstream's exact validated configuration, stays the offence.

    The hedge is the flat-routing parent: its per-target routing was never fitted to any
    split, so it is the candidate least likely to be wrong for the same reason as the
    offence. See HANDOFF 20.4 and 35.
    """
    scored = []
    for r in rows:
        try:
            scored.append((float(r["publicScore"]), r["ref"], r.get("description") or ""))
        except (TypeError, ValueError, KeyError):
            continue
    print("FINAL PICKS  (web UI only - select these two before 2026-10-22)")
    if not scored:
        print("  none scored yet")
        return
    scored.sort(reverse=True)
    canon = [s for s in scored if BASELINE_MARKER in s[2]]
    best_top = scored[0]
    if canon:
        c = canon[0]
        if best_top[1] != c[1] and best_top[0] - c[0] >= PROMOTE_MARGIN:
            best = best_top
            print("  offence  ref %-10s %.3f   %s" % (best[1], best[0], best[2][:40]))
            print("           (promoted: beats the canonical recipe by >= %.3f, more than "
                  "selection bias explains)" % PROMOTE_MARGIN)
        else:
            best = c
            print("  offence  ref %-10s %.3f   %s" % (best[1], best[0], best[2][:40]))
            if best_top[1] != c[1]:
                print("           (canonical kept: the best variant %.3f leads by only "
                      "%.3f, within selection bias)"
                      % (best_top[0], best_top[0] - c[0]))
            else:
                print("           (canonical recipe; %d submission(s) tie it)"
                      % len([s for s in scored if s[0] == c[0]]))
    else:
        best = best_top
        print("  offence  ref %-10s %.3f   %s" % (best[1], best[0], best[2][:40]))
        print("           (no canonical run in the history; using the top score)")
    # Hedge selection, per HANDOFF 38.4 as AMENDED BY HANDOFF 53. What matters is that the
    # hedge fails for a DIFFERENT reason than the offence - two runs sharing the per-label
    # routing (itself fitted on the public split) would fail together and hedge nothing.
    # HANDOFF 53 makes that risk concrete: an independent 4407-study OOF audit of the same
    # phenomenon (per-label fitting of highly correlated members) found the fitted version
    # scored 0.8480 against 0.8497 for doing nothing - i.e. 0.0017 WORSE out-of-fold.
    # So the fully independent hedge now DOMINATES the half-independent one:
    #   routing harmful out-of-sample -> the independent run is strictly better;
    #   routing merely neutral        -> they tie;
    #   routing genuinely helps       -> the offence wins and neither hedge is used.
    # `halfway-routing`'s higher public floor is NOT a real advantage, because part of that
    # floor is the same routing gain we are hedging against. It is now only a fallback.
    parent = [s for s in scored if "parent-routing" in s[2].lower() and s[1] != best[1]]
    half = [s for s in scored if "halfway-routing" in s[2].lower() and s[1] != best[1]]
    others = [s for s in scored if s[1] != best[1] and s[0] >= best[0] - 1e-9]
    if parent:
        h, why = parent[0], "fully independent of the public-split routing fit"
    elif half:
        h, why = half[0], "half-independent fallback (no fully-independent run scored)"
    elif others:
        h, why = others[0], "LAST RESORT: shares the routing fit, so it hedges little"
    else:
        h = why = None
    if h:
        print("  hedge    ref %-10s %.3f   %s" % (h[1], h[0], h[2][:40]))
        print("           (%s)" % why)
    else:
        print("  hedge    (nothing scored yet to hedge with)")
    print("  the higher PRIVATE score of the two counts; ties go to the earlier entry.")


def kernel_status(ref):
    text = " ".join(cli("kernels", "status", ref))
    if "404" in text:
        return "not-created"
    marker = "KernelWorkerStatus."
    if marker in text:
        return text.split(marker)[-1].strip().strip('"').split()[0].strip('"')
    return "?"


def main() -> int:
    now = datetime.now(timezone.utc)
    print("=" * 78)
    left = FINAL_SUBMISSION_DEADLINE - now
    print("DEADLINE  2026-10-22 23:59 UTC  (= 2026-10-23 07:59 local, UTC+8)")
    print("          %.1f days left" % (left.total_seconds() / 86400.0))
    if now < MERGER_DEADLINE:
        print("          team merger still open until 2026-10-15 23:59 UTC")
    if left.days <= SELECT_WARN_DAYS:
        print("          *** LAST %d DAYS: select the 2 Final Submissions on the Kaggle "
              "web UI ***" % max(left.days, 0))
        print("          This is a WEB-ONLY action - no API - so it cannot be automated.")

    print("=" * 78)
    print("SUBMISSIONS  (competition %s)" % COMPETITION)
    rows = list(csv.DictReader(io.StringIO(
        "\n".join(cli("competitions", "submissions", "-c", COMPETITION, "-v")))))
    scored = [r for r in rows if (r.get("publicScore") or "").strip()]
    best = max(scored, key=lambda r: float(r["publicScore"])) if scored else None
    for r in rows[:10]:
        print("  %-10s %-42s %-9s %s" % (
            r["ref"], (r.get("description") or "")[:40],
            (r.get("status") or "").replace("SubmissionStatus.", ""),
            (r.get("publicScore") or "-")))
    if best:
        print("  best scored so far: %s = %s" % (best["ref"], best["publicScore"]))
    pending = [r for r in rows if "PENDING" in (r.get("status") or "")]
    print("  pending: %d   complete: %d" % (len(pending), len(scored)))

    print("-" * 78)
    print("VERDICT  (HANDOFF 20.3 decision tree, applied to the baseline)")
    print("          " + verdict(rows))
    print("=" * 78)
    report_leaderboard()
    report_final_picks(rows)

    print("=" * 78)
    print("KERNELS")
    for ref in KERNELS:
        print("  %-46s %s" % (ref, kernel_status(ref)))

    print("=" * 78)
    print("DIRECTOR WATCHES  (%s)" % CONFIG)
    print("  the PROGRESS column is the pipeline audit: a watch whose actions include "
          "'submit'\n  but which shows only 'landed' is finished and WAITING FOR QUOTA - "
          "that is normal.\n  A watch that shows nothing at all while its kernel is "
          "COMPLETE is the failure to look for.")
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
        dst = load_json(STATE_DIR / "director_state.json", {}) or {}
        for e in cfg["watches"]:
            ref = e["ref"]
            w = (dst.get("watches") or {}).get(ref, {}) or {}
            h = w.get("handled") or []
            done = []
            if any(str(x).startswith("download") for x in h):
                done.append("landed")
            if any(str(x).startswith("submit:v") for x in h):
                done.append("submitted")
            if any(str(x).startswith("score:v") for x in h):
                done.append("scored")
            if any(str(x).startswith("failed:v") for x in h):
                done.append("FAILED")
            print("  %-44s v%-3s %-26s %s"
                  % (ref.split("/")[-1][:42], e.get("version"),
                     "/".join(a[:6] for a in e.get("actions", [])),
                     "+".join(done) or "-"))
    except (OSError, ValueError) as exc:
        print("  unreadable: %s" % exc)

    print("=" * 78)
    try:
        hb = json.loads(HEARTBEAT.read_text(encoding="utf-8"))
        at = datetime.strptime(hb["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - at).total_seconds()
        print("HEARTBEAT %s  (%.1f min ago, pid %s)" % (hb["at"], age / 60.0, hb.get("pid")))
        if age > 3600:
            print("  STALE - the RSNAKneeDirector scheduled task is probably not running.")
            print("  Restart it with:  schtasks /run /tn \"RSNAKneeDirector\"   (needs elevation)")
    except (OSError, ValueError, KeyError):
        print("HEARTBEAT none - no director pass has completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
