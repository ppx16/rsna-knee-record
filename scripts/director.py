# -*- coding: utf-8 -*-
"""Unattended director: poll Kaggle kernels, land their output, submit, keep score.

Why this exists
---------------
Kaggle runs our notebooks server-side, so training and inference keep going while
this machine is off. What does NOT keep going is the chain that turns a finished
kernel into a leaderboard entry: download output -> submit -> record the score.
That chain used to be manual, so a run finishing at 03:00 sat idle until someone
noticed. This script is that chain, and it is the only thing that has to be
restarted after a reboot.

Design rules (all learned the hard way, see HANDOFF.md)
-------------------------------------------------------
* IDEMPOTENT. Every (kernel ref, version, action) triple is recorded in
  state/director_state.json before it is considered done. Re-running the script
  after a crash, a reboot, or a double-launch never re-submits a version. The
  competition has a daily submission cap; a duplicate submit is a wasted day.
* NEVER DESTRUCTIVE. This script only reads Kaggle state and writes local files.
  It does not push notebook versions, because pushing a version cancels/destroys
  the in-flight version's output, and only the latest version stays downloadable.
  Pushing stays a deliberate, human-decided act.
* RESUMABLE AFTER REBOOT. `--once` does a single pass and exits, so a Windows
  scheduled task at logon can call it cheaply; `--loop` is for a long-running
  background session. Both share the same state file.
* LOUD BUT QUIET. Everything goes to state/director.log with a timestamp, and
  notable events (completed, submitted, scored, failed) are appended to
  state/INBOX.md as a short human-readable digest.

Measured CLI facts this script depends on (2026-09-11, this machine)
-------------------------------------------------------------------
* The installed `kaggle` (2.1.2) prints an "outdated version" warning to STDOUT
  *before* the CSV header. Any `DictReader` over raw stdout parses garbage, so
  run_cli() strips Warning:/blank lines first.
* `kaggle kernels status <ref>` prints `... has status "KernelWorkerStatus.X"`.
* `kaggle competitions submissions -c <comp> -v` prints the CSV with a blank line
  between rows. Column `ref` holds a numeric submission id, not a kernel ref.
* The CLI reaches Kaggle through the Clash system proxy because Python's
  urllib reads the Windows registry ProxyEnable/ProxyServer. curl does not.
* Large downloads from storage.googleapis.com intermittently die with
  SSLError(SSLEOFError(8)). Retrying is the fix; it is flaky, not size-gated.

Usage
-----
    python scripts/director.py --once            # one pass, then exit
    python scripts/director.py --loop            # poll forever
    python scripts/director.py --once --dry-run  # show what it *would* do
"""
import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
CONFIG_PATH = STATE_DIR / "director.json"
STATE_PATH = STATE_DIR / "director_state.json"
LOG_PATH = STATE_DIR / "director.log"
INBOX_PATH = STATE_DIR / "INBOX.md"
HEARTBEAT_PATH = STATE_DIR / "director.heartbeat"
PASSES_PATH = STATE_DIR / "director.passes.log"
LANDING = ROOT / "outputs" / "_landing"

COMPETITION = "rsna-knee-abnormality-detection"
TERMINAL = {"COMPLETE", "ERROR", "CANCELLED"}

# Watched by default when state/director.json is absent. `version` is the version
# this script believes is in flight; `director.py --record-push` updates it.
DEFAULT_CONFIG = {
    "poll_seconds": 300,
    "competition": COMPETITION,
    "watches": [
        {
            "ref": "ppxl16/rsna-knee-ensemble-v1",
            "label": "ensemble-v1 (0.941-class public stack)",
            "version": 2,
            "actions": ["download", "submit", "score"],
            "submit_message": "public 0.941-class ensemble stack (kernel_sources restored)",
        },
        {
            "ref": "ppxl16/rsna-knee-train-v1",
            "label": "GPU training (biomedclip branch)",
            "version": 5,
            "actions": ["download"],
        },
        {
            "ref": "ppxl16/rsna-knee-tpu-train-v1",
            "label": "TPU training (HBM decider)",
            "version": 20,
            "actions": ["download"],
        },
    ],
}


# --------------------------------------------------------------------------- io

def _stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    line = "[%s] %s" % (_stamp(), msg)
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def inbox(msg):
    """Append a short digest line the human can read without opening the log."""
    try:
        INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
        new = not INBOX_PATH.exists()
        with open(INBOX_PATH, "a", encoding="utf-8") as fh:
            if new:
                fh.write("# Director inbox\n\nNewest last. One line per notable event.\n\n")
            fh.write("- [%s] %s\n" % (_stamp(), msg))
    except OSError:
        pass


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def save_watch(ref, st):
    """Persist ONE watch's change without destroying the rest of the ledger.

    ⚠️ Why this exists (found 2026-09-12, scripts/test_director_state.py):
    the file's schema is {"watches": {ref: st}}, but the action helpers were handed
    only the per-watch sub-dict and called save_json(STATE_PATH, st) - a plain dump.
    So every action rewrote the whole state file as a BARE SUB-DICT, and only the
    pass-end save_json(STATE_PATH, state) put the structure back.

    Two consequences, both silent:
      1. Between an action and the pass-end save the file is corrupt, so a process
         killed in that window (the machine is shut down regularly) loses it. The
         next pass calls watches.setdefault(...) and rebuilds every other watch with
         an EMPTY `handled` list - so every action re-runs, INCLUDING RE-SUBMITTING
         and burning the 5-per-UTC-day quota.
      2. Any exception in a later watch's action skips the pass-end save entirely and
         makes (1) permanent.

    Re-reading and merging costs one small file read per action and makes each save
    independently safe.
    """
    state = load_json(STATE_PATH, {"watches": {}})
    if not isinstance(state, dict):
        state = {}
    state.setdefault("watches", {})[ref] = st
    save_json(STATE_PATH, state)


LOCK_PATH = STATE_DIR / "director.lock"
# A pass normally takes seconds; only a download makes it slow. Reclaiming on age alone
# costs a full hour of dead automation when a pass is killed mid-flight (which happened:
# a manual run cut off by a 600 s timeout left a lock behind and blocked four scheduled
# ticks), so liveness of the holding pid is checked first and age is only the fallback.
LOCK_STALE_SECONDS = 1800


def _pid_alive(pid):
    """True/False, or None when liveness cannot be determined on this platform."""
    try:
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    except Exception:
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False
        except Exception:
            return None


class Busy(Exception):
    """Another director pass holds the lock."""


class _Lock(object):
    """Exclusive lock so a background loop and a logon task never both submit.

    The state file's `handled` list makes submits idempotent, but two passes that
    read state inside the same window would both see the action as pending. The
    competition allows only a handful of submissions per day, so a duplicate is
    expensive; an O_EXCL lock file closes the window.
    """

    def __enter__(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        for attempt in (0, 1, 2):
            try:
                fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, ("%d %s\n" % (os.getpid(), _stamp())).encode())
                os.close(fd)
                return self
            except FileExistsError:
                pass
            holder, age = None, 0.0
            try:
                age = time.time() - LOCK_PATH.stat().st_mtime
                parts = LOCK_PATH.read_text(errors="replace").split()
                holder = int(parts[0]) if parts else None
            except (OSError, ValueError):
                pass
            if attempt < 2:
                alive = _pid_alive(holder) if holder else None
                if alive is False:
                    log("reclaiming director lock: holder pid %s is gone" % holder)
                elif alive is None and age > LOCK_STALE_SECONDS:
                    log("reclaiming director lock: unverifiable holder, %.0fs old" % age)
                elif alive and age > LOCK_STALE_SECONDS:
                    log("reclaiming director lock: holder pid %s alive but %.0fs stale"
                        % (holder, age))
                else:
                    raise Busy("held by pid %s, %.0fs old" % (holder, age))
                try:
                    LOCK_PATH.unlink()
                except OSError:
                    pass
                continue
            raise Busy("held by pid %s, %.0fs old" % (holder, age))

    def __exit__(self, *exc):
        try:
            LOCK_PATH.unlink()
        except OSError:
            pass
        return False


# ------------------------------------------------------------------ kaggle cli

def run_cli(args, timeout=900):
    """Run the kaggle CLI and return (returncode, cleaned_stdout_lines)."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"          # zh-CN Windows: CLI mis-reads UTF-8 without it
    try:
        proc = subprocess.run(
            ["kaggle"] + list(args),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, ["<timeout after %ss>" % timeout]
    except OSError as exc:
        return 127, ["<could not launch kaggle: %s>" % exc]
    raw = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    lines = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s or s.startswith("Warning: Looks like you're using an outdated"):
            continue
        lines.append(ln.rstrip())
    return proc.returncode, lines


STATUS_RE = re.compile(r'has status "KernelWorkerStatus\.([A-Z_]+)"')


def kernel_status(ref):
    """Return the KernelWorkerStatus name, or None if it could not be read."""
    rc, lines = run_cli(["kernels", "status", ref], timeout=120)
    for ln in lines:
        m = STATUS_RE.search(ln)
        if m:
            return m.group(1)
    log("  status unreadable for %s (rc=%s): %s" % (ref, rc, " | ".join(lines)[:200]))
    return None


def download_output(ref, dest, attempts=4, need=None):
    """Download a kernel's output files. Only the LATEST version is available.

    ⚠️ `kaggle kernels output` TRUNCATES SILENTLY on a proxy/SSL error and still exits 0.
    Measured 2026-09-11 on ppxl16/rsna-knee-ens-parent-routing: the first pull returned 3
    entries with no submission.csv, while the retry returned 135 files including it. A
    "did any file arrive" check therefore passes on a partial download - which is how the
    director concluded "no submission.csv" and, because it had already recorded the
    download as handled, would never have looked again.

    `need` names a file that must be present for the download to count as complete.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(1, attempts + 1):
        rc, lines = run_cli(["kernels", "output", ref, "-p", str(dest)], timeout=3600)
        got = [p for p in dest.rglob("*") if p.is_file() and p.stat().st_size > 0]
        have_need = (need is None) or any(p.name == need for p in got)
        if got and have_need:
            return True, dest, got
        why = ("%s never arrived" % need) if got else "nothing arrived"
        log("  output attempt %d/%d for %s: %s (rc=%s, %d file(s)): %s"
            % (i, attempts, ref, why, rc, len(got), " | ".join(lines[-2:])[:180]))
        if i < attempts:
            time.sleep(20 * i)
    return False, dest, []


SUBMISSIONS_HEADER = ["ref", "fileName", "date", "description", "status",
                      "publicScore", "privateScore"]


def my_submissions():
    """Return this account's submissions for the competition, newest first."""
    rc, lines = run_cli(["competitions", "submissions", "-c", COMPETITION, "-v"],
                        timeout=300)
    text = "\n".join(lines)
    if "fileName" not in text:
        log("  could not read submissions (rc=%s): %s" % (rc, text[:200]))
        return []
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        if row.get("fileName"):
            rows.append(row)
    return rows


def submit(ref, version, csv_path, message):
    """Submit a code competition entry through the Python API, not the CLI.

    Measured 2026-09-11, same arguments, same minute:
        kaggle competitions submit -c <comp> -k <ref> -v <n> -f submission.csv -m ...
            -> 400 Client Error: Bad Request for
               https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateCodeSubmission
               (and the CLI prints no response body, so the cause is invisible)
        KaggleApi.competition_submit_code(...) with the identical arguments
            -> success, submission ref 56166132
    The CLI path is simply not usable for code competitions on this version, so
    the API client is the only route. file_name must be the name of the file the
    kernel wrote (submission.csv), not a local path.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        return False, "kaggle package not importable: %s" % exc
    try:
        api = KaggleApi()
        api.authenticate()
        resp = api.competition_submit_code(
            file_name=Path(csv_path).name,
            message=message,
            competition=COMPETITION,
            kernel=ref,
            kernel_version=int(version),
            quiet=True,
        )
    except Exception as exc:                      # noqa: BLE001 - report, never raise
        body = ""
        r = getattr(exc, "response", None)
        if r is not None:
            raw = getattr(r, "data", None) or getattr(r, "body", None)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            body = " body=%s" % str(raw)[:300]
        return False, "%s: %s%s" % (type(exc).__name__, exc, body)

    def _get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    ref_id = _get(resp, "ref")
    return True, "ref=%s" % ref_id


# ------------------------------------------------------------------- watching

def load_config():
    cfg = load_json(CONFIG_PATH, None)
    if not cfg:
        cfg = DEFAULT_CONFIG
        save_json(CONFIG_PATH, cfg)
        log("wrote default config -> %s" % CONFIG_PATH)
    return cfg


def parse_csv_rows(path):
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def do_download(ref, version, entry, st):
    key = "download:v%s" % version
    if key in st["handled"]:
        return st.get("landing_dir")
    slug = ref.split("/")[-1]
    dest = LANDING / ("%s_v%s_%s" % (slug, version, _stamp().replace(":", "")))
    # An inference kernel is only useful once submission.csv is on disk, so require it.
    # Leaving the download UNHANDLED on failure is deliberate: the next pass retries.
    need = "submission.csv" if "submit" in entry.get("actions", []) else None
    ok, dest, files = download_output(ref, dest, need=need)
    if not ok:
        inbox("DOWNLOAD INCOMPLETE %s v%s (missing %s) - will retry next pass"
              % (ref, version, need or "files"))
        log("  download incomplete for %s v%s; NOT marking it handled" % (ref, version))
        return None
    log("  landed %d file(s) -> %s" % (len(files), dest))
    st["landing_dir"] = str(dest)
    st["handled"].append(key)
    save_watch(ref, st)
    inbox("landed %s v%s -> %s (%d files)" % (ref, version, dest.relative_to(ROOT), len(files)))
    return str(dest)


def do_submit(cfg, ref, version, entry, st):
    key = "submit:v%s" % version
    if key in st["handled"]:
        return True
    landing = st.get("landing_dir")
    if not landing:
        log("  nothing landed for %s v%s; skipping submit" % (ref, version))
        return False
    cands = sorted(Path(landing).rglob("submission.csv"))
    if not cands:
        log("  no submission.csv under %s" % landing)
        inbox("NO submission.csv in %s v%s output" % (ref, version))
        return False
    csv_path = cands[0]
    rows = parse_csv_rows(csv_path)
    if not rows:
        log("  submission.csv at %s is empty of rows" % csv_path)
        return False
    msg = entry.get("submit_message") or ("%s v%s" % (entry.get("label", ref), version))
    ok, detail = submit(ref, version, csv_path, msg)
    if not ok:
        # Kaggle caps this competition at 5 submissions per UTC day and the cap is hit by
        # a bare 400 with no response body. A watch whose submit is blocked by the cap is
        # retried every pass, which is correct, but the inbox must not get one identical
        # line every 15 minutes for the rest of the day.
        notice = "submitfail:v%s" % version
        log("  submit failed: %s" % detail)
        if notice not in st["handled"]:
            st["handled"].append(notice)
            save_watch(ref, st)
            inbox("SUBMIT BLOCKED %s v%s: %s (retrying every pass; the daily cap is 5 per "
                  "UTC day)" % (ref, version, detail[:160]))
        return False
    st["handled"].append(key)
    st["submitted_message"] = msg
    save_watch(ref, st)
    log("  submitted %s v%s (%d data rows) -> %s" % (ref, version, len(rows), detail))
    inbox("SUBMITTED %s v%s - %s (%s)" % (ref, version, msg, detail))
    return True


def our_rank():
    """Our current leaderboard position, or None.

    Why the director records this: the leaderboard ranks a TEAM by its best submission, so
    "did our rank move" only ever answers "is this our new best" - and once several variants
    all display 0.941 that is un-attributable, because the displayed score is rounded to
    three decimals while the rank is full precision (HANDOFF 24.2). Snapshotting the rank at
    the moment a score lands is what makes a rank movement attributable to THAT submission.
    """
    try:
        import csv as _csv
        import io as _io
        import zipfile
    except ImportError:
        return None
    lb_dir = ROOT / "downloads" / "lb"
    lb_dir.mkdir(parents=True, exist_ok=True)
    run_cli(["competitions", "leaderboard", COMPETITION, "-d", "-p", str(lb_dir)], timeout=300)
    zips = sorted(lb_dir.glob("*.zip"))
    if not zips:
        return None
    try:
        with zipfile.ZipFile(zips[0]) as z:
            rows = list(_csv.DictReader(_io.StringIO(
                z.read(z.namelist()[0]).decode("utf-8-sig", "replace"))))
    except (OSError, KeyError, zipfile.BadZipFile, ValueError):
        return None
    for r in rows:
        if "ppxl16" in (r.get("TeamMemberUserNames") or "") or \
                "ppxl16" in (r.get("TeamName") or ""):
            try:
                return int(r["Rank"])
            except (TypeError, ValueError):
                return None
    return None


RANK_TRACK = STATE_DIR / "rank_track.jsonl"


def track_rank():
    """Append the current leaderboard rank to a rolling log, once per pass.

    ⚠️ Why EVERY pass and not only when a score lands (which is what score_history does):
    this machine is switched off regularly, and a score that arrives while it is off gets
    its rank snapshotted whenever the next pass finally runs - contaminated by however
    much drift happened in between.  That is not hypothetical: on 2026-09-12 three
    submissions scored during a 4h20m outage and all three rank readings came back as
    158, which is unattributable.

    A dense series survives that.  Rank is a property of the TEAM's best submission, so a
    change between two consecutive passes can be attributed to whichever submission was
    pending in that window - the gap just widens when the machine is off.

    Never raises: our_rank() already returns None on most failures, but its leaderboard
    command is not inside a try, and a pass must not die because a rank probe did.
    """
    try:
        rank = our_rank()
    except Exception:                                   # noqa: BLE001 - see docstring
        return
    if rank is None:
        return
    try:
        with open(RANK_TRACK, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": _stamp(), "rank": rank}) + "\n")
    except OSError:
        pass


def do_score(cfg, ref, version, entry, st):
    """Attach the newest public score to this version, once, plus the rank it produced."""
    key = "score:v%s" % version
    if key in st["handled"]:
        return
    rows = my_submissions()
    if not rows:
        return
    msg = st.get("submitted_message")
    match = None
    for row in rows:
        if msg and row.get("description", "").strip() == msg.strip():
            match = row
            break
    if match is None:
        return                      # not scored yet, or the description did not round-trip
    score = (match.get("publicScore") or "").strip()
    if not score:
        # ⚠️ A FAILED submission also has an empty publicScore. Without this branch the
        # director waits forever for a score that is never coming and reports nothing - so a
        # timeout at the 9-hour limit would look exactly like "still queued". That is the
        # worst possible silence: the whole point of watching the 9h limit is to hear about
        # it.
        status = (match.get("status") or "").replace("SubmissionStatus.", "")
        if status and status.upper() not in ("PENDING", "COMPLETE"):
            notice = "failed:v%s" % version
            if notice not in st["handled"]:
                st["handled"].append(notice)
                st["submission_status"] = status
                save_watch(ref, st)
                inbox("SUBMISSION %s %s v%s - it will never score. If it is a timeout, "
                      "that is the 9h limit; a leaner variant is the only way to get a "
                      "number (HANDOFF 20.8)." % (status.upper(), ref, version))
                log("  submission status for %s v%s is %s - no score will arrive"
                    % (ref, version, status))
        return
    st["handled"].append(key)
    st["public_score"] = score
    rank = our_rank()
    st["public_rank"] = rank
    hist = st.setdefault("score_history", [])
    hist.append({"version": version, "score": score, "rank": rank, "at": _stamp()})
    save_watch(ref, st)
    log("  public LB score for %s v%s = %s (our rank now %s)" % (ref, version, score, rank))
    inbox("SCORE %s v%s = **%s** (public LB)  rank %s" % (ref, version, score, rank))


def poll_once(cfg, dry=False):
    state = load_json(STATE_PATH, {"watches": {}})
    watches = state.setdefault("watches", {})
    comp = cfg.get("competition", COMPETITION)

    # Liveness. Section 9 of HANDOFF records a watcher that died silently at 12:36 and
    # was not noticed for hours, because a healthy watcher and a dead one look identical
    # from the outside. Two artifacts make the difference visible:
    #   director.heartbeat   - overwritten every pass; `--status` reports its age
    #   director.passes.log  - append-only, one line per pass, so history survives a
    #                          manual `--once` overwriting the heartbeat file
    HEARTBEAT_PATH.write_text(json.dumps(
        {"at": _stamp(), "pid": os.getpid(), "kernels": len(cfg["watches"])},
        indent=2), encoding="utf-8")
    try:
        with open(PASSES_PATH, "a", encoding="utf-8") as fh:
            fh.write("%s pid=%d\n" % (_stamp(), os.getpid()))
    except OSError:
        pass
    track_rank()

    skipped = 0   # watches already finished for their version: no API call, see the skip below
    polled = 0    # watches that actually cost a `kernels status` call this pass
    for entry in cfg["watches"]:
        ref = entry["ref"]
        version = int(entry.get("version", 0))
        st = watches.setdefault(ref, {"handled": [], "version": version})
        st["version"] = version
        st.setdefault("handled", [])
        actions = entry.get("actions", ["download"])

        # ⚠️ 2026-09-20: ONLY POLL WHAT CAN STILL CHANGE.
        # Every pass used to call `kernels status` once per watch, and `cfg["watches"]` keeps
        # every kernel ever pushed -- about 25 calls every 15 minutes, forever, whether or not
        # anything was running. That volume is what got the account blocked at 10:40Z, right
        # after the tasks were re-enabled: EVERY read path returned
        #   403 Client Error: Forbidden for url: https://api.kaggle.com/v1/...
        # (and `Your client does not have permission to get URL ... from this server` on the
        # submissions endpoint) while pushes off the token FILE kept working -- which reads
        # like a permissions bug and is not one.
        # A finished watch cannot change, so it must not be polled. A later push calls
        # `--record-push`, which bumps `version`, which makes the watch outstanding again.
        done_keys = {"download": "download:v%s" % version,
                     "submit": "submit:v%s" % version,
                     "score": "score:v%s" % version}
        outstanding = [a for a in actions if done_keys.get(a) not in st["handled"]]
        if not outstanding and st.get("status") in TERMINAL:
            skipped += 1
            continue

        polled += 1     # count the ATTEMPT: a call that 403s still cost a call, and an
                        # under-counting diagnostic is exactly how this got missed
        status = kernel_status(ref)
        if status is None:
            continue
        previous = st.get("status")
        st["status"] = status
        st["last_poll"] = _stamp()
        if previous != status:
            log("%s -> %s (was %s)" % (ref, status, previous))
        if status not in TERMINAL:
            save_json(STATE_PATH, state)
            continue

        # A completed run is worth announcing exactly once.
        notice = "notice:%s:%s" % (version, status)
        if notice not in st["handled"]:
            st["handled"].append(notice)
            save_json(STATE_PATH, state)
            if status == "COMPLETE":
                inbox("COMPLETE %s v%s (%s)" % (ref, version, entry.get("label", "")))
            else:
                inbox("%s %s v%s (%s) - needs a look" % (status, ref, version,
                                                         entry.get("label", "")))
            log("%s v%s -> %s" % (ref, version, status))

        if dry:
            # Report only what would ACTUALLY run. The first version printed `actions`
            # verbatim, so it announced "would run ['download','submit','score']" for
            # experiments that had already been submitted - a diagnostic that lies is worse
            # than no diagnostic, because it reads as "the work is still pending".
            todo = []
            for a in actions:
                done_key = {"download": "download:v%s" % version,
                            "submit": "submit:v%s" % version,
                            "score": "score:v%s" % version}.get(a)
                if done_key and done_key in st["handled"]:
                    continue
                todo.append(a)
            log("  dry-run: %-46s v%s would run %s"
                % (ref.split("/")[-1], version, todo if todo else "nothing (all done)"))
            continue

        for action in actions:
            if action == "download":
                do_download(ref, version, entry, st)
            elif action == "submit":
                do_submit(cfg, ref, version, entry, st)
            elif action == "score":
                do_score(cfg, ref, version, entry, st)
            else:
                log("  unknown action %r" % action)
    log("API calls this pass: polled %d of %d watches (%d finished -> skipped)"
        % (polled, len(cfg["watches"]), skipped))
    save_json(STATE_PATH, state)
    return state


# ----------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--once", action="store_true", help="single pass, then exit")
    ap.add_argument("--loop", action="store_true", help="poll until killed")
    ap.add_argument("--interval", type=float, default=None,
                    help="seconds between passes (default: config poll_seconds)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report status and planned actions without touching Kaggle")
    ap.add_argument("--record-push", metavar="REF:VERSION",
                    help="tell the director a new version was pushed")
    ap.add_argument("--status", action="store_true", help="print state and exit")
    args = ap.parse_args(argv)

    if args.record_push:
        ref, _, ver = args.record_push.partition(":")
        cfg = load_config()
        hit = False
        for entry in cfg["watches"]:
            if entry["ref"] == ref:
                entry["version"] = int(ver)
                hit = True
        if not hit:
            cfg["watches"].append({"ref": ref, "version": int(ver),
                                   "actions": ["download", "submit", "score"]})
        save_json(CONFIG_PATH, cfg)
        log("recorded push %s v%s" % (ref, ver))
        return 0

    if args.status:
        state = load_json(STATE_PATH, {})
        hb = load_json(HEARTBEAT_PATH, None)
        if hb:
            try:
                from datetime import datetime as _dt
                at = _dt.strptime(hb["at"], "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - at).total_seconds()
                print("heartbeat: %s (%.1f min ago, pid %s)"
                      % (hb["at"], age / 60.0, hb.get("pid")))
                if age > 3600:
                    print("  ⚠️  STALE - the scheduled task is probably not running.")
                    print("      schtasks /run /tn \"RSNAKneeDirector\"   (needs elevation)")
            except (ValueError, KeyError):
                print("heartbeat: unreadable %r" % hb)
        else:
            print("heartbeat: NONE - no pass has ever completed")
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    cfg = load_config()
    interval = args.interval if args.interval is not None else cfg.get("poll_seconds", 300)

    if not args.loop:
        try:
            with _Lock():
                poll_once(cfg, dry=args.dry_run)
        except Busy as exc:
            log("another director pass is active (%s); nothing to do" % exc)
        return 0

    log("director looping every %ss (competition=%s)" % (interval, cfg.get("competition")))
    while True:
        try:
            with _Lock():
                poll_once(cfg, dry=args.dry_run)
        except Busy as exc:
            log("another director pass is active (%s); skipping this tick" % exc)
        except KeyboardInterrupt:
            log("interrupted; exiting")
            return 0
        except Exception as exc:                       # never die on a poll error
            log("poll error: %s: %s" % (type(exc).__name__, exc))
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
