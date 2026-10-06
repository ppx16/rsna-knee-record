"""Keep the two Kaggle GPU slots busy with CV fold kernels -- safely, and idempotently.

Kaggle allows at most 2 concurrent batch GPU sessions; a third push is rejected with
"Maximum batch GPU session count of 2 reached".  A previous naive queue died on exactly that
error after successfully pushing one kernel, which silently left 8 kernels unpushed for hours
(HANDOFF 09-14 snapshot).  This one never dies on that error.

    python scripts/cv_queue.py --once        # one pass: fill whatever slots are free (use this)
    python scripts/cv_queue.py               # loop until every target is COMPLETE
    python scripts/cv_queue.py --status      # just print the table

State lives in state/cv_queue.json so the script is safe to re-run after a reboot: a kernel
already pushed and not ERRORed is never pushed twice.
"""
from __future__ import annotations


# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded
# `print("... \u26a0 ...")` raises UnicodeEncodeError at that line and kills the script MID-REPORT, while
# everything printed earlier is already on screen -- so the run looks complete. Self-contained on purpose:
# it imports `sys` itself and depends on nothing else in this tree.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream: leave it alone, prints still degrade
    pass
# ----------------------------------------------------------------------------------------------------
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = os.path.join(ROOT, "state", "cv_targets.json")
STATE = os.path.join(ROOT, "state", "cv_queue.json")
LOG = os.path.join(ROOT, "state", "cv_queue.log")
GPU_LIMIT = "Maximum batch GPU session count"
SLOTS = 2

# A push error that looks like a credential/network hiccup gets retried instead of permanently
# holding the target (see the comment at the failure branch).  These are the substrings that have
# actually been observed on this machine; anything else is treated as a real problem.
TRANSIENT = ("Maximum batch GPU session", "Maximum weekly GPU quota", "Save the token", "SSL",
             "EOF occurred", "timed out", "timeout", "Unauthorized", "401", "Connection",
             "Temporary failure", "read timeout", "RemoteDisconnected",
             # 403 Forbidden -- added 2026-09-20 after a server-side block on the account made
             # EVERY push fail with `403 ... kernels.KernelsApiService/SaveKernel`. Without this
             # the queue burned its attempts and set PERMANENT holds on k24-fold0,
             # ens-v2src-seedalt, norm-fold0 and the four g-fold controls -- i.e. a temporary
             # platform block silently deleted work from the schedule. That is A21's lesson
             # exactly: **the test is "does the failure look like it will fix itself", not
             # "did it fail".** A genuinely bad slug now costs a few extra retries before the
             # same hold, which is the cheap direction to be wrong in.
             "403", "Forbidden", "Permission", "denied")
MAX_ATTEMPTS = 6


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def say(msg: str) -> None:
    line = "[%s] %s" % (now(), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def save_json(path, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def kaggle(args, timeout=300):
    """Run the kaggle CLI; return (rc, stdout+stderr)."""
    exe = os.environ.get("KAGGLE_EXE", "kaggle")
    return subprocess_run([exe] + args, timeout=timeout)


def subprocess_run(cmd, timeout=300):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except FileNotFoundError:
        return 127, "not found: %s" % cmd[0]


def status_of(ref: str) -> str:
    """COMPLETE | RUNNING | QUEUED | ERROR | CANCEL | MISSING"""
    rc, out = kaggle(["kernels", "status", ref])
    if "COMPLETE" in out:
        return "COMPLETE"
    if "RUNNING" in out:
        return "RUNNING"
    if "QUEUED" in out:
        return "QUEUED"
    if "ERROR" in out:
        return "ERROR"
    if "CANCEL" in out:
        return "CANCEL"
    if "404" in out or "Not Found" in out or "denied" in out:
        return "MISSING"
    return "UNKNOWN"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single pass, then exit")
    ap.add_argument("--status", action="store_true", help="print the table and exit")
    ap.add_argument("--sleep", type=int, default=900, help="seconds between passes in loop mode")
    ap.add_argument("--max-pushes", type=int, default=0, help="stop after N pushes (0 = no cap)")
    a = ap.parse_args(argv)

    targets = load_json(TARGETS, [])
    if not targets:
        say("no targets in %s" % TARGETS)
        return 1
    st = load_json(STATE, {"pushed": {}, "complete": {}})
    st.setdefault("pushed", {})
    st.setdefault("complete", {})

    pushes = 0
    while True:
        live = []
        for t in targets:
            ref = t["ref"]
            if ref in st["complete"]:
                continue
            s = status_of(ref)
            if s == "COMPLETE":
                st["complete"][ref] = now()
                save_json(STATE, st)
                say("COMPLETE  %-42s %s" % (ref, t.get("note", "")))
                # Fetch the output of a finished TRAINING kernel automatically.  The director
                # does this for the ensemble variants it watches, but the queue deliberately does
                # NOT register training kernels with the director (they produce no submission.csv,
                # see the guard further down) -- so without this the fold outputs would only ever
                # appear if a human ran `kaggle kernels output` by hand.  With the queue on a
                # 20-minute timer the next session then finds downloads/cv/<tag>/ already populated
                # and can score immediately.
                # ⚠️ uses t["dir"], NOT the bare name `d` -- `d` is only bound inside the push
                # loop further down, so referring to it here raised UnboundLocalError on the
                # FIRST target that ever completed (2026-09-15 02:11, cv-a-fold0).  Exactly the
                # bug class this branch's own comment warns about: code that cannot be reached
                # by a smoke test, and fires only at the moment you least want it to.
                if not t.get("dir", "").startswith("kernel-exp"):
                    # Directory name -> the tag the kernel actually used, so the folder matches
                    # the tag convention other tools expect:
                    #   kernel-cv-a-fold0   -> cva0     (arm kernels use the cv<arm><fold> tag)
                    #   kernel-cv-gap-fold0 -> gap0     (variant kernels use <variant><fold>)
                    base = os.path.basename(t.get("dir", ""))
                    m2 = re.match(r"kernel-cv-(?P<name>.+)-fold(?P<fold>\d+)$", base)
                    tag = (("cv%s%s" % (m2.group("name"), m2.group("fold")))
                           if m2 and m2.group("name") in ("a", "b", "g")
                           else ("%s%s" % (m2.group("name"), m2.group("fold")))
                           if m2 else base.replace("kernel-cv-", ""))
                    dst = os.path.join("downloads", "cv", tag)
                    os.makedirs(dst, exist_ok=True)
                    rc4, out4 = kaggle(["kernels", "output", ref, "-p", dst], timeout=1800)
                    n_files = len(glob.glob(os.path.join(dst, "*")))
                    say("FETCHED   %-42s %d file(s) -> %s%s"
                        % (ref, n_files, dst.replace("\\", "/"),
                           "" if rc4 == 0 else "  (rc=%d)" % rc4))
                continue
            if s in ("RUNNING", "QUEUED"):
                live.append((ref, s))
        save_json(STATE, st)

        free = SLOTS - len(live)
        if a.status:
            for t in targets:
                ref = t["ref"]
                mark = ("done" if ref in st["complete"] else
                        "pushed" if ref in st["pushed"] else "todo")
                print("%-10s %-42s %s" % (mark, ref, t.get("note", "")))
            print("live=%d free=%d" % (len(live), free))
            return 0

        say("live=%d free=%d | %s" % (len(live), free,
                                      ", ".join("%s(%s)" % (r.split("/")[-1], s) for r, s in live)))

        progressed = False
        live_refs = {r for r, _s in live}
        if free > 0:
            for t in targets:
                if free <= 0:
                    break
                ref, d = t["ref"], t["dir"]
                if ref in st["complete"]:
                    continue
                # ⚠️ 2026-09-14 BUG FIX.  The first version counted free slots from the live
                # kernels but then pushed from the TOP of the target list without skipping the
                # ones already running -- so the moment a slot freed it RE-PUSHED the highest
                # priority kernel that was still running, and Kaggle cancels the old version
                # when a new one is pushed.  It did exactly that to cv-a-fold0 at 12:19:50,
                # throwing away ~3.5 h of a 2.8 h run.  The `--status` table had been calling
                # already-pushed kernels "todo" for hours; that disagreement was the symptom,
                # not a cosmetic detail.  Two guards now: never push something that is live,
                # and never push something this queue has already pushed once.
                if ref in live_refs:
                    say("LIVE      %-42s (already running -- NOT re-pushing)" % ref)
                    continue
                if ref in st["pushed"]:
                    say("ALREADY   %-42s (pushed %s -- not pushing again)"
                        % (ref, st["pushed"][ref]))
                    continue
                if t.get("hold"):
                    say("HOLD      %-42s %s" % (ref, t.get("hold")))
                    continue
                # Kernels live under kernels/ since the 2026-09-26 reorganisation, while
                # state/cv_targets.json still stores the old relative paths.  Resolve at use time
                # instead of rewriting the data: that keeps older configs working and does not have
                # to be redone at the next move.  The string checks above still see the stored value.
                _dp = os.path.join("kernels", d)
                rc, out = kaggle(["kernels", "push", "-p", _dp if os.path.isdir(_dp) else d],
                                 timeout=900)
                tail = out.strip().splitlines()[-1] if out.strip() else "(no output)"
                if "successfully pushed" in out:
                    st["pushed"][ref] = now()
                    save_json(STATE, st)
                    say("PUSHED    %-42s v%s" % (ref, t.get("note", "")))
                    # Register the new version with the director, otherwise the kernel runs and
                    # lands but NOTHING ever downloads or submits it.  That step used to be a
                    # manual `director.py --record-push REF:VERSION` and is exactly the kind of
                    # thing that gets forgotten across a reboot; the push output states the
                    # version, so parse it and do it here.
                    #
                    # ONLY for ensemble variants (kernel-exp/...) -- those are the kernels that
                    # write a submission.csv and are meant to be scored.  A CV *training* fold
                    # produces no submission at all; registering one would send the director
                    # looking for a submission that does not exist.
                    if d.replace("\\", "/").startswith("kernel-exp/"):
                        m = re.search(r"Kernel version (\d+) successfully pushed", out)
                        if m:
                            rc3, out3 = subprocess_run(
                                [sys.executable, os.path.join("scripts", "director.py"),
                                 "--record-push", "%s:%s" % (ref, m.group(1))])
                            _ = rc3
                            say("RECORDED  %-42s v%s -> %s"
                                % (ref, m.group(1),
                                   (out3 or "").strip().splitlines()[-1][:90] if out3 else "?"))
                    else:
                        say("SKIP-REG  %-42s (training kernel: no submission.csv by design)" % ref)
                    free -= 1
                    progressed = True
                    pushes += 1
                    time.sleep(15)          # let Kaggle register the session
                elif GPU_LIMIT in out:
                    say("GPU FULL  %-42s -> stop this pass" % ref)
                    free = 0
                else:
                    # ⚠️ 2026-09-14: the first version set a PERMANENT `hold` on ANY push error.
                    # When the Kaggle token rotated, every push failed once with
                    # "Save the token to ~/.kaggle/access_token" and the queue then permanently
                    # blocked 12 targets -- including the p0 variant that has to land before the
                    # 00:00 UTC submission window.  A transient credential/network hiccup must not
                    # be indistinguishable from a genuinely broken kernel.
                    say("PUSH FAIL %-42s %s" % (ref, tail[:160]))
                    t["attempts"] = int(t.get("attempts", 0)) + 1
                    if any(k in out for k in TRANSIENT) and t["attempts"] < MAX_ATTEMPTS:
                        say("RETRY     %-42s attempt %d/%d -- transient, will try again next pass"
                            % (ref, t["attempts"], MAX_ATTEMPTS))
                    else:
                        t["hold"] = "push failed after %d attempt(s): %s" % (t["attempts"], tail[:120])
                        say("HELD      %-42s %s" % (ref, t.get("hold")))
                    save_json(TARGETS, targets)

        if a.max_pushes and pushes >= a.max_pushes:
            say("hit --max-pushes=%d" % a.max_pushes)
            return 0
        if a.once:
            say("--once done (pushed %d)" % pushes)
            return 0
        if not progressed and free <= 0:
            time.sleep(a.sleep)
        elif not progressed and free > 0 and all(t["ref"] in st["complete"] or t.get("hold") for t in targets):
            say("nothing left to do")
            return 0


if __name__ == "__main__":
    sys.exit(main())
