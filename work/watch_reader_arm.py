"""Watch the reader arm's PREVIEW run and report whether the reader leg actually worked.

The push run executes on the 3-study public preview, so it completes in minutes -- which means the
reader leg's success or failure is knowable BEFORE spending a submission on a 7-hour server-side
re-run. That is the whole reason to look at this log now rather than after submitting.

⚠️ A silent no-op is the failure mode to rule out: if the reader leg throws, the code RESTORES the
stack's submission and prints a FAILED line, so the presence of `reader leg: BLENDED` is the evidence,
not the absence of an error.
"""
import json
import pathlib
import datetime as _dt
import time

from kaggle.api.kaggle_api_extended import KaggleApi

REF = "ppxl16/rsna-knee-0943cfmsft2-rdr"
SLUG = "rsna-knee-0943cfmsft2-rdr"
WORK = pathlib.Path("work/klog_rdr")
OUT = pathlib.Path("state/reader_arm_preview.json")

api = KaggleApi()
# ⚠️ RETRY THE AUTH CALL ITSELF, not just the import. This line authenticates ONCE at module scope, before
# any polling loop, so a transient api.kaggle.com SSLEOFError here is instant death -- and for a watcher that
# means the result it waited hours for is never read. See work/patch_auth_guard.py.
# Self-contained on purpose: it imports what it needs, so it does not depend on the host file's imports.
import sys as _auth_sys
import time as _auth_time

_auth_last = None
for _auth_try in range(6):
    try:
        api.authenticate()
        break
    except Exception as _auth_e:  # noqa: BLE001
        _auth_last = _auth_e
        _auth_sys.stderr.write("api.authenticate() attempt %d failed: %s\n"
                               % (_auth_try + 1, type(_auth_e).__name__))
        _auth_sys.stderr.flush()
        _auth_time.sleep(8)
else:
    _auth_sys.stderr.write("COULD NOT AUTHENTICATE after 6 attempts (%r) -- TRANSPORT failure, "
                           "not a result.\n" % (_auth_last,))
    raise SystemExit(2)


def status():
    for _ in range(3):
        try:
            return str(getattr(api.kernels_status(REF), "status", ""))
        except Exception:  # noqa: BLE001
            time.sleep(8)
    return ""


def fetch():
    WORK.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        try:
            api.kernels_output(REF, path=str(WORK), file_pattern="%s.log" % SLUG, force=True, quiet=True)
            h = sorted(WORK.rglob("*.log"))
            if h:
                raw = h[0].read_text(encoding="utf-8", errors="replace")
                out = []
                for ln in raw.splitlines():
                    ln = ln.lstrip(",").strip()
                    if not ln:
                        continue
                    try:
                        d = json.loads(ln).get("data")
                    except Exception:
                        d = None
                    if d:
                        out.append(d)
                return "".join(out)
        except Exception:  # noqa: BLE001
            time.sleep(12)
    return ""


print("watching %s ..." % REF, flush=True)
for _ in range(60):                       # ~1 h at 60 s
    st = status()
    if st.endswith(("COMPLETE", "ERROR", "CANCEL")):
        break
    time.sleep(60)

st = status()
log = fetch()
print("status:", st, "| log chars:", len(log))

rec = {"ref": REF, "status": st, "log_chars": len(log),
       "at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
if log:
    rec["reader_lines"] = [l.strip() for l in log.splitlines() if "reader leg" in l]
    rec["blended"] = any("reader leg: BLENDED" in l for l in log.splitlines())
    rec["failed"] = any("reader leg FAILED" in l for l in log.splitlines())
    rec["stack_complete"] = "COMPLETE: /kaggle/working/submission.csv" in log
    rec["elapsed_s"] = max([int(m) for m in __import__("re").findall(r"\[\s*(\d+)s\]", log)] or [0])
else:
    rec["log"] = "UNREAD (transport) -- NOT a result"

OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(json.dumps(rec, indent=2))
