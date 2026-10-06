"""Is anything of ours running or queued on Kaggle right now?

Three traps, all of which produced a FALSE "NONE" before this script was written:

1. `kernels_list` rows carry NO status field at all (`_status` does not exist). Reading it
   yields `""` for every kernel -- which looks exactly like "nothing is running".
2. `kernels_status(ref)` returns an OBJECT, not a dict. Calling `.get("status")` on it raises
   AttributeError; if that is swallowed by a broad `except`, every kernel silently becomes
   UNKNOWN and the summary still prints a confident "NONE". Use the attribute.
3. `api.kaggle.com` intermittently dies with SSLEOFError mid-call. That is transport, not a
   terminal state -- retry, but do not let the retry wrapper manufacture a verdict.

A status probe that cannot distinguish "idle" from "I failed to ask" is worse than no probe.

A FOURTH false verdict, found while using it: `kernels_status` returns the string
`KernelWorkerStatus.RUNNING`, not `RUNNING`, so comparing against a bare {"RUNNING","QUEUED"} set
made every kernel look idle -- `ACTIVE: none` printed WHILE A KERNEL WAS RUNNING. Compare the
suffix after the last dot. (See is_active().)

⚠️ AND A POWERSHELL NOTE THAT HAS BEEN MISREAD ALL SESSION: piping this (or anything) through
`Select-Object -First N` makes PowerShell close the pipe early, and the python process then exits
with code 1 / -1 -- a BROKEN PIPE, not a failure. Measured: exit 0 without truncation, -1 with.
Several "pwsh went red" moments in this repo were this and nothing else. Read the OUTPUT, and if
the exit code is the only signal, re-run without truncation before believing it.
"""
import time

from kaggle.api.kaggle_api_extended import KaggleApi

ACTIVE = {"RUNNING", "QUEUED"}


def is_active(st: str) -> bool:
    """⚠️ `kernels_status` returns e.g. `KernelWorkerStatus.RUNNING`, NOT `RUNNING`.

    The first version of this file compared the raw string against a bare {"RUNNING","QUEUED"}
    and therefore printed a confident `ACTIVE: none` WHILE A KERNEL WAS RUNNING -- the fourth
    false verdict this one probe has produced. Compare the suffix after the last dot.
    """
    return st.rsplit(".", 1)[-1].strip().upper() in ACTIVE

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


def status(ref, tries=3):
    last = None
    for _ in range(tries):
        try:
            res = api.kernels_status(ref)
            st = getattr(res, "status", None)
            if st is None:
                return f"NO_STATUS_FIELD({type(res).__name__})"
            return str(st)
        except Exception as e:  # noqa: BLE001  (SSLEOFError and friends)
            last = e
            time.sleep(3)
    return f"UNASKED({type(last).__name__})"


ks = api.kernels_list(user="ppxl16", sort_by="dateRun", page_size=10)
print(f"{'ref':<52} {'last_run':<20} status")
active, unknown = [], []
for k in ks:
    st = status(k.ref)
    lr = str(getattr(k, "last_run_time", ""))[:19]
    if is_active(st):
        active.append(k.ref)
    elif st.startswith(("UNASKED", "NO_STATUS_FIELD")):
        unknown.append(k.ref)
    print(f"{k.ref:<52} {lr:<20} {st}{'   <== ACTIVE' if is_active(st) else ''}")

print()
print("ACTIVE :", active or "none")
print("UNKNOWN:", unknown or "none")
if unknown:
    print("=> INCONCLUSIVE: at least one kernel could not be asked. Do not read this as idle.")
else:
    print("=> every kernel answered; verdict above is earned.")
