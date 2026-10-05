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
"""
import time

from kaggle.api.kaggle_api_extended import KaggleApi

ACTIVE = {"RUNNING", "QUEUED"}

api = KaggleApi()
api.authenticate()


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
    if st in ACTIVE:
        active.append(k.ref)
    elif st.startswith(("UNASKED", "NO_STATUS_FIELD")):
        unknown.append(k.ref)
    print(f"{k.ref:<52} {lr:<20} {st}{'   <== ACTIVE' if st in ACTIVE else ''}")

print()
print("ACTIVE :", active or "none")
print("UNKNOWN:", unknown or "none")
if unknown:
    print("=> INCONCLUSIVE: at least one kernel could not be asked. Do not read this as idle.")
else:
    print("=> every kernel answered; verdict above is earned.")
