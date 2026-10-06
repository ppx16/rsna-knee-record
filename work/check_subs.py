"""Check the pending submission only (one API call, no board download)."""
from kaggle.api.kaggle_api_extended import KaggleApi

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
subs = api.competition_submissions("rsna-knee-abnormality-detection")
for s in sorted(subs, key=lambda s: str(getattr(s, "date", "")), reverse=True)[:5]:
    print(f"{getattr(s,'date','')}  ref={getattr(s,'ref','')}  "
          f"status={str(getattr(s,'status','')).split('.')[-1]:9s} "
          f"public={getattr(s,'publicScore',None)}  {str(getattr(s,'description',''))[:52]}")
