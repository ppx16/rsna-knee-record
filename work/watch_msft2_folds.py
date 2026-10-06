"""Watch `ppxl16/rsna-knee-msft2-f1` to completion, then MEASURE what the prereg needs.

The prereg (`docs/PREREG-2026-10-05-five-fold-msft2.md` §3) makes fold 1 do double duty: it is one of
the folds we need AND it is the cost measurement. "~2.6 h per fold" is inherited from a ConvNeXt
estimate and the CoAtNet fold cost has never been measured directly, so this extracts it from the run's
own log rather than accepting the inherited number.

Deliberately NOT a busy-poll loop in the foreground: TPU queueing has been observed at ~26 h, and
`A33` records that queueing does not consume quota, so waiting costs wall-clock only.

⚠️ Transport errors are retried, never treated as terminal -- and the script REFUSES to report a
verdict it did not earn: an unreadable log is reported as unread, not as a result.
"""

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
import json
import pathlib
import re
import datetime as _dt
import time

from kaggle.api.kaggle_api_extended import KaggleApi

REF = "ppxl16/rsna-knee-msft2-f1"
SLUG = "rsna-knee-msft2-f1"
STATE = pathlib.Path("state")
WORK = pathlib.Path("work/klog_folds")
OUT_JSON = STATE / "msft2_f1_result.json"
POLL_S = 600
MAX_POLLS = 200          # ~33 h of wall clock; TPU queue has been seen at ~26 h

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
            time.sleep(10)
    return ""


def fetch_log():
    WORK.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        try:
            api.kernels_output(REF, path=str(WORK / SLUG), file_pattern="%s.log" % SLUG,
                               force=True, quiet=True)
            hits = sorted((WORK / SLUG).rglob("*.log"))
            if hits:
                raw = hits[0].read_text(encoding="utf-8", errors="replace")
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
            time.sleep(15)
    return ""


print("watching %s ..." % REF)
last = None
for i in range(MAX_POLLS):
    st = status()
    if st and st != last:
        print("[%s] %s" % (time.strftime("%H:%M:%S"), st), flush=True)
        last = st
    if st.endswith(("COMPLETE", "ERROR", "CANCEL")):
        break
    time.sleep(POLL_S)

st = status()
print("terminal status:", st)
log = fetch_log()
print("log chars:", len(log))

rec = {"ref": REF, "status": st, "log_chars": len(log), "at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

if log:
    # the log's own wall clock: the largest [  NNNNs] stamp
    stamps = [int(m.group(1)) for m in re.finditer(r"\[\s*(\d+)s\]", log)]
    if stamps:
        rec["elapsed_s"] = max(stamps)
        rec["elapsed_h"] = round(max(stamps) / 3600.0, 3)
    m = re.search(r"SWA over epochs \[([^\]]*)\] \| gold ([\d.]+)", log)
    if m:
        rec["swa_epochs"] = m.group(1)
        rec["swa_gold"] = float(m.group(2))
    m = re.search(r"TOPK epochs \[([^\]]*)\] \| best ([\d.]+) \| epoch-ensemble ([\d.]+)", log)
    if m:
        rec["topk_epochs"], rec["best"], rec["ens"] = m.group(1), float(m.group(2)), float(m.group(3))
    m = re.search(r"\[cv\] fold (\d+)/(\d+): train (\d+) \| OOF held-out (\d+)", log)
    if m:
        rec["fold"], rec["nfolds"], rec["n_train"], rec["n_oof"] = (
            int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
    rec["wrote_oof"] = sorted(set(re.findall(r"wrote ([^\s]*oof[^\s]*)", log, re.I)))[:6]
    rec["zeroshot_verdict"] = bool(re.search(r"\[zeroshot\] VERDICT: PASS", log))
    rec["head_loaded"] = bool(re.search(r"HEAD loaded: 8/8", log))
    rec["errors"] = re.findall(r"^.*(Traceback|Error:|error:).*$", log, re.M)[:4]
else:
    rec["log"] = "UNREAD (transport) -- this is NOT a result"

OUT_JSON.write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(json.dumps(rec, indent=2))
print("wrote", OUT_JSON)
