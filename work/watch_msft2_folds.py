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
import json
import pathlib
import re
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
api.authenticate()


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

rec = {"ref": REF, "status": st, "log_chars": len(log), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ")}

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
