"""Watch the 3D volumetric ARM to its held-out macro-AUC and apply the pre-registered gate.

The gate is fixed in docs/PREREG-2026-10-06q-3d-smoke.md section 4 and is NOT adjustable after the fact:
solo macro-AUC >= 0.90 on the LARGE-n instrument (the held-out fold), against msft2's 0.8847 measured on the
same instrument. gold-58 cannot judge this line -- its CI is +/-0.021, wider than the whole band.

A run that never starts, or whose log cannot be read, is recorded as NOT READ. It is never folded into a
verdict: failing to ask is not an answer.
"""
from __future__ import annotations

import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import datetime as dt
import json
import pathlib
import re
import time

K = None
for _i in range(8):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi as K

        break
    except Exception as _e:  # noqa: BLE001
        _sys.stderr.write("import try %d: %s\n" % (_i + 1, type(_e).__name__))
        time.sleep(8)
if K is None:
    raise SystemExit("could not import KaggleApi -- TRANSPORT, not a result")

REF = "ppxl16/rsna-knee-tpu-3darm"
WORK = pathlib.Path("work/klog_3darm")
OUT = pathlib.Path("state/arm3d_result.json")

api = K()
for _t in range(8):
    try:
        api.authenticate()
        break
    except Exception:  # noqa: BLE001
        time.sleep(8)


def status():
    for _ in range(3):
        try:
            return str(getattr(api.kernels_status(REF), "status", ""))
        except Exception:  # noqa: BLE001
            time.sleep(8)
    return ""


print("watching %s ..." % REF, flush=True)
for _ in range(160):                      # up to 40 h at 15-min polls
    st = status()
    print("[%s] %s" % (time.strftime("%H:%M:%S"), st), flush=True)
    if st.endswith(("COMPLETE", "ERROR", "CANCEL")):
        break
    time.sleep(900)

st = status()
WORK.mkdir(parents=True, exist_ok=True)
log = ""
for _ in range(4):
    try:
        api.kernels_output(REF, path=str(WORK), file_pattern="rsna-knee-tpu-3darm.log",
                           force=True, quiet=True)
        hits = sorted(WORK.rglob("*.log"))
        if hits:
            raw = hits[0].read_text(encoding="utf-8", errors="replace")
            out = []
            for ln in raw.splitlines():
                ln = ln.lstrip(",").strip()
                if not ln:
                    continue
                try:
                    d = json.loads(ln).get("data")
                except Exception:  # noqa: BLE001
                    d = None
                if d:
                    out.append(d)
            log = "".join(out)
            break
    except Exception:  # noqa: BLE001
        time.sleep(12)

rec = {"ref": REF, "status": st, "log_chars": len(log),
       "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
if log:
    rec["lines"] = [l.strip() for l in log.splitlines() if "[3darm]" in l][:50]
    m = re.findall(r"HELD-OUT fold (\d+) macro-AUC ([\d.]+)\s+\(n=(\d+)\)", log)
    if m:
        rec["held_fold"] = int(m[-1][0])
        rec["macro_auc"] = float(m[-1][1])
        rec["n_val"] = int(m[-1][2])
        rec["history"] = [[int(x[0]), float(x[1])] for x in m]
    rec["errors"] = re.findall(r"^.*(Traceback|Error|!!).*$", log, re.M)[:6]
    a = rec.get("macro_auc")
    rec["decision"] = (
        "MACRO-AUC %.4f >= 0.90 -> CLEARS THE MEMBER BAR (msft2 is 0.8847 on this instrument); worth wiring "
        "as a member and testing one arm" % a if a is not None and a >= 0.90 else
        "MACRO-AUC %.4f < 0.90 -> below the member bar; record honestly and do NOT wire it" % a
        if a is not None else
        "no held-out AUC line in the log -> NOT READ, and this is not a rejection")
else:
    rec["log"] = "UNREAD (transport) -- NOT a result"

OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(json.dumps(rec, indent=2))
print("wrote", OUT)
