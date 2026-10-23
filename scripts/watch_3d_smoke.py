"""Watch the 3D volumetric SMOKE to a verdict and record the two numbers the next step depends on.

WHY IT EXISTS. `ppxl16/rsna-knee-tpu-3dsmoke` answers ONE question -- does a Conv3d stack compile under
PJRT, step, and stay inside HBM -- and it is the cheapest step on the only inductive bias this repository
has never tried (FACTS A160: the gate's base contains no CoAtNet/Raptor, the whole frontier stack is 2D, so
a real volume is the one differencing axis). Per `docs/PREREG-2026-10-06q-3d-smoke.md`:

    PASS  -> scale the SAME file into a real arm (width/depth/hw/studies are flags for exactly that)
    FAIL  -> the axis closes at minimum cost, and we say so instead of spending TPU on it

⚠️ It records `median_step_s` and the HBM shape, NOT just the verdict, because the verdict alone cannot tell
us whether a real arm is feasible: if a step costs tens of seconds, 4407 studies will never fit a 9 h
session and PASS would be misleading.

⚠️ A queue that never drains is recorded as NOT READ. It is never folded into FAIL -- failing to ask is not
an answer, which this repository has had to relearn several times.
"""
from __future__ import annotations

# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass
# ----------------------------------------------------------------------------------------------------
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
    except Exception as _e:  # noqa: BLE001  -- the import itself authenticates
        _sys.stderr.write("import try %d: %s\n" % (_i + 1, type(_e).__name__))
        _sys.stderr.flush()
        time.sleep(8)
if K is None:
    raise SystemExit("could not import KaggleApi -- TRANSPORT, not a result")

REF = "ppxl16/rsna-knee-tpu-3dsmoke"
WORK = pathlib.Path("work/klog_3dsmoke")
OUT = pathlib.Path("state/smoke3d_result.json")

api = K()
_last = None
for _t in range(8):
    try:
        api.authenticate()
        break
    except Exception as _e:  # noqa: BLE001
        _last = _e
        time.sleep(8)
else:
    raise SystemExit("could not authenticate (%r) -- TRANSPORT, not a result" % (_last,))


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
            api.kernels_output(REF, path=str(WORK), file_pattern="rsna-knee-tpu-3dsmoke.log",
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
                return "".join(out)
        except Exception:  # noqa: BLE001
            time.sleep(12)
    return ""


print("watching %s ..." % REF, flush=True)
for _ in range(120):                       # up to 30 h at 15-min polls; TPU queues have hit ~26 h
    st = status()
    print("[%s] %s" % (time.strftime("%H:%M:%S"), st), flush=True)
    if st.endswith(("COMPLETE", "ERROR", "CANCEL")):
        break
    time.sleep(900)

st = status()
log = fetch()
rec = {"ref": REF, "status": st, "log_chars": len(log),
       "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
if log:
    rec["lines"] = [l.strip() for l in log.splitlines() if "[3d]" in l][:40]
    ms = re.search(r"median step ([\d.]+)s", log)
    rec["median_step_s"] = float(ms.group(1)) if ms else None
    mm = re.search(r"model: Conv3d stack width=(\d+), ([\d.]+)M params, input \(([^)]*)\)", log)
    if mm:
        rec["width"] = int(mm.group(1))
        rec["params_m"] = float(mm.group(2))
        rec["input"] = mm.group(3)
    rec["passed"] = "[3d] VERDICT: PASS" in log
    rec["failed"] = "[3d] VERDICT: FAIL" in log or "!! " in log
    rec["errors"] = re.findall(r"^.*(Traceback|Error|error:|!!).*$", log, re.M)[:6]
    # ⚠️ the verdict alone does not decide feasibility -- cost per step does
    if rec["passed"] and rec["median_step_s"]:
        per_study = rec["median_step_s"] / 2.0          # bs=2 in the smoke
        rec["extrapolated_4407_hours"] = round(per_study * 4407 / 3600.0, 2)
    rec["decision"] = ("PASS -> the mechanism works; scale the same file into a real 3D arm, whose local "
                       "gate is solo >= 0.90 on a LARGE-n instrument (gold-58's CI is +/-0.021 and cannot "
                       "judge that line), against msft2's 0.8847 on the same instrument"
                       if rec.get("passed") else
                       "FAIL -> the 3D axis closes at minimum cost; do NOT spend TPU on it"
                       if rec.get("failed") else
                       "no verdict line in the log -> NOT READ, and this is not a rejection")
else:
    rec["log"] = "UNREAD (transport) -- NOT a result"

OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(json.dumps(rec, indent=2))
print("wrote", OUT)
