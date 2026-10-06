"""Watch the TTA arm's PREVIEW and report whether Stack++'s source actually composes and runs.

This is plumbing verification, not an experiment read. What proves composition:
  * `reader leg (tta): 3 checkpoints; src=...; data=...`  -- the mounted Stack++ dataset was found;
  * the reader subprocess EXITING 0 (it imports `knee` and `preprocess` from beside it, so an import or
    interface mismatch fails there);
  * `wrote /kaggle/working/_own.csv`                                   -- the reader produced output;
  * `reader leg (tta): BLENDED ... Synovitis_w=0.08 other_w=0.30`      -- the PER-LABEL blend executed;
  * the ABSENCE of `reader leg (tta) FAILED`                            -- the fallback was not taken.

⚠️ As on the plain leg, `0.0000 of label cells moved` is meaningless on 3 studies and is not a failure: at
w=0.30 the stack's order survives unless the reader reverses it hard enough to overcome the weight.
"""
import datetime as dt
import json
import pathlib
import re
import time

from kaggle.api.kaggle_api_extended import KaggleApi

REF = "ppxl16/rsna-knee-0943cfmsft2-rdrtta"
SLUG = "rsna-knee-0943cfmsft2-rdrtta"
WORK = pathlib.Path("work/klog_rdrtta")
OUT = pathlib.Path("state/tta_arm_preview.json")

api = KaggleApi()
api.authenticate()


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
                    except Exception:
                        d = None
                    if d:
                        out.append(d)
                return "".join(out)
        except Exception:  # noqa: BLE001
            time.sleep(12)
    return ""


print("watching %s ..." % REF, flush=True)
for _ in range(60):                      # ~1 h
    if status().endswith(("COMPLETE", "ERROR", "CANCEL")):
        break
    time.sleep(60)

st = status()
log = fetch()
print("status:", st, "| log chars:", len(log))

rec = {"ref": REF, "status": st, "log_chars": len(log),
       "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
if log:
    rec["reader_lines"] = [l.strip() for l in log.splitlines() if "reader leg" in l]
    rec["wrote_own"] = "wrote /kaggle/working/_own.csv" in log
    rec["blended"] = any("reader leg (tta): BLENDED" in l for l in log.splitlines())
    rec["failed"] = any("reader leg (tta) FAILED" in l for l in log.splitlines())
    rec["stack_complete"] = "COMPLETE: /kaggle/working/submission.csv" in log
    rec["composed"] = bool(rec["blended"] and not rec["failed"])
    rec["errors"] = re.findall(r"^.*(Traceback|ModuleNotFoundError|ImportError|RuntimeError).*$", log, re.M)[:5]
else:
    rec["log"] = "UNREAD (transport) -- NOT a result"

OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(json.dumps(rec, indent=2))
