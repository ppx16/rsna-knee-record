"""Watch the distilled-checkpoint ZERO-SHOT GATE and read it against its pre-registered rule.

The gate answers ONE question: does `kawaharataishi/rsna-knee-coatnet-distilled-checkpoint` -- which
claims `auc 0.9231`, above our msft2's 0.9069 -- actually work under OUR geometry? Its geometry is
undocumented, which is why it was rejected last round; but that rejection is testable rather than final.

PRE-REGISTERED READING (written before the result, so it cannot be rationalised afterwards):

  gold-58 macro-AUC        verdict
  ---------------------    ---------------------------------------------------------------
  >= 0.92                  geometry matches AND the checkpoint is as strong as claimed
                           -> a real CoAt-family candidate; wire it as a member and test one arm
  0.90 - 0.92              geometry matches, checkpoint usable but no better than msft2 (0.9069)
                           -> weak candidate; only worth an arm if a cheap slot is going spare
  0.85 - 0.90              loadable but below the A167 member gate (solo >= ~0.90)
                           -> reject as a member
  < 0.85                   *** geometry mismatch *** -> reject permanently, and record that the
                           checkpoint's own `auc 0.9231` was measured under a different pipeline

⚠️ `[zeroshot] VERDICT: PASS/FAIL` is NOT the criterion here. That verdict compares against the v5
parent bar (0.8963), which is the wrong reference for a checkpoint from a different author. The number
to read is the gold macro-AUC itself.

⚠️ And a `--limit 64` run was used to keep it cheap: the limit only truncates `train_ids`, so the gold
set and k_eval are untouched and the number stays comparable (the flag's own comment says exactly this).
"""
import datetime as dt
import json
import pathlib
import re
import time

from kaggle.api.kaggle_api_extended import KaggleApi

REF = "ppxl16/rsna-knee-msft2-gate-distilled"
SLUG = "rsna-knee-msft2-gate-distilled"
WORK = pathlib.Path("work/klog_gate_distilled")
OUT = pathlib.Path("state/gate_distilled_result.json")

api = KaggleApi()
# ⚠️ A LONG-RUNNING WATCHER MUST SURVIVE A TRANSIENT AUTH FAILURE. `api.kaggle.com` intermittently dies
# with SSLEOFError during OAuth, and this process authenticates ONCE at module scope before its polling
# loop -- so an unguarded failure here kills the watcher instantly and the result is never read. The scope
# of this script is hours; the failure mode is seconds. Retry it.
_last = None
for _try in range(8):
    try:
        api.authenticate()
        break
    except Exception as _e:  # noqa: BLE001
        _last = _e
        print("auth attempt %d failed: %s" % (_try + 1, type(_e).__name__), flush=True)
        time.sleep(10)
else:
    raise SystemExit("could not authenticate after 8 attempts (%r) -- TRANSPORT, not a result" % _last)


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
for _ in range(200):                    # ~33 h at 10-min polls -- TPU queues have been observed
                                        # at ~26 h (FACTS A33), so a 13 h window could expire
                                        # BEFORE the gate ever ran, leaving the result unread.
    st = status()
    if st.endswith(("COMPLETE", "ERROR", "CANCEL")):
        break
    time.sleep(600)

st = status()
log = fetch()
print("status:", st, "| log chars:", len(log))

rec = {"ref": REF, "status": st, "log_chars": len(log),
       "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
if log:
    zs = re.search(r"\[zeroshot\] zero training steps: GOLD macro-AUC ([\d.]+)", log)
    vd = re.search(r"\[zeroshot\] VERDICT: (\w+) \(parent bar [\d.]+, got ([\d.]+)\)", log)
    rec["gold_macro_auc"] = float(zs.group(1)) if zs else None
    rec["verdict_line"] = vd.group(0) if vd else None
    rec["head_loaded"] = bool(re.search(r"HEAD loaded: 8/8", log))
    rec["head_tensors_carried"] = bool(re.search(r"head tensors carried: 8", log))
    rec["warm_start"] = (re.search(r"warm start -> (\S+)", log).group(1) if "warm start ->" in log else None)
    rec["per_label"] = (re.search(r"\[zeroshot\] per-label (\{.*\})", log).group(1)[:400]
                        if "[zeroshot] per-label" in log else None)
    rec["errors"] = re.findall(r"^.*(Traceback|Error:|error:).*$", log, re.M)[:4]
    a = rec["gold_macro_auc"]
    rec["decision"] = ("GEOMETRY MATCHES, checkpoint as strong as claimed -> candidate"
                       if a is not None and a >= 0.92 else
                       "geometry matches, weaker than msft2 -> weak candidate" if a is not None and a >= 0.90 else
                       "below the A167 member gate -> reject as a member" if a is not None and a >= 0.85 else
                       "*** GEOMETRY MISMATCH *** -> reject permanently" if a is not None else
                       "no gold AUC in the log -> INCONCLUSIVE, do not treat as a rejection")
else:
    rec["log"] = "UNREAD (transport) -- NOT a result"

OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(json.dumps(rec, indent=2))
print("wrote", OUT)
