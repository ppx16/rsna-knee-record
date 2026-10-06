"""Read BOTH finished runs from their own logs.

TWO QUESTIONS, and both change decisions:

1. The reader arm ran in ~26 min, not the 6.5-8 h the author reports for their stack. Did the submitted
   run process the FULL hidden test set (1322 studies) or only the 3-study preview? If it was the full
   set, then `0.0000 of label cells moved` is a REAL failure and must be understood -- on 1322 studies a
  30 % rank leg cannot be a no-op. If it was the preview, the submission was never actually run and the
  26 min means something else.

2. `msft2-f1` finished on TPU v5e-8. That gives the per-fold cost that the five-fold prereg needed and
   that "~2.6 h, inherited from a ConvNeXt estimate" only guessed at.
"""
import json
import pathlib
import re
import time

from kaggle.api.kaggle_api_extended import KaggleApi

jobs = [("ppxl16/rsna-knee-0943cfmsft2-rdr", "rsna-knee-0943cfmsft2-rdr", "work/klog_rdr2"),
        ("ppxl16/rsna-knee-msft2-f1", "rsna-knee-msft2-f1", "work/klog_f1")]

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


def render(ref, slug, outdir):
    d = pathlib.Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        try:
            api.kernels_output(ref, path=str(d), file_pattern="%s.log" % slug, force=True, quiet=True)
            break
        except Exception as e:  # noqa: BLE001
            print("   fetch retry: %s" % type(e).__name__)
            time.sleep(10)
    hits = sorted(d.rglob("*.log"))
    if not hits:
        return ""
    raw = hits[0].read_text(encoding="utf-8", errors="replace")
    out = []
    for ln in raw.splitlines():
        ln = ln.lstrip(",").strip()
        if not ln:
            continue
        try:
            dd = json.loads(ln).get("data")
        except Exception:
            dd = None
        if dd:
            out.append(dd)
    txt = "".join(out)
    (d / "rendered.txt").write_text(txt, encoding="utf-8")
    return txt


for ref, slug, outdir in jobs:
    print("=" * 96)
    print(ref)
    try:
        print("   status:", getattr(api.kernels_status(ref), "status", "?"))
    except Exception as e:  # noqa: BLE001
        print("   status failed:", type(e).__name__)
    t = render(ref, slug, outdir)
    print("   log chars:", len(t))
    if not t:
        print("   (no log)")
        continue
    if "rdr" in slug:
        for pat, label in [
            (r"reader leg: .*", "READER LEG"),
            (r"COMPLETE: /kaggle/working/submission\.csv.*", "STACK COMPLETE"),
            (r"\[eda\] [^\n]*", "EDA"),
            (r"\[.*?\] test g1: .*", "TEST G1"),
            (r".*studies.*with series metadata.*", "TEST STUDIES"),
            (r"\[phase\].*\"stage\": \"(public_raptor|residual_coat|global96_coat|d4)\".*", "PHASES"),
        ]:
            hits = re.findall(pat, t)
            if hits:
                print("   -- %s --" % label)
                for h in hits[:8]:
                    print("      " + str(h)[:150])
    else:
        print("   -- fold-1 evidence --")
        for pat in [r"\[cv\] fold \d+/\d+: train \d+ \| OOF held-out \d+[^\n]*",
                    r"\[raptor\] loaded backbone[^\n]*",
                    r"\[raptor\] head tensors carried[^\n]*",
                    r"\[msft2\] HEAD loaded[^\n]*",
                    r"\[zeroshot\] VERDICT[^\n]*",
                    r"SWA over epochs[^\n]*",
                    r"TOPK epochs[^\n]*",
                    r"\[xla\] xla_device[^\n]*",
                    r"[^\n]*Traceback[^\n]*", r"[^\n]*Error[^\n]*"]:
            for h in re.findall(pat, t)[:4]:
                print("      " + str(h)[:170])
    ts = [float(x) for x in re.findall(r"\[\s*([\d.]+)s\]", t)]
    if ts:
        print("   -- max [Ns] stamp: %.1f s (%.2f h)" % (max(ts), max(ts) / 3600))
