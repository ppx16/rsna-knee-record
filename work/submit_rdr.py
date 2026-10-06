"""Submit the reader arm, with a fresh BEFORE snapshot taken first.

Accepts the timeout risk knowingly: the fallback logic inside the kernel protects the CONTENT of the
submission if the reader leg fails, but nothing can protect it if the whole kernel exceeds the 9 h cap.
The cost of that outcome is one submission slot (5/day, ~16 days left), not the final pair.

⚠️ `file_name` must be the name the KERNEL wrote (`submission.csv`), not a local path -- and the CLI
cannot do this at all (400 with no body; scripts/director.py L348-360).
"""
import datetime as dt
import json
import pathlib

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
REF = "ppxl16/rsna-knee-0943cfmsft2-rdr"
VERSION = 1
MSG = ("0943cfmsft2 + the public 2.5D ConvNeXt reader as a separate 30% rank leg "
       "(goodpjw2008/rsna-knee-2-5d-convnext-reader, 3 fold checkpoints). The stack is unchanged; the "
       "reader is a different architecture family blended as its own leg, not inside the Raptor "
       "sub-blend. Safe fallback: any reader failure keeps the stack's submission")

snaps = sorted(pathlib.Path("state").glob("lb_snapshot_*.json"))
before = json.loads(snaps[-1].read_text(encoding="utf-8")) if snaps else None
print("=== BEFORE ===")
print(json.dumps(before, indent=2) if before else "(none)")

api = KaggleApi()
api.authenticate()
print("\nsubmitting %s v%d ..." % (REF, VERSION))
resp = api.competition_submit_code(file_name="submission.csv", message=MSG, competition=COMP,
                                   kernel=REF, kernel_version=VERSION, quiet=True)
print("response:", resp)
ref = getattr(resp, "ref", None) or (resp.get("ref") if isinstance(resp, dict) else None)
print("SUBMISSION REF:", ref)

rec = {"at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "arm": "0943cfmsft2-rdr", "kernel": REF, "version": VERSION, "ref": ref,
       "before_snapshot": before, "message": MSG,
       "note": "server-side re-run expected 6.5-8 h (+~20 min reader); 9 h cap is the risk"}
out = "state/submit_%s.json" % (ref or "unknown")
pathlib.Path(out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
print("recorded ->", out)
