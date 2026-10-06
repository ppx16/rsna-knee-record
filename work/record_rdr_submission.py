"""Append the reader arm submission to the ledger."""
import csv
import pathlib

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
subs = api.competition_submissions("rsna-knee-abnormality-detection")
row = next((s for s in subs if str(getattr(s, "ref", "")) == "56872230"), None)
date = str(getattr(row, "date", "")) if row else ""
print("date:", date)

DESC = ("0943cfmsft2-rdr - the stack UNCHANGED plus the public 2.5D ConvNeXt reader "
        "(goodpjw2008/rsna-knee-2-5d-convnext-reader, 3 fold checkpoints) as a SEPARATE 30% rank leg. "
        "Differs from every earlier arm in the COMPOSITION: cn0/msft2 go inside the Raptor sub-blend, "
        "this is its own leg of the final rank blend from a different architecture family. Author's own "
        "sweep on the community stack: 0.15->0.944, 0.30->0.944, 0.45->0.942, stack alone 0.943. "
        "Preview verified: stack SHA unchanged (095bcf70), reader ran 3 ckpts in 27s, no error. "
        "Safe fallback keeps the stack's submission if the reader fails. "
        "Before-snapshot 2026-10-06T06:11:10Z: rank 645, line 527, margin -118, in-block 16.03%")

led = pathlib.Path("state/lb_ground_truth.csv")
with led.open("a", encoding="utf-8", newline="") as fh:
    csv.writer(fh).writerow([date, "56872230", DESC, "", "PENDING"])
print("appended ledger row; rows now:", sum(1 for _ in led.open(encoding='utf-8')) - 1)
