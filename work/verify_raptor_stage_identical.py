"""The decisive check on "one axis only": the RAPTOR stage output must be BYTE-IDENTICAL.

The flattened routing is an OUTER step applied after the Raptor views are rendered. So if the two
arms carry the same Raptor composition (i.e. both really have the msft2 view), their `_raptor.csv`
intermediates must match exactly -- and the final submission must differ anyway, because the outer
step differs. That combination is what proves the edit is isolated.

If `_raptor.csv` DIFFERS, the arms are not the same Raptor stack and the "one axis" claim is false.
"""
import hashlib
import pathlib
import time

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
CACHE = pathlib.Path("work/klog_ab")

PAIRS = [("ppxl16/rsna-knee-0943cfmsft2", "rsna-knee-0943cfmsft2"),
         ("ppxl16/rsna-knee-0943cfmsft2flat", "rsna-knee-0943cfmsft2flat")]


def fetch(ref, slug, pattern):
    for attempt in range(4):
        try:
            api.kernels_output(ref, path=str(CACHE / slug), file_pattern=pattern,
                               force=True, quiet=True)
            return
        except Exception as e:  # noqa: BLE001
            print("   %s %s attempt %d: %s" % (slug, pattern, attempt, type(e).__name__))
            time.sleep(5)


for ref, slug in PAIRS:
    fetch(ref, slug, "_raptor.csv")

print("%-30s %-12s %s" % ("slug", "bytes", "sha256[:24]"))
sha = {}
for ref, slug in PAIRS:
    hits = sorted((CACHE / slug).rglob("_raptor.csv"))
    if not hits:
        print("%-30s MISSING" % slug)
        continue
    b = hits[0].read_bytes()
    sha[slug] = hashlib.sha256(b).hexdigest()
    print("%-30s %-12d %s" % (slug, len(b), sha[slug][:24]))

print()
if len(sha) == 2:
    a, b = (sha["rsna-knee-0943cfmsft2"], sha["rsna-knee-0943cfmsft2flat"])
    print("RAPTOR STAGE IDENTICAL:", a == b)
    if a == b:
        print("=> the Raptor composition is the same in both arms, so msft2-ours is present in")
        print("   the flat arm too (cfmsft2 measurably moved our rank, so it carries it).")
        print("   Combined with the differing final SHA, the flattening edit is ISOLATED.")
    else:
        print("=> !! the Raptor stage differs -- the arms are NOT the same stack minus routing.")
