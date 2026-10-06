"""Fresh public sweep: anything NEW since the 07:0x read, focusing on kernels and datasets.

The community is demonstrably active today -- the 0.944 notebook that gave us the reader leg was posted
at 01:29Z, and the reader leg is the only lever this campaign has found from public work. So the sweep is
worth repeating rather than trusting a two-hour-old read.

What counts as a find: a NEW kernel or dataset that changes something measurable. Titles and vote counts
are not evidence; the sweep's job is only to surface candidates for the artifact checks.
"""
import datetime as dt
import pathlib

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
now = dt.datetime.now(dt.timezone.utc)
print("sweep at %s" % now.strftime("%Y-%m-%dT%H:%M:%SZ"))

api = KaggleApi()
api.authenticate()
seen = pathlib.Path("work/newpub_1006")

print()
print("=== competition kernels, newest 30 (marking what we have already pulled) ===")
try:
    ks = api.kernels_list(competition=COMP, sort_by="dateRun", page_size=30) or []
except Exception as e:  # noqa: BLE001
    print("  list failed:", str(e)[:120])
    ks = []
mine = {p.name for p in seen.iterdir() if p.is_dir()}
for k in ks:
    ref = str(getattr(k, "ref", ""))
    slug = ref.split("/", 1)[-1]
    lr = str(getattr(k, "last_run_time", ""))[:19]
    votes = getattr(k, "total_votes", "?")
    age_h = ""
    try:
        t = dt.datetime.strptime(lr, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
        age_h = "%.1fh" % ((now - t).total_seconds() / 3600)
    except Exception:
        pass
    tag = "PULLED" if slug in mine else "      "
    print("   %-58s %-19s %-6s v=%-4s %s" % (slug[:58], lr, age_h, votes, tag))

print()
print("=== datasets: newest for this competition (search by common stems) ===")
found = {}
for q in ("rsna knee", "knee mri", "rsna-knee"):
    try:
        got = api.dataset_list(search=q) or []
    except Exception as e:  # noqa: BLE001
        print("   search %-12r failed %s" % (q, str(e)[:60]))
        continue
    for d in got:
        found[getattr(d, "ref", "?")] = (str(getattr(d, "title", ""))[:64],
                                         str(getattr(d, "lastUpdated", ""))[:19])
print("   %d distinct knee datasets found" % len(found))
for ref, (title, upd) in sorted(found.items(), key=lambda kv: kv[1][1], reverse=True)[:22]:
    print("   %-56s %-19s %s" % (ref[:56], upd, title))
