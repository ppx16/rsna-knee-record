"""What did the teams ABOVE us publish? Query the top of the board properly.

Two questions:
  1. do the leaders publish anything at all?
  2. if they do, is it RSNA-specific and does it hint at a method we lack?
"""
import json
import pathlib
import time

from kaggle.api.kaggle_api_extended import KaggleApi

WORK = pathlib.Path(r"D:\kaggle\rsna-knee\work")
above = json.loads((WORK / "above_us.json").read_text(encoding="utf-8"))

# rank -> usernames, taken from the live board (top 40)
TOP = [
    (1, "rhoskeri", ["rhoskeri"]),
    (2, "Pa3baJluHa", ["hettpugymajlhuk"]),
    (3, "joinT fusion", ["nantohock", "tomokihirose", "wadakoki"]),
    (4, "Wassim Dobbi", ["wassimdobbi"]),
    (5, "YOKKOISHO", ["shin9915", "syamoji141chan", "tennogh"]),
    (6, "TomKroumov", ["tom1115"]),
    (7, "dalab", ["junpeimorioka", "shuyaakagawa", "sotakurihara1", "sugiuraakie"]),
    (8, "NMarkS", ["nmarks"]),
    (9, "knee-deep", ["shunyaiida", "t88take", "takashisomeya", "takeharuyoshikawa", "yosukeyam"]),
    (10, "Scott Willis", ["scottgwillis"]),
    (11, "takoi & charmq", ["charmq", "takoihiraokazu"]),
    (12, "AndreyKvachov", ["akvachov"]),
    (13, "Parag & Ian", ["paragkale", "vaillant"]),
    (14, "daimaru", ["daimaruk"]),
    (15, "Attention Is All You Knee", ["aleksanderhan", "aurax7", "coolinlai", "davidsarrat", "zongzishuang"]),
    (17, "Volkopat", ["volkopat"]),
    (19, "Q", ["quan0095"]),
    (20, "Mazzutti", ["profmazzutti"]),
    (25, "zosamu", ["zosamu"]),
    (26, "mimizuku", ["kuroshiro07", "takumashiga"]),
    (27, "Optimo", ["optimo"]),
]

api = KaggleApi()
api.authenticate()

report = {}
for rank, team, users in TOP:
    print("=" * 78)
    print(f"rank {rank:>3}  {team}")
    print("=" * 78)
    for u in users:
        try:
            ks = api.kernels_list(user=u, page_size=50)
        except Exception:
            ks = []
        try:
            ds = api.dataset_list(user=u)
        except Exception:
            ds = []
        kt = [str(getattr(k, "title", "") or "") for k in ks]
        dt = [(str(getattr(d, "title", "") or ""), float(getattr(d, "total_bytes", 0) or 0)) for d in ds]
        kr = [t for t in kt if "rsna" in t.lower() or "knee" in t.lower()]
        dr = [(t, b) for t, b in dt if "rsna" in t.lower() or "knee" in t.lower()]
        report[u] = {"kernels": kt, "datasets": dt}
        print(f"  {u:<20} kernels={len(kt):<3} (rsna/knee {len(kr):<2})  datasets={len(dt):<3} (rsna/knee {len(dr)})")
        for t in kr[:5]:
            print(f"        K  {t[:72]}")
        for t, b in dr[:8]:
            print(f"        D  {t[:58]:<58} {b/1048576:8.1f} MB")
        time.sleep(0.3)

(WORK / "top_teams_public.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nwrote {WORK / 'top_teams_public.json'}")
