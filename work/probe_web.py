import sys, os, json, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import requests
cfg = pathlib.Path(os.environ.get("KAGGLE_CONFIG_DIR", pathlib.Path.home()/".kaggle")) / "kaggle.json"
if not cfg.exists():
    cfg = pathlib.Path(os.environ.get("KAGGLE_CONFIG_DIR","")) / "kaggle.json"
print("config:", cfg, cfg.exists())
d = json.loads(cfg.read_text())
u, k = d["username"], d["key"]
print("user:", u, "key len:", len(k))
S = requests.Session(); S.auth = (u, k)
comp = "rsna-knee-abnormality-detection"
for url in [
    "https://www.kaggle.com/api/v1/competitions/list",
    f"https://www.kaggle.com/api/v1/competitions/submissions/list/{comp}",
    f"https://www.kaggle.com/api/v1/competitions/{comp}/submissions",
]:
    try:
        r = S.get(url, timeout=25)
        print("%-72s -> %s  len=%d  ct=%s" % (url.replace("https://www.kaggle.com",""), r.status_code, len(r.content), r.headers.get("content-type","")[:30]))
        if r.status_code == 200 and "json" in r.headers.get("content-type",""):
            print("     body head:", r.text[:200].replace("\n"," "))
    except Exception as e:
        print("%-72s -> EXC %s" % (url.replace("https://www.kaggle.com",""), str(e)[:90]))
