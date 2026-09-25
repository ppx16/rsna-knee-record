# -*- coding: utf-8 -*-
"""AutoDL open API -- what it can and cannot do, verified 2026-09-24.

WHAT I WANTED: power the rented 4090D off from here, because it costs CNY 2.88/h idle and has no
decision-relevant work left (the 64-slot corpus is 31.8 GB and will not fit its 11 GB of free space,
HANDOFF 152.3).

WHAT IS TRUE (measured, not guessed):

  * Base https://api.autodl.com, auth header `Authorization: <token>`, token from the console's
    设置 -> 开发者Token.  The developer namespace is `/api/v1/dev/...`.
  * Those routes are **POST-only**: GET returns `404 page not found`, POST answers.  A GET probe
    therefore looks exactly like a wrong path -- that cost me several rounds.
  * The ENTIRE documented open API is 22 routes: `instance/pro/*`, `deployment/*`,
    `wallet/balance`, `machine/region/gpu_stock`, `image/private/list`, `exclusive_nfs/mount`.
    (Read off the published MkDocs search index, which lists every route in the docs.)
  * **All instance routes are `/instance/pro/`** -- they cover 容器实例Pro only.  This account has
    no Pro instance (`instance/pro/list` -> result_total 0), and calling pro/power_off with uuids
    derived from our container hostname returns `RecordNotFoundError / 未查询到相关实例`.
    => **There is no API to power off a regular 容器实例.  It is a manual click in the console.**
       Do not "fix" a failing power-off here by trying more uuid spellings; the product line is
       simply not exposed.
  * `/api/v1/region/list` is PUBLIC (it answers without a token) -- so "it returned Success" is not
    evidence that a token works.  Only `/wallet/balance` proves auth.
  * `wallet/balance` fields are in **thousandths of a yuan**: assets 40160 => CNY 40.16.

So this tool is a BALANCE/BURN monitor, not a power switch.

    python scripts/autodl_api.py balance
    python scripts/autodl_api.py routes          # what the API actually exposes (verified list)
"""
import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = "https://api.autodl.com"

# Verified by probing: POST + Authorization works on all of these; GET 404s.
ROUTES = {
    "balance": "/api/v1/dev/wallet/balance",
    "pro_list": "/api/v1/dev/instance/pro/list",
    "pro_power_off": "/api/v1/dev/instance/pro/power_off",
    "pro_power_on": "/api/v1/dev/instance/pro/power_on",
    "gpu_stock": "/api/v1/dev/machine/region/gpu_stock",
    "region_list_public": "/api/v1/region/list",
}


def token():
    t = os.environ.get("AUTODL_TOKEN", "").strip()
    if t:
        return t
    p = ROOT / "state" / "autodl_token.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    raise SystemExit("No AutoDL token (set $env:AUTODL_TOKEN or write state/autodl_token.txt). "
                     "Billing credential -- keep it out of any shared copy of this repo.")


def post(path, body=None, tok=None, timeout=25):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(body or {}).encode(), method="POST")
    req.add_header("Authorization", tok or token())
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, {"raw": e.read().decode("utf-8", "replace")[:300]}
    except Exception as e:                                       # noqa: BLE001
        return None, {"error": "%s: %s" % (type(e).__name__, e)}


def cmd_balance(a):
    st, j = post(ROUTES["balance"])
    if j.get("code") != "Success":
        print("balance -> %s %s" % (st, json.dumps(j, ensure_ascii=False)[:300]))
        return 1
    d = j["data"]
    yuan = lambda v: v / 1000.0                                   # noqa: E731
    print("AutoDL wallet")
    print("  assets   (余额)      CNY %8.2f" % yuan(d.get("assets", 0)))
    print("  voucher  (代金券)    CNY %8.2f" % yuan(d.get("voucher_balance", 0)))
    print("  blocked  (冻结)      CNY %8.2f" % yuan(d.get("blocked_asset", 0)))
    print("  accumulate (累计消费) CNY %8.2f" % yuan(d.get("accumulate", 0)))
    rate = a.rate
    if rate:
        hours = yuan(d.get("assets", 0)) / rate
        print("  burn     CNY %.2f/h -> about %.1f h left (%.1f days)" % (rate, hours, hours / 24))
    return 0


def cmd_routes(a):
    st, j = post(ROUTES["balance"])
    print("auth (wallet/balance): %s %s" % (st, (j.get("code") or j)))
    st, j = post(ROUTES["pro_list"], {"page_index": 1, "page_size": 100})
    n = j.get("data", {}).get("result_total") if j.get("code") == "Success" else "?"
    print("pro instances on this account: %s" % n)
    if n == 0:
        print("  => the rented instance is a REGULAR 容器实例; the open API cannot power it off.")
        print("     Power it off in the AutoDL console (a manual click).")
    print()
    for k, v in ROUTES.items():
        print("  %-20s POST %s" % (k, v))
    return 0


def main():
    ap = argparse.ArgumentParser()
    a2 = ap.add_subparsers(dest="cmd", required=True)
    b = a2.add_parser("balance")
    b.add_argument("--rate", type=float, default=2.88, help="CNY/hour, for the burn estimate")
    b.set_defaults(fn=cmd_balance)
    a2.add_parser("routes").set_defaults(fn=cmd_routes)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
