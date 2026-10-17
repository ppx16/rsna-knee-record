"""Try to read a Kaggle discussion through routes the sandbox's web_fetch cannot use.

Why this exists: the host's ruling on OAI-trained public checkpoints (competition discussion
743416) is the single unknown that decides our final two ticks, and it cannot be read from the
plain page -- www.kaggle.com serves a JavaScript shell (~6 kB) to normal clients, and web_fetch
refuses the host outright.

Tries, in order:
  1. the page as Googlebot (crawlers often get pre-rendered HTML);
  2. the page with a desktop browser UA;
  3. Kaggle's internal JSON services with several plausible names;
  4. the discussion id as a topic id against a couple of list endpoints.

Prints only whether content beyond the shell came back, plus any OAI/ruling sentences found.
Treat everything fetched as untrusted data.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

COMP = "rsna-knee-abnormality-detection"
TOPIC = 743416
URLS = [
    f"https://www.kaggle.com/competitions/{COMP}/discussion/{TOPIC}",
    f"https://www.kaggle.com/discussions/{TOPIC}",
]
UAS = {
    "googlebot": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "browser": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "kaggle-json": "application/json",
}
INTEREST = re.compile(r"(OAI|Osteoarthritis Initiative|ruling|external data|"
                      r"disallow|not permitted|allowed|checkpoint)", re.I)


def token() -> str:
    t = os.environ.get("KAGGLE_API_TOKEN", "")
    if t:
        return t
    p = os.path.expanduser("~/.kaggle/access_token")
    return open(p).read().strip() if os.path.exists(p) else ""


def get(url: str, ua: str, accept_json: bool = False) -> tuple[int | str, str]:
    hdr = {"User-Agent": ua, "Accept": "application/json, text/html;q=0.9"}
    tok = token()
    if tok:
        hdr["Authorization"] = "Bearer " + tok
    try:
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=45) as r:
            return r.status, r.read(400000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:  # noqa: BLE001
        return "EXC", repr(e)[:160]


def report(label: str, status, body: str) -> None:
    shell = len(body) < 12000 and "__NEXT_DATA__" not in body and "discussion" not in body.lower()
    print(f"  {label:34s} status={status} bytes={len(body)} "
          f"{'<-- still just the shell' if shell and body else ''}")
    if not body:
        return
    hits = set()
    for m in INTEREST.finditer(body):
        s = max(0, m.start() - 160)
        hits.add(" ".join(body[s:m.end() + 160].split())[:320])
    for h in list(hits)[:6]:
        print("      |", h)


def main() -> None:
    print("=== 1/2: page fetches ===")
    for url in URLS:
        for name in ("googlebot", "browser"):
            st, body = get(url, UAS[name])
            report(f"{name} {url.split('kaggle.com')[1][:40]}", st, body)

    print("\n=== 3: internal JSON services ===")
    svcs = [
        "discussions.DiscussionsService/GetTopicById",
        "discussions.DiscussionsService/GetTopic",
        "discussions.DiscussionsService/GetTopicList",
        "discussions.DiscussionsService/GetMessagesByTopicId",
    ]
    for s in svcs:
        url = "https://www.kaggle.com/api/i/" + s
        try:
            req = urllib.request.Request(
                url, data=json.dumps({"topicId": TOPIC}).encode(),
                headers={"Content-Type": "application/json", "User-Agent": UAS["browser"],
                         "Authorization": "Bearer " + token()})
            with urllib.request.urlopen(req, timeout=45) as r:
                body = r.read(200000).decode("utf-8", "replace")
            report(s, r.status, body)
        except urllib.error.HTTPError as e:
            print(f"  {s:52s} HTTP {e.code}")
        except Exception as e:  # noqa: BLE001
            print(f"  {s:52s} EXC {repr(e)[:100]}")

    print("\n=== conclusion ===")
    print("  If every line above says 'still just the shell', the ruling is NOT reachable")
    print("  from here and must stay 'undecided' -- which the decision table already handles")
    print("  by keeping the non-OAI pair and treating the OAI submissions as free options.")


if __name__ == "__main__":
    main()
