"""Follow the rented-4090 3D-arm run from here, without holding an SSH session open.

Polls the box every 20 minutes, appends the training log tail locally, and records the held-out macro-AUC
whenever it appears. Stops on a FINAL line, on an error, or after ~24 h.

It exists because the whole point of renting the box was to stop waiting in a queue -- so nothing here may
itself become a thing to wait on: no interactive ssh, no blocking read, and the local record is written every
poll so an interrupted watcher still leaves evidence.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import time

KEY = str(pathlib.Path.home() / ".ssh" / "autodl_ed25519")
HOST = "root@connect.cqa1.seetacloud.com"
PORT = "34769"
OUT = pathlib.Path("work/arm3d_remote.log")
RES = pathlib.Path("state/arm3d_remote_result.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
RES.parent.mkdir(parents=True, exist_ok=True)

CMD = ("tail -c 4000 /root/arm3d_long.log 2>/dev/null; echo '---MARK---'; "
       "ls -la /kaggle/input/datasets/dreaddevelopment/*/ 2>/dev/null | grep -E 'vols|ids' ; "
       "echo '---MARK---'; tail -2 /root/launch_out.log 2>/dev/null")


def pull():
    r = subprocess.run(["ssh", "-i", KEY, "-p", PORT, "-o", "BatchMode=yes",
                        "-o", "ConnectTimeout=25", "-o", "StrictHostKeyChecking=accept-new", HOST, CMD],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout or "") + (r.stderr or "")


print("following the rented 4090 ...", flush=True)
for i in range(72):                       # ~24 h at 20-min polls
    txt = pull()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with OUT.open("a", encoding="utf-8") as f:
        f.write("\n===== poll %d @ %s =====\n%s\n" % (i, stamp, txt))

    aucs = re.findall(r"HELD-OUT fold (\d+) macro-AUC ([\d.]+)\s+\(n=(\d+)\)", txt)
    if aucs:
        fold, auc, n = aucs[-1]
        RES.write_text(
            '{"at": "%s", "held_fold": %s, "macro_auc": %s, "n_val": %s, "history": %s, '
            '"gate": "solo >= 0.90 on this large-n instrument (msft2 = 0.8847 same instrument)", '
            '"decision": "%s"}\n'
            % (stamp, fold, auc, n, str([[int(a), float(b)] for a, b, _ in aucs]),
               "CLEARS THE MEMBER BAR -- worth wiring and testing one arm" if float(auc) >= 0.90
               else "BELOW the member bar -- record honestly, do not wire it"),
            encoding="utf-8")
        print("[%s] AUC %s (n=%s) fold %s" % (stamp, auc, n, fold), flush=True)

    if "FINAL held-out macro-AUC" in txt:
        print("[%s] training finished" % stamp, flush=True)
        break
    if re.search(r"Traceback|CUDA out of memory|!!", txt):
        print("[%s] ERROR seen -- read %s" % (stamp, OUT), flush=True)
        break
    time.sleep(1200)

print("watcher done; evidence in", OUT, "and", RES)
