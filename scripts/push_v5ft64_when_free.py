# -*- coding: utf-8 -*-
"""
Push kernels/kernel-v9ft as soon as a GPU session frees.

Kaggle allows 2 concurrent batch GPU sessions. Right now both are held: this project's
eval kernel (ppxl16/rsna-knee-eval-v5v9) and the ARC clone. Neither should be cancelled,
so the push is retried on a bounded budget instead of a timer: it stops the moment it
succeeds, and gives up after MAX_TRIES so it can never become a background daemon.

This is deliberately NOT a scheduled task -- it is one bounded wait on one specific
blocker, which is what the standing instruction "别太依赖设自动定时任务" asks for.
"""
import subprocess
import sys
import time

DIR = r"kernels\kernel-v5ft64"
KID = "ppxl16/rsna-knee-v5ft64"
WAIT_S = 240
MAX_TRIES = 25


def push():
    p = subprocess.run(["kaggle", "kernels", "push", "-p", DIR],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=r"D:\kaggle\rsna-knee")
    return (p.stdout or "") + (p.stderr or "")


def main():
    for i in range(MAX_TRIES):
        out = push()
        low = out.lower()
        if "successfully pushed" in low:
            print(f"[{time.strftime('%H:%M:%S')}] PUSHED on try {i+1}", flush=True)
            print(out.strip()[-400:], flush=True)
            return
        if "session count" in low or "maximum batch" in low:
            print(f"[{time.strftime('%H:%M:%S')}] try {i+1}/{MAX_TRIES}: no GPU slot yet",
                  flush=True)
            time.sleep(WAIT_S)
            continue
        # any other error is a real problem -- stop rather than hammer it
        print(f"[{time.strftime('%H:%M:%S')}] try {i+1}: unexpected response, stopping",
              flush=True)
        print(out.strip()[-800:], flush=True)
        return
    print(f"gave up after {MAX_TRIES} tries; no GPU session freed", flush=True)


if __name__ == "__main__":
    main()
