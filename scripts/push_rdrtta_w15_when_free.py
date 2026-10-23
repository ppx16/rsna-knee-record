# -*- coding: utf-8 -*-
"""Push the RSNA rdrtta-w15 kernel (TTA + w=0.15) as soon as a batch GPU session frees.

Written out in full rather than string-replaced from a sibling: the last time I generated a
pusher by copying push_v9ft_when_free.py and replacing two literals, PowerShell ate the
backslash escape, DIR kept pointing at kernel-v9ft while KID said v5ft64, and the script
silently re-pushed the WRONG KERNEL. Cheap to write, expensive to get wrong.

The RSNA GPU slots are held by the v9ft re-run (accidental, will error on the guard) and
whatever else. Bounded retry, stops on success, gives up after MAX_TRIES.
"""
import subprocess
import time

DIR = r"kernels\kernel-0943cfmsft2-rdrtta-w15"
KID = "ppxl16/rsna-knee-0943cfmsft2-rdrtta-w15"
WAIT_S = 240
MAX_TRIES = 30


def main():
    assert DIR.endswith("rdrtta-w15"), DIR
    assert KID.endswith("rdrtta-w15"), KID
    for i in range(MAX_TRIES):
        p = subprocess.run(["kaggle", "kernels", "push", "-p", DIR],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=r"D:\kaggle\rsna-knee")
        out = (p.stdout or "") + (p.stderr or "")
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
        print(f"[{time.strftime('%H:%M:%S')}] try {i+1}: unexpected response, stopping",
              flush=True)
        print(out.strip()[-800:], flush=True)
        return
    print(f"gave up after {MAX_TRIES} tries", flush=True)


if __name__ == "__main__":
    main()
