# -*- coding: utf-8 -*-
"""Measure how fast one URL actually downloads, BEFORE committing to a multi-hour transfer.

WHY THIS EXISTS.  I asserted "the 64-slot corpus cannot be moved to a rented box" on the strength of a
single-stream measurement (1.6 MB/s), then measured 16 parallel streams on the box and got 0.89 MB/s
-- WORSE.  Meanwhile the 44-slot corpus had gone across at 14.9 MB/s over the same kind of link.  The
difference turned out to be dataset bundles (storage.googleapis.com, per-connection throttle that
parallelism beats) versus kernel outputs (kaggleusercontent.com, which does not aggregate).

So: never plan a transfer from an assumed rate.  Point this at the URL, give it a few seconds, and
get the real number.

    python scripts/url_speedtest.py --url-file .testtmp/vols_url.txt
    python scripts/url_speedtest.py --url-file u.txt --streams 16 --mb-per-stream 8
"""
import argparse
import pathlib
import sys
import threading
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
MB = 1024 * 1024


def fetch(url, start, length, out, done, errors, timeout):
    req = urllib.request.Request(url)
    req.add_header("Range", "bytes=%d-%d" % (start, start + length - 1))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            buf = b""
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                buf += chunk
        out.append(len(buf))
    except Exception as e:                                     # noqa: BLE001
        errors.append("%s: %s" % (type(e).__name__, e))
    finally:
        done[0] += 1


def run(url, streams, per_stream_mb, timeout, offset):
    out, errors, done = [], [], [0]
    length = per_stream_mb * MB
    ths = [threading.Thread(target=fetch, args=(url, offset + i * length, length, out, done,
                                                 errors, timeout))
           for i in range(streams)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    dt = max(time.time() - t0, 1e-6)
    got = sum(out)
    rate = got / dt / MB
    print("streams %2d x %3d MB requested | got %.1f MB in %.1f s"
          % (streams, per_stream_mb, got / MB, dt))
    print("  AGGREGATE %.2f MB/s   (%.2f MB/s per stream)" % (rate, rate / streams))
    if got:
        print("  => 31.8 GB would take %.1f min at this rate" % (31.8 * 1024 / rate / 60))
    for e in errors[:4]:
        print("  stream error: %s" % e)
    return rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--url-file")
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--mb-per-stream", type=int, default=8)
    ap.add_argument("--offset", type=int, default=1_000_000_000,
                    help="start byte, so the sample is not all headers")
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()
    url = a.url or (pathlib.Path(a.url_file).read_text(encoding="utf-8").strip()
                    if a.url_file else None)
    if not url:
        raise SystemExit("need --url or --url-file")
    print("host: %s" % url.split("/")[2])
    run(url, a.streams, a.mb_per_stream, a.timeout, a.offset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
