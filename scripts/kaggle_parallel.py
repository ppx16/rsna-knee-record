# -*- coding: utf-8 -*-
"""Parallel-chunked resumable downloader -- because one connection to Kaggle/GCS is throttled.

MEASURED on the AutoDL box, 2026-09-24 (HANDOFF 146.4):

    single stream, 30 MB                    3.47 MB/s
    6 parallel streams, 30 MB each          2.6-3.3 MB/s EACH  -> ~15 MB/s aggregate
    through AutoDL's 学术加速 proxy         18-116 KB/s  (30-100x SLOWER -- do not use it)

So the object is not bandwidth-limited, it is limited PER CONNECTION.  `kaggle_resume.py`
(fixed above) gives 0.3-3 MB/s; splitting the file into N ranges and fetching them
concurrently gives ~15 MB/s, i.e. 11 GB in ~12 minutes instead of 6+ hours.

Design: preallocate the output to its full size, give each worker a disjoint byte range that it
fills with its own file handle (no shared seek state, no locks on the hot path), and persist each
chunk's progress to a sidecar so a killed run resumes instead of restarting.  Only APPENDS within
a chunk, so a crash can never leave a hole that looks complete.

    python scripts/kaggle_parallel.py --url '<signed>' --out f.zip --expected 11922316421 --streams 12
    python scripts/kaggle_parallel.py --ref owner/slug --bundle --outdir . --streams 12
"""
import argparse
import json
import os
import pathlib
import sys
import threading
import time

import requests

UA = {"User-Agent": "kaggle-parallel/1.0"}
BLOCK = 1 << 20


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or u == "TB":
            return "%.2f %s" % (n, u)
        n /= 1024.0


def resolve(api, ref, file_name=None):
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
    owner, slug = ref.split("/")
    with api.build_kaggle_client() as kaggle:
        req = ApiDownloadDatasetRequest()
        req.owner_slug = owner
        req.dataset_slug = slug
        if file_name:
            req.file_name = file_name
        resp = kaggle.datasets.dataset_api_client.download_dataset(req)
    return resp.request.url


def head_size(url, tries=6):
    """Total size of a signed URL.

    HEAD is the cheap way, but kaggleusercontent.com signed URLs answer HEAD with **404**, which
    looks exactly like an expired or wrong URL -- measured 2026-09-24, and it stalled a whole
    download.  A one-byte ranged GET is always allowed and reports the true length in Content-Range
    (`bytes 0-0/3981164672`), so fall back to that before giving up.
    """
    last = None
    for i in range(tries):
        try:
            r = requests.head(url, headers=UA, timeout=30, allow_redirects=True)
            n = r.headers.get("Content-Length")
            if r.status_code < 400 and n:
                return int(n)
            last = "HTTP %d" % r.status_code
        except Exception as e:
            last = e
        # HEAD refused (or the host does not implement it): ask for exactly one byte and read the
        # total out of Content-Range.
        try:
            h = dict(UA)
            h["Range"] = "bytes=0-0"
            r = requests.get(url, headers=h, timeout=30, allow_redirects=True, stream=True)
            cr = r.headers.get("Content-Range", "")          # "bytes 0-0/3981164672"
            if "/" in cr:
                total = cr.rsplit("/", 1)[1].strip()
                if total.isdigit() and int(total) > 0:
                    r.close()
                    return int(total)
            if r.status_code < 400 and r.headers.get("Content-Length"):
                n = int(r.headers["Content-Length"])
                r.close()
                return n
            code = r.status_code
            r.close()
            last = "HTTP %d (HEAD and ranged GET both unusable)" % code
        except Exception as e:
            last = e
        time.sleep(min(20, 2 ** i))
    raise RuntimeError("cannot determine size: %s" % last)


def fetch(url, out: pathlib.Path, expected=None, streams=12, max_tries=60, rotate_mb=32, rotate_sec=0):
    out.parent.mkdir(parents=True, exist_ok=True)
    total = expected or head_size(url)
    if total <= 0:
        raise RuntimeError("bad size")
    side = out.with_suffix(out.suffix + ".parts")

    n = max(1, min(streams, (total + (8 << 20) - 1) // (8 << 20)))
    bounds = []
    per = total // n
    for i in range(n):
        s = i * per
        e = total - 1 if i == n - 1 else (s + per - 1)
        bounds.append([s, e, 0])          # start, end, done-within-chunk

    # resume bookkeeping
    if out.exists() and out.stat().st_size == total and side.exists():
        try:
            saved = json.loads(side.read_text())
            if saved.get("total") == total and len(saved.get("done", [])) == n:
                for i, d in enumerate(saved["done"]):
                    bounds[i][2] = min(int(d), bounds[i][1] - bounds[i][0] + 1)
        except Exception:
            pass

    if not out.exists() or out.stat().st_size != total:
        with open(out, "wb") as f:
            f.truncate(total)
        for b in bounds:
            b[2] = 0

    done0 = sum(b[2] for b in bounds)
    if done0 >= total:
        print("ok   already complete (%s)" % human(total))
        return 0
    print("-> %s\n   %s total, %d streams, %s already present" % (out, human(total), n, human(done0)))

    lock = threading.Lock()
    stop = threading.Event()
    errors = []

    def worker(idx):
        s, e, have = bounds[idx]
        tries = 0
        while (s + have) <= e and not stop.is_set():
            cur = s + have
            try:
                headers = dict(UA)
                headers["Range"] = "bytes=%d-%d" % (cur, e)
                r = requests.get(url, headers=headers, stream=True, timeout=(20, 120))
                if r.status_code == 416:
                    break
                if r.status_code != 206:
                    raise RuntimeError("expected 206, got %d" % r.status_code)
                # ROTATE: the per-connection throttle decays with the connection's AGE, not with
                # total bytes moved -- measured on the box: a sustained 12-stream run fell to
                # 1.3-1.8 MB/s, while freshly opened connections still did 2.5-5.0 MB/s EACH.
                # So close and reopen every `rotate` bytes; the outer while reopens at `have`.
                conn = 0
                t_conn = time.time()
                with open(out, "r+b") as f:
                    f.seek(cur)
                    for blk in r.iter_content(BLOCK):
                        if not blk:
                            continue
                        if stop.is_set():
                            break
                        f.write(blk)
                        have += len(blk)
                        conn += len(blk)
                        with lock:
                            bounds[idx][2] = have
                        if rotate_mb and conn >= rotate_mb * (1 << 20):
                            break
                        # TIME-based rotation too: a throttled connection can deliver almost
                        # nothing while never erroring, so a byte threshold alone never fires.
                        # Measured: the pipe drops to <1 MB/s after a burst of a few GB, while
                        # freshly opened connections still do 2.5-5 MB/s each.
                        if rotate_sec and (time.time() - t_conn) >= rotate_sec:
                            break
                r.close()
                tries = 0
            except Exception as ex:
                tries += 1
                if tries == 1:
                    # Print the FIRST failure of every chunk.  A silent retry loop looks exactly
                    # like a stalled download (which is how a bad URL cost 40 s here on 2026-09-24).
                    sys.stdout.write("\n   !! chunk %d: %s: %s\n" % (idx, type(ex).__name__, ex))
                    sys.stdout.flush()
                if tries > max_tries:
                    errors.append("chunk %d: %s" % (idx, ex))
                    return
                time.sleep(min(30, 2 ** min(tries, 5)))

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()

    last_report = 0.0
    mark, mark_t = done0, t0
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(2)
            with lock:
                done = sum(b[2] for b in bounds)
            if time.time() - mark_t > 10:
                rate = (done - mark) / max(time.time() - mark_t, 1e-6)
                eta = (total - done) / rate if rate > 0 else 0
                sys.stdout.write("\r   %6.2f%%  %s / %s  %6.1f MB/s  eta %4.0fs   " % (
                    100.0 * done / total, human(done), human(total), rate / 1e6, eta))
                sys.stdout.flush()
                mark, mark_t = done, time.time()
            if time.time() - last_report > 15:
                try:
                    side.write_text(json.dumps({"total": total, "done": [b[2] for b in bounds]}))
                except Exception:
                    pass
                last_report = time.time()
    except KeyboardInterrupt:
        stop.set()
    for t in threads:
        t.join(timeout=30)
    sys.stdout.write("\r" + " " * 78 + "\r")

    got = sum(b[2] for b in bounds)
    if got >= total and not errors:
        try:
            side.write_text(json.dumps({"total": total, "done": [b[2] for b in bounds]}))
        except Exception:
            pass
        print("OK   %s  (%.1f min, %.1f MB/s avg)" % (
            human(got), (time.time() - t0) / 60, got / max(time.time() - t0, 1e-6) / 1e6))
        return 0
    print("BAD  %s / %s   errors=%s" % (human(got), human(total), errors[:3]))
    return 1


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="parallel-chunked resumable downloader")
    ap.add_argument("--url")
    ap.add_argument("--ref")
    ap.add_argument("--file")
    ap.add_argument("--bundle", action="store_true")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("."))
    ap.add_argument("--expected", type=int)
    ap.add_argument("--streams", type=int, default=12)
    ap.add_argument("--rotate-mb", type=int, default=32,
                    help="close+reopen each connection after this many MB (0 = never). The "
                         "per-connection throttle decays with connection AGE, so rotating restores "
                         "full speed: sustained 12-stream fell to 1.5 MB/s while fresh connections "
                         "still did 2.5-5.0 MB/s each.")
    ap.add_argument("--rotate-sec", type=int, default=0,
                    help="also reopen every N seconds. Needed because a throttled connection can "
                         "deliver ~0 MB/s forever without ever erroring, so a byte threshold alone "
                         "never fires.")
    a = ap.parse_args()

    if a.url:
        out = a.out
        if not out:
            print("!! --out required with --url")
            return 2
        return fetch(a.url, out, a.expected, a.streams, rotate_mb=a.rotate_mb, rotate_sec=a.rotate_sec)

    if not a.ref:
        print(__doc__)
        return 2
    from kaggle_resume import api_client  # local sibling; only needed for --ref
    api = api_client()
    if a.bundle:
        slug = a.ref.split("/")[-1]
        return fetch(resolve(api, a.ref, None), a.out or (a.outdir / (slug + ".zip")), a.expected,
                     a.streams, rotate_mb=a.rotate_mb, rotate_sec=a.rotate_sec)
    files = api.dataset_list_files(a.ref).files
    name = a.file or (files[0].name if len(files) == 1 else None)
    if not name:
        print("!! pass --file (dataset has %d files) or --bundle" % len(files))
        return 2
    return fetch(resolve(api, a.ref, name), a.out or (a.outdir / name), a.expected, a.streams,
                 rotate_mb=a.rotate_mb, rotate_sec=a.rotate_sec)


if __name__ == "__main__":
    raise SystemExit(main())
