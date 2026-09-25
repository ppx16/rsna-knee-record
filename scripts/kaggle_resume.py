# -*- coding: utf-8 -*-
"""Resumable Kaggle dataset downloader -- survives the connection drops that restart `kaggle datasets download`.

WHY THIS EXISTS
    `kaggle datasets download` already has retry logic, but when the server answers a Range
    request with 200 instead of 206 (or sends no Content-Length) it TRUNCATES the partial file
    and starts from zero.  Moving 22 GB over a home line, that happened here on 2026-09-24:
    the file went 7.30 GB -> 6.80 GB and the elapsed-time counter reset, i.e. ~40 minutes of
    transfer thrown away.  This script only ever APPENDS: if the bytes on disk are a prefix of
    the object (which they are, because they were written sequentially), a Range request
    continues from them.  Retries are unlimited and backed off; the final size is checked
    against the size the Kaggle API reports for that file.

    python scripts/kaggle_resume.py --ref dreaddevelopment/knee-raptor-corpus \
        --out D:\\kaggle\\corpus\\zips\\knee-raptor-corpus.zip
    python scripts/kaggle_resume.py --ref owner/slug --out f.zip --all      # every file in the dataset

    # a plain URL also works (e.g. a competition file's signed URL):
    python scripts/kaggle_resume.py --url '<signed>' --out train.csv --expected 12345

Designed to be copied into the rented box along with the rest of the migration kit: the same
drop-prone link is exactly what a fresh machine will hit.
"""
import argparse
import os
import pathlib
import sys
import time

import requests

CHUNK = 4 << 20
UA = {"User-Agent": "kaggle-resume/1.0"}


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or u == "TB":
            return "%.2f %s" % (n, u)
        n /= 1024.0


def api_client():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def list_files(api, ref: str):
    """[(name, total_bytes)] straight from the Kaggle API -- our expected-size oracle."""
    owner, slug = ref.split("/")
    out = []
    for f in api.dataset_list_files(ref).files:
        out.append((f.name, int(getattr(f, "totalBytes", 0) or 0)))
    return out


def resolve_url(api, ref: str, file_name: str = None) -> str:
    """file_name=None asks for the WHOLE dataset as one zip bundle (`<slug>.zip`), which is what
    `kaggle datasets download -d ref` fetches -- same request, just without file_name."""
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


def resolve_url_retry(api, ref: str, name: str, tries: int = 8) -> str:
    """The redirect to storage.googleapis.com is what fails here (SSLEOFError, intermittently) --
    the same flake that makes `kaggle competitions download` die on a 60 KB CSV.  Retry it."""
    last = None
    for i in range(tries):
        try:
            return resolve_url(api, ref, name)
        except Exception as e:
            last = e
            wait = min(30, 2 ** min(i, 5))
            sys.stdout.write("   !! url %d/%d failed (%s) -- retry in %ds\n" % (i + 1, tries, type(e).__name__, wait))
            sys.stdout.flush()
            time.sleep(wait)
    raise last


def head_size(url: str):
    try:
        r = requests.head(url, headers=UA, timeout=30, allow_redirects=True)
        n = r.headers.get("Content-Length")
        if r.status_code < 400 and n:
            return int(n), r.headers.get("Accept-Ranges", "")
    except Exception:
        pass
    return None, ""


def fetch(url: str, out: pathlib.Path, expected=None, max_tries=100) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    # Sidecar: after a verified complete download we record the size, so a re-run can skip the
    # whole thing WITHOUT resolving a URL -- important here because the sign step itself fails
    # in bursts, and re-resolving twelve finished files just to no-op wastes minutes.
    marker = out.with_suffix(out.suffix + ".ok")
    if out.exists() and marker.exists():
        try:
            want = int(marker.read_text().split()[0])
            if out.stat().st_size == want:
                print("ok   %s already complete (%s) -- skipping" % (out.name, human(want)))
                return 0
        except Exception:
            pass
    total = expected
    if total is None:
        total, _ = head_size(url)
    have = out.stat().st_size if out.exists() else 0
    if total and have > total:            # corrupt/oversized: start over rather than append garbage
        print("!! local file %s is LARGER than the object; truncating" % human(have))
        out.unlink()
        have = 0
    print("-> %s\n   have %s / %s" % (out, human(have), human(total) if total else "?"))
    t0, mark, mark_t = time.time(), have, time.time()
    tries = 0
    while total is None or have < total:
        tries += 1
        if tries > max_tries:
            print("!! giving up after %d tries" % max_tries)
            return 1
        try:
            headers = dict(UA)
            if have:
                headers["Range"] = "bytes=%d-" % have
            r = requests.get(url, headers=headers, stream=True, timeout=(20, 120))
            if r.status_code == 416:                       # already complete
                break
            if r.status_code not in (200, 206):
                raise RuntimeError("HTTP %d" % r.status_code)
            if total is None:
                cl = r.headers.get("Content-Length")
                total = (int(cl) + (have if r.status_code == 206 else 0)) if cl else None
                print("   total from response: %s" % (human(total) if total else "unknown"))
            if r.status_code == 200 and have:
                print("   !! server ignored Range (200) -- but we will NOT truncate; appending is wrong, "
                      "so this is fatal for this attempt")
                r.close()
                time.sleep(5)
                continue
            mode = "ab" if have else "wb"
            with open(out, mode) as f:
                for chunk in r.iter_content(CHUNK):
                    if not chunk:
                        continue
                    f.write(chunk)
                    have += len(chunk)
                    if time.time() - mark_t > 10:
                        rate = (have - mark) / max(time.time() - mark_t, 1e-6)
                        eta = (total - have) / rate if (total and rate > 0) else 0
                        sys.stdout.write("\r   %6.2f%%  %s / %s  %6.1f MB/s  eta %4.0fs   " % (
                            100.0 * have / total if total else 0, human(have),
                            human(total) if total else "?", rate / 1e6, eta))
                        sys.stdout.flush()
                        mark, mark_t = have, time.time()
            r.close()
            if total and have >= total:
                break
            # a short read with no exception: retry from where we are
        except Exception as e:
            wait = min(30, 2 ** min(tries, 5))
            sys.stdout.write("\n   !! %s: %s -- retry %d in %ds\n" % (type(e).__name__, e, tries, wait))
            sys.stdout.flush()
            time.sleep(wait)
    sys.stdout.write("\r" + " " * 78 + "\r")
    got = out.stat().st_size if out.exists() else 0
    ok = (total is None) or (got == total)
    if ok:
        try:
            marker.write_text("%d %s %s" % (got, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), url.split("?")[0]))
        except Exception:
            pass
    print("%s %s  %s%s" % ("OK  " if ok else "BAD ", human(got),
                           "(%.1f min, %.1f MB/s avg)" % ((time.time() - t0) / 60,
                                                          got / max(time.time() - t0, 1e-6) / 1e6),
                           "" if ok else "  expected %s" % human(total)))
    return 0 if ok else 1


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="resumable Kaggle downloader")
    ap.add_argument("--ref", help="owner/slug")
    ap.add_argument("--file", help="one file inside the dataset")
    ap.add_argument("--bundle", action="store_true",
                    help="fetch the WHOLE dataset as <slug>.zip (what `kaggle datasets download` does)")
    ap.add_argument("--all", action="store_true", help="every file in the dataset, one by one")
    ap.add_argument("--out", type=pathlib.Path, help="output path (required unless --all)")
    ap.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("."))
    ap.add_argument("--url", help="plain URL instead of a dataset ref")
    ap.add_argument("--expected", type=int, default=None)
    a = ap.parse_args()

    if a.url:
        if not a.out:
            print("!! --out is required with --url")
            return 2
        return fetch(a.url, a.out, a.expected)

    if not a.ref:
        print(__doc__)
        return 2
    api = api_client()
    if a.bundle:
        slug = a.ref.split("/")[-1]
        out = a.out or (a.outdir / (slug + ".zip"))
        return fetch(resolve_url_retry(api, a.ref, None), pathlib.Path(out), a.expected)

    files = list_files(api, a.ref)
    print("dataset %s: %d file(s)" % (a.ref, len(files)))
    for n, b in files:
        print("   %-40s %s" % (n, human(b) if b else "?"))
    if a.all:
        rc = 0
        for n, b in files:
            rc |= fetch(resolve_url_retry(api, a.ref, n), a.outdir / n, b)
        return rc
    want = a.file or next((n for n, _ in files if n.endswith(".zip")), None) \
        or (files[0][0] if len(files) == 1 else None)
    if not want:
        print("!! cannot tell which file to fetch; pass --file")
        return 2
    size = dict(files).get(want)
    out = a.out or (a.outdir / want)
    return fetch(resolve_url_retry(api, a.ref, want), pathlib.Path(out), size or a.expected)


if __name__ == "__main__":
    raise SystemExit(main())
