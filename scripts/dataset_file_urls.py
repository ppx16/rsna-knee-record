# -*- coding: utf-8 -*-
"""Resolve per-FILE signed download URLs for a Kaggle dataset version.

WHY THIS EXISTS.  The rented box has no Kaggle credentials (deliberately: the 44-slot corpus was
pulled with pre-signed URLs so nothing sensitive ever had to live on a third-party machine), and a
`--ref`-based download needs auth.  The alternative -- fetching the dataset BUNDLE -- is a zip, and
unpacking it needs the zip AND the extracted files at once: 29.66 + 29.66 = 59.3 GiB, which does NOT
fit the box's ~47.7 GiB of free space after the swap.

Kaggle exposes a per-file URL: `ApiDownloadDatasetRequest` with `file_name` set, and the CLI reads the
file's location straight out of `response.request.url`.  So resolve here (where credentials are), hand
the URLs to the box, and let `scripts/kaggle_parallel.py --url` pull each file with N streams and
resume.  Peak disk on the box is then the corpus itself: 29.66 GiB, no zip.

    python scripts/dataset_file_urls.py --ref ppxl16/rsna-knee-corpus64-dense --out state/urls.tsv
"""
import argparse
import pathlib
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="owner/dataset-slug")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="write name<TAB>url lines here")
    ap.add_argument("--pattern", default=None, help="only names containing this substring")
    ap.add_argument("--version", type=int, default=None)
    a = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

    api = KaggleApi()
    api.authenticate()
    owner, slug = a.ref.split("/")

    listing = api.dataset_list_files(a.ref).files or []
    # ⚠️ the listing is PAGINATED (default 20).  A dataset with 24 files would silently lose 4 of them,
    # and a corpus that is missing one shard still trains -- it just pairs studies with the wrong
    # volumes.  Ask for a big page explicitly.
    try:
        listing = api.dataset_list_files(a.ref, page_size=200).files or listing
    except TypeError:
        pass
    names = [f.name for f in listing]
    if a.pattern:
        names = [n for n in names if a.pattern in n]
    print("files: %d" % len(names))
    if len(names) > 20:
        print("  (>20 confirms the page-size fix took effect)")

    rows = []
    with api.build_kaggle_client() as kaggle:
        for n in names:
            last = None
            for attempt in range(6):
                try:
                    req = ApiDownloadDatasetRequest()
                    req.owner_slug = owner
                    req.dataset_slug = slug
                    req.dataset_version_number = a.version
                    req.file_name = n
                    resp = kaggle.datasets.dataset_api_client.download_dataset(req)
                    url = resp.request.url
                    if url and "X-Goog-Signature" in url:
                        rows.append((n, url))
                        print("  %-24s -> %d chars (try %d)" % (n, len(url), attempt + 1))
                        break
                    last = "no signed url in response"
                except Exception as e:                             # noqa: BLE001
                    last = "%s: %s" % (type(e).__name__, str(e)[:90])
                # MEASURED: this call is cheap (6.4 s for a 3.97 GB file) -- it follows the redirect to
                # get the signed URL and does NOT stream the body, so retrying costs seconds, not GB.
                time.sleep(min(20, 2 ** attempt))
            else:
                print("  %-24s FAILED: %s" % (n, last))
    missing = [n for n in names if n not in {r[0] for r in rows}]
    if missing:
        print("!! %d file(s) unresolved: %s" % (len(missing), ", ".join(missing)))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    # ASCII + LF, no BOM: this file is shipped to a Linux box and consumed by a shell loop
    with open(a.out, "w", encoding="ascii", newline="\n") as fh:
        for n, u in rows:
            fh.write("%s\t%s\n" % (n, u))
    print("wrote %d url(s) -> %s" % (len(rows), a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
