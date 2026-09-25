# -*- coding: utf-8 -*-
"""Print the SIGNED download URLs of a kernel's OUTPUT files, so the rented box can pull them
without ever seeing our Kaggle credentials.

WHY: getting a 7.9 GB shard from Kaggle onto the rented box had three bad options
  * local download then scp        -- the home->box uplink measured 0.41 MB/s -> ~5.4 h for 8 GB
  * a temporary credential window   -- works, but puts the whole account on a third-party VM
  * publish as a dataset            -- needs a local download+upload round trip (hours, unknown
                                       upload speed)
The CLI's own `kernels_output` does `requests.get(item.url)`, i.e. the API hands back a signed GCS
URL per output file.  We resolve those HERE (we hold the credentials) and hand over only the URLs.
No account access on the box, no dataset, no round trip.

    python scripts/kernel_output_urls.py --ref ppxl16/rsna-knee-corpus64-s0 --pattern 'vols_'
"""
import argparse
import pathlib
import sys

from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def api_client():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="owner/kernel-slug")
    ap.add_argument("--pattern", default=None, help="only names containing this substring")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="write name<TAB>url lines here")
    a = ap.parse_args()

    owner, slug = (a.ref.split("/") + [""])[:2]
    api = api_client()
    token = None
    lines = []
    with api.build_kaggle_client() as kaggle:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner
            req.kernel_slug = slug
            req.page_size = 100
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            for item in (resp.files or []):
                if a.pattern and a.pattern not in item.file_name:
                    continue
                lines.append("%s\t%s" % (item.file_name, item.url))
                print("%-40s %10s  url %d chars" % (
                    item.file_name, getattr(item, "total_bytes", "?"), len(item.url)), flush=True)
            token = getattr(resp, "next_page_token", None)
            if not token:
                break

    if not lines:
        print("!! no output files matched (has the version finished? a RUNNING version exposes nothing)")
        return 1
    if a.out:
        a.out.write_text("\n".join(lines) + "\n", encoding="ascii")
        print("wrote %d url(s) -> %s" % (len(lines), a.out))
    else:
        print("\n--- name<TAB>url ---")
        for ln in lines:
            print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
