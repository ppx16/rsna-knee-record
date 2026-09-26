"""Call the code-competition submission endpoint directly to read the server's error body.

Why this exists: `kaggle competitions submit` prints only
    "400 Client Error: Bad Request for url: .../CreateCodeSubmission"
and throws the response body away.  A bare 400 has at least three candidate causes here
(FACTS A8 daily submission cap, a structural mismatch such as internet being enabled, or a bad
version/file name), so guessing from the status code alone would be the same mistake as reading
the gbk push failure as a quota problem.  This asks the server what it actually objected to.

Usage:
    python scripts/probe_submit_code.py [--kernel ppxl16/rsna-knee-0943] [--version 1] [--dry]
"""
import argparse
import json
import sys
import traceback


def describe(obj, depth=0):
    """Best-effort dump of whatever the client raised, without assuming its shape."""
    out = []
    pad = "  " * depth
    out.append("%stype: %s" % (pad, type(obj).__name__))
    out.append("%sstr : %s" % (pad, str(obj)[:2000]))
    for attr in ("body", "status", "reason", "message", "response", "args"):
        val = getattr(obj, attr, None)
        if val is None:
            continue
        if attr == "response":
            try:
                out.append("%sresponse.status_code: %s" % (pad, val.status_code))
                out.append("%sresponse.text: %s" % (pad, val.text[:2000]))
            except Exception as exc:            # pragma: no cover - defensive
                out.append("%sresponse: <unreadable %s>" % (pad, exc))
        elif attr == "args":
            out.append("%sargs: %r" % (pad, val))
        else:
            out.append("%s%s: %s" % (pad, attr, val))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", default="rsna-knee-abnormality-detection")
    ap.add_argument("--kernel", default="ppxl16/rsna-knee-0943")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--file", default="submission.csv")
    ap.add_argument("--message", default="0943 speedy k94 - 20/20 DINOv2 fingerprints, outer 0.65")
    ap.add_argument("--dry", action="store_true",
                    help="print what would be sent, and do not call the endpoint")
    args = ap.parse_args()

    payload = {
        "file_name": args.file,
        "competition_name": args.competition,
        "kernel": args.kernel,
        "kernel_version": args.version,
        "submission_description": args.message,
    }
    print("payload: %s" % json.dumps(payload))
    if args.dry:
        print("--dry: not calling the endpoint")
        return 0

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print("authenticated as: %s" % getattr(api, "config_values", {}).get("username", "?"))

    try:
        resp = api.competition_submit_code(
            args.file, args.message, args.competition,
            kernel=args.kernel, kernel_version=args.version,
        )
    except Exception as exc:
        print("=" * 70)
        print("SUBMIT RAISED - here is everything the client knows:")
        print("=" * 70)
        print(describe(exc))
        print("-" * 70)
        traceback.print_exc()
        return 1

    print("=" * 70)
    print("SUBMIT SUCCEEDED")
    print("=" * 70)
    print(resp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
