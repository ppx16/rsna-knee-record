# -*- coding: utf-8 -*-
"""Decode a Kaggle kernel `.log` (a stream of JSON records) into plain text.

`kaggle kernels output` writes logs as `[{...},{...}]` where each record is
`{"stream_name": "stdout"|"stderr", "time": <t>, "data": "..."}`.  Reading those raw
burns context on escaping, and decoding them with an inline `python -c` in PowerShell
gets its quotes eaten (HANDOFF 74.4 #7 -- which is why this is a file).

    python scripts/klog.py downloads/diagslots3/rsna-knee-diag-slots.log
    python scripts/klog.py <log> --err        # stderr only (tracebacks)
    python scripts/klog.py <log> --tail 40
"""
import json
import pathlib
import re
import sys

REC = re.compile(r'\{"stream_name".*?\}(?=,\{|\]$)', re.S)


def payload(text: str) -> str:
    """Drop anything the CLI printed before the JSON array.

    `kaggle kernels logs` writes "Warning: Looks like you're using an outdated `kaggle`
    version ..." to the same stream, so the captured file starts with that line and
    `json.loads` on the whole text fails.  The regex fallback then has to guess, and on a
    2026-09-26 run it produced 5 bytes -- which looked exactly like a pipeline that never
    ran, and the acceptance test reported FAIL for two runs that had in fact completed with
    20/20 fingerprints.  A log decoder that turns success into failure is worse than no
    decoder, so: find the array and read from there.
    """
    for marker in ('[{"stream_name"', '[{'):
        i = text.find(marker)
        if i >= 0:
            return text[i:]
    return text


def records(text: str):
    """Yield (stream, data). Tries whole-array JSON first, falls back to regex."""
    text = payload(text)
    try:
        for r in json.loads(text):
            yield r.get("stream_name", "?"), r.get("data", "")
        return
    except Exception:
        pass
    # Some logs are truncated mid-array; salvage record by record.
    for m in REC.finditer(text):
        try:
            r = json.loads(m.group(0))
        except Exception:
            continue
        yield r.get("stream_name", "?"), r.get("data", "")


def main() -> int:
    # Windows consoles default to a legacy code page (GBK here). Kernel logs are decoded
    # with errors="replace", so printing them can die with UnicodeEncodeError -- a decoder
    # that cannot print what it decoded is not a decoder. Force UTF-8 + replacement.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    args = sys.argv[1:]
    p = pathlib.Path(args[0])
    only_err = "--err" in args
    tail = 0
    if "--tail" in args:
        tail = int(args[args.index("--tail") + 1])

    out = []
    for stream, data in records(p.read_text(encoding="utf-8", errors="replace")):
        if only_err and stream != "stderr":
            continue
        out.append(data.rstrip("\n"))
    lines = "\n".join(out).splitlines()
    if tail:
        lines = lines[-tail:]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
