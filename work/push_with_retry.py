"""Push a kernel with retries.

The Kaggle API intermittently dies with SSLEOFError during OAuth introspection, which is a
TRANSPORT failure, not a build or permission problem -- so it must be retried rather than
reported as a failure. (Same family as the `-match ERROR` slip: do not let a transport error
masquerade as a terminal state.)
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

folder = pathlib.Path(sys.argv[1])
tries = int(sys.argv[2]) if len(sys.argv) > 2 else 8

for attempt in range(1, tries + 1):
    try:
        api = KaggleApi()
        api.authenticate()
        api.kernels_push(str(folder))
        print(f"PUSHED on attempt {attempt}")
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"attempt {attempt}: {msg}")
        if "SSLError" not in msg and "MaxRetry" not in msg and "UNEXPECTED_EOF" not in msg:
            print("not a transport error -- giving up (this one is real)")
            raise SystemExit(1)
        time.sleep(8 + 6 * attempt)

print("all attempts hit transport errors")
raise SystemExit(1)
