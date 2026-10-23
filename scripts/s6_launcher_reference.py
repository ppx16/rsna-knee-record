"""K1 component-B launcher = S6F0/infer_s6f0.py with Kaggle paths.  Runs Heliosli's compact_infer.main() UNCHANGED.
As on the laptop: sys.path = [cloud_code, heliosli, ...] and the training.download_data stub (only the chunked-archive reader
uses it; we read the DICOM directory), installed in the main process AND in every spawn worker (CStudies._setup).
Three K1 guards, none of which changes a prediction on the laptop's path (each acts only where that path would fail):
- distutils: the vendored MONAI 0.6.0 imports distutils.util.strtobool (Python >= 3.12 has no stdlib distutils; setuptools'
  shim provides it on the laptop).  Only if that import fails, a module holding the stdlib's strtobool is installed.
- setup failure (review M1): compact_infer's _setup sets self.native before it loads the SKM-TEA V-Net, so a failed load left a
  worker scoring every later study with field-centre views and no anchor, silently.  Here a failed _setup resets self.native
  (the next study re-runs it; probe_infer.install_native is memoised, the one step that is not re-entrant) and re-raises (this
  study becomes a fallback, retried); a scored item without an anchor is turned into a failed item (fallback, retried).
K1_PX_DUMP (parity mode only): every item's pixel / meta hashes are appended to <dir>/px_<pid>.jsonl; the item is not changed."""
import hashlib, json, os, sys, types
SRC = __SRC__
CODE = __CODE__
ANCHOR_VIEWS = ("v3k", "v4")


def _strtobool(val):
    """distutils.util.strtobool (CPython 3.11 / setuptools._distutils), unchanged semantics."""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"invalid truth value {val!r}")


def install_distutils():
    if "distutils.util" in sys.modules:
        return
    try:
        import distutils.util  # noqa: F401  (setuptools' shim on Python >= 3.12)
    except Exception:
        d, u = types.ModuleType("distutils"), types.ModuleType("distutils.util")
        d.__path__ = []
        u.strtobool = _strtobool
        d.util = u
        sys.modules["distutils"], sys.modules["distutils.util"] = d, u
        os.environ["K1_DISTUTILS_SHIM"] = "1"


def install_paths():
    sys.dont_write_bytecode = True
    for p in (SRC, CODE):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "training.download_data" not in sys.modules:
        m = types.ModuleType("training.download_data")
        m.ChunkedArchive = None
        sys.modules["training.download_data"] = m
    install_distutils()


def memoise_install_native():
    import probe_infer as PI
    if getattr(PI, "_k1_memo", False):
        return
    orig, cache = PI.install_native, {}

    def install_native(lean=True):
        if lean not in cache:
            cache[lean] = orig(lean)
        return cache[lean]
    PI.install_native = install_native
    PI._k1_memo = True


install_paths()
import compact_infer as CI

assert os.path.dirname(os.path.abspath(CI.__file__)) == SRC, CI.__file__
PX_DUMP = os.environ.get("K1_PX_DUMP")


class LocalCStudies(CI.CStudies):
    def _setup(self):
        install_paths()
        memoise_install_native()
        try:
            super()._setup()
            if self.views in ANCHOR_VIEWS and self.bank is None:
                raise RuntimeError("K1: the SKM-TEA anchor bank did not load")
        except BaseException:
            self.native = None
            raise

    def __getitem__(self, i):
        out = super().__getitem__(i)
        if out.get("error") is None and self.views in ANCHOR_VIEWS and out.get("anchor") is None:
            out = dict(out, error="K1: item built without an SKM-TEA anchor", images=out["images"][:0], meta=out["meta"][:0])
        if PX_DUMP:
            try:
                img = out["images"].numpy()
                meta = out["meta"].numpy()
                rec = {"uid": out["uid"], "error": out["error"], "n": int(len(img)),
                       "img_sha256": hashlib.sha256(img.tobytes()).hexdigest(),
                       "meta_sha256": hashlib.sha256(meta.astype("float32").tobytes()).hexdigest(),
                       "win_sha1_10": [hashlib.sha1(w.tobytes()).hexdigest()[:10] for w in img],
                       "img_mean": float(img.mean()) if len(img) else None, "anchor": out.get("anchor"),
                       "seconds": out.get("seconds")}
            except Exception as exc:
                rec = {"uid": out.get("uid"), "hash_error": repr(exc)[:200]}
            with open(os.path.join(PX_DUMP, "px_%d.jsonl" % os.getpid()), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        return out


CI.CStudies = LocalCStudies

if __name__ == "__main__":
    CI.main()
