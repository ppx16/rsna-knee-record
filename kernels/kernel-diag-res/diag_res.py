"""CPU-only: how much real detail is there above 224 px in the corpus slices?

WHY (HANDOFF 92)
    The corpus stores 336x336 slices but the baseline trains at --res 224, and CV EXP #2
    (`cv-r288-fold0`) proposes 288 at ~10 GPU-hours.  That is only worth spending if the
    corpus actually CONTAINS detail above 224.  If the slices were themselves built by
    upscaling something smaller, --res 288 buys nothing and the experiment is wasted.

    This measures it directly and needs no GPU slot: for each target resolution R, resample a
    slice down to R and back up to 336, then compare with the original.

        PSNR(R) high and flat   -> an R-resolution image already carries (almost) everything
                                   the 336 slice has.  Training above that R is wasted.
        PSNR(R) still climbing  -> there IS detail above R, so the experiment has headroom.

    Round-trip PSNR is the standard way to ask "what is the effective resolution of this
    array"; the knee of the curve is the answer.

OUTPUT
    * the round-trip PSNR curve at R = 112 .. 320
    * mean |Laplacian| (a sharpness proxy) and lag-1 pixel autocorrelation, which both jump
      when an image has been upsampled
    * an explicit VERDICT line keyed to the two thresholds that matter for the decision
"""
import glob
import time

import numpy as np
from PIL import Image


def find(name):
    hits = sorted(glob.glob("/kaggle/input/**/" + name, recursive=True))
    return hits[0] if hits else None


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * np.log10(255.0 ** 2 / mse))


def roundtrip(img, R):
    """336 -> R -> 336, bilinear both ways (close enough to F.interpolate)."""
    d = np.asarray(Image.fromarray(img).resize((R, R), Image.BILINEAR))
    u = np.asarray(Image.fromarray(d).resize(img.shape, Image.BILINEAR))
    return u


def main():
    t0 = time.time()
    p = find("all_vols.npy")
    m = find("all_masks.npy")
    print("all_vols -> %s" % p, flush=True)
    if not p:
        raise SystemExit("corpus not mounted")
    vols = np.load(p, mmap_mode="r")
    masks = np.load(m, mmap_mode="r") if m else None
    N, D, H, W = vols.shape
    print("shape %s  (%d studies, %d slices, %dx%d)" % (vols.shape, N, D, H, W), flush=True)

    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=40, replace=False)
    slices = []
    for i in idx:
        v = vols[i]
        if masks is not None:
            valid = np.where(np.asarray(masks[i]) > 0)[0]
            c = int(valid[len(valid) // 2]) if len(valid) else D // 2
        else:
            c = D // 2
        slices.append(np.asarray(v[c]))
    S = np.stack(slices)
    print("sampled %d slices" % len(S), flush=True)

    print("\n== round-trip PSNR: down to R then back up to %d ==" % H, flush=True)
    print("  %-6s %8s %8s %10s" % ("R", "PSNR", "dPSNR", "resid std"), flush=True)
    prev = None
    curve = {}
    for R in (112, 140, 168, 196, 224, 256, 288, 320):
        v = float(np.mean([psnr(s, roundtrip(s, R)) for s in S]))
        resid = float(np.mean([np.std(s.astype(np.float64) - roundtrip(s, R)) for s in S]))
        d = "" if prev is None else "%+.2f" % (v - prev)
        print("  %-6d %8.2f %8s %10.2f" % (R, v, d, resid), flush=True)
        curve[R] = v
        prev = v

    lap = np.mean([np.abs(np.diff(s.astype(np.float64), axis=0)).mean() for s in S])
    # lag-1 autocorrelation along both axes
    ac = []
    for s in S:
        x = s.astype(np.float64)
        x = x - x.mean()
        den = (x * x).sum()
        if den > 0:
            ac.append(((x[:-1, :] * x[1:, :]).sum() + (x[:, :-1] * x[:, 1:]).sum()) / (2 * den))
    print("\n== texture statistics ==", flush=True)
    print("  mean |d/dy| (sharpness proxy) : %.3f" % lap, flush=True)
    print("  lag-1 autocorrelation        : %.4f" % float(np.mean(ac)), flush=True)
    print("  (an UPSAMPLED array pushes autocorrelation toward ~0.999 and |d/dy| toward ~0)",
          flush=True)

    # The decision this exists to make.
    p224, p288, p320 = curve[224], curve[288], curve[320]
    print("\n== VERDICT for cv-r288-fold0 ==", flush=True)
    print("  PSNR(224)=%.2f  PSNR(288)=%.2f  PSNR(320)=%.2f" % (p224, p288, p320), flush=True)
    if p288 - p224 < 0.5:
        print("  -> 224 already carries essentially everything above it: **--res 288 has almost", flush=True)
        print("     no headroom**; treat a flat r288 result as EXPECTED, not as a failed run.", flush=True)
    elif p288 - p224 < 1.5:
        print("  -> small but real headroom between 224 and 288 (~%.2f dB)." % (p288 - p224), flush=True)
        print("     r288 is a fair test; expect a small effect at best.", flush=True)
    else:
        print("  -> REAL headroom above 224 (~%.2f dB): the 224 baseline is throwing away" % (p288 - p224),
              flush=True)
        print("     detail the corpus still has. r288 is well motivated; consider 336 too.", flush=True)
    print("\nDONE in %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
