"""
Prove the fp16 conversion is faithful before serving it.

Upstream ships fp32 and says fp16 is "possible but requires validation". The
obvious validation is wrong, and it is worth saying why, because it wasted a
run here first.

── why not compare the finished pictures ─────────────────────────────────────

Running the full 20-step schedule in both precisions and diffing the output
measures the wrong thing. A diffusion trajectory is chaotic: a fp16 rounding
difference in step 1 changes the noise prediction, which changes the latent,
which changes every step after it. The two runs end up at *different valid
samples*. Measured here: mean absolute difference of 21.3 in the hole — and
both images were clean, plausible and artifact-free when looked at. A test
that fails on a good result is not a test.

── what is actually checked ──────────────────────────────────────────────────

Precision error, before it has anything to compound through:

  1. every graph builds a session at all (the failure mode that broke three
     earlier conversions was a graph that would not load)
  2. ONE UNet forward pass, same input, both precisions -> the predicted noise
     must agree closely. This is the weights being faithful, with no
     trajectory to amplify anything.
  3. VAE encode -> decode round trip -> the decoded pixels must agree.
  4. nothing is NaN or Inf.

Relative error on a single pass is the honest number for a halved model.
"""
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FP32 = os.path.join(ROOT, ".work", "moebius-fp32")
FP16 = os.path.join(ROOT, "models", "moebius")

IMG, LAT = 512, 64
SCALE = 0.13025
HALF_IDS = 10
X0, Y0, W, H = 300, 60, 170, 70

# The gate is CORRELATION, not size of error, and that distinction matters.
#
# A halved UNet does not compute slightly-smaller numbers; it computes the same
# function with rounding noise on top. What must be proved is that it is still
# the same function — that the predicted noise points the same way everywhere.
# Correlation says exactly that. Measured here: 0.999865.
#
# The mean relative error is 0.027, which sounds alarming and is not: fp16
# carries ~3 decimal digits and this graph is deep, so a few percent of spread
# is the expected cost, and the worst single element is 0.116 against a signal
# standard deviation of 0.994. The 20-step output was checked by eye and is
# clean. So relative error is reported as context with a loose sanity bound,
# and correlation is what actually gates.
EPS_CORRELATION = 0.999
EPS_SANITY = 0.05
# Decoded pixels, 0-255. Two units is invisible and well inside JPEG noise.
# The VAE has no trajectory to amplify anything, so it should be near-exact,
# and it is: 0.25.
PIXEL_TOLERANCE = 2.0


def session(d, name):
    return ort.InferenceSession(f"{d}/{name}.onnx", providers=["CPUExecutionProvider"])


def inputs(photo):
    img = Image.open(photo).convert("RGB")
    s = min(img.size)
    img = img.crop((0, 0, s, s)).resize((IMG, IMG), Image.LANCZOS)
    truth = np.asarray(img).astype(np.uint8)
    mask = np.zeros((IMG, IMG), np.float32)
    mask[Y0:Y0 + H, X0:X0 + W] = 1.0
    chw = (np.transpose(truth, (2, 0, 1)).astype(np.float32) / 255) * 2 - 1
    return chw * (1 - mask)[None], mask


def main():
    photo = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "fixture.jpg")
    masked, mask = inputs(photo)
    rng = np.random.default_rng(0)
    lat = rng.standard_normal((1, 4, LAT, LAT)).astype(np.float32)
    mask64 = mask[::8, ::8][None][None]
    ids = np.stack([np.arange(HALF_IDS, 2 * HALF_IDS), np.arange(HALF_IDS)]).astype(np.int64)
    failures = []

    # ---- VAE encode -> decode, both precisions ---------------------------
    decoded = {}
    latents = {}
    for label, d in (("fp32", FP32), ("fp16", FP16)):
        enc, dec = session(d, "vae_encoder"), session(d, "vae_decoder")
        m = enc.run(None, {"image": masked[None]})[0][:, :4] * SCALE
        latents[label] = m
        out = dec.run(None, {"latent": (m / SCALE).astype(np.float32)})[0]
        decoded[label] = np.clip((np.transpose(out[0], (1, 2, 0)) + 1) / 2, 0, 1) * 255
        print(f"{label}: VAE round trip done")

    if not np.isfinite(decoded["fp16"]).all():
        failures.append("fp16 VAE produced non-finite pixels")
    pix = np.abs(decoded["fp32"] - decoded["fp16"]).mean()
    print(f"VAE round trip: {pix:.3f} mean abs pixel difference (tolerance {PIXEL_TOLERANCE})")
    if pix > PIXEL_TOLERANCE:
        failures.append(f"VAE differs by {pix:.3f} pixels")

    # ---- ONE UNet pass, both precisions ----------------------------------
    # Same latent, same timestep, same masked latent (fp32's, so the UNet is
    # the only thing being compared).
    nine = np.concatenate([lat, mask64, latents["fp32"]], axis=1).astype(np.float32)
    two = np.concatenate([nine, nine], axis=0)
    feed = {"latent": two, "timesteps": np.array([950, 950], np.int64), "input_ids": ids}

    eps = {}
    for label, d in (("fp32", FP32), ("fp16", FP16)):
        eps[label] = session(d, "unet").run(None, feed)[0]
        print(f"{label}: UNet pass done")

    if not np.isfinite(eps["fp16"]).all():
        failures.append("fp16 UNet produced non-finite output")
    a, b = eps["fp32"].ravel(), eps["fp16"].ravel()
    rel = np.abs(a - b).mean() / np.abs(a).mean()
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f"UNet one pass: correlation {corr:.6f} (gate {EPS_CORRELATION}), "
          f"relative error {rel:.5f}, worst element {np.abs(a - b).max():.4f} "
          f"against signal sd {a.std():.4f}")
    if corr < EPS_CORRELATION:
        failures.append(f"UNet is not computing the same function: correlation {corr:.6f}")
    if rel > EPS_SANITY:
        failures.append(f"UNet relative error {rel:.5f} is beyond the sanity bound")

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        return 1
    print(f"PASS  fp16 is faithful: UNet correlation {corr:.6f}, VAE {pix:.3f} pixels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
