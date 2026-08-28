"""
Halve Moebius so it fits a GitHub Pages site, and prove every graph still loads.

fp32 is 1.24 GB and a published Pages site may be no larger than 1 GB, so this
is not an optimisation — without it the model cannot be served at all, in this
repo or any other.

Upstream's note is that fp16 is "possible but requires validation". It earned
that three times over here, and all three failures produce a file of exactly
the expected size that will not build a session:

  1. disable_shape_infer=True
       Type parameter (T) of Optype (Div) bound to different types
       (tensor(float16) and tensor(float)) in /encoder/mid_block/attentions.0/Div_1
  2. shape inference on, nothing blocked
       Type (tensor(float16)) of output arg (…/attentions.0/Cast_2_output_0)
       does not match expected type (tensor(float))
  3. same, on the UNet, in a different place
       …/m/diff_model/time_proj/Cast does not match expected type (tensor(float))

The cause in every case is a Cast the original exporter placed deliberately —
around VAE mid-block attention, and around the UNet's timestep projection —
whose fixed float32 type the converter cannot honour once its neighbours are
halved. So those specific nodes are left in fp32 and everything else converts.

Blocking is kept as NARROW as possible, because it costs size. Blocking every
node with "attn" in the name also works and leaves the UNet at 749 MB, which
is 289 MB worse for no benefit. The 23 timestep nodes are the ones that matter.

  unet         907 MB -> 460 MB   23 timestep nodes kept fp32
  vae_decoder  198 MB -> ~100 MB  attention nodes kept fp32
  vae_encoder  137 MB ->  71 MB   attention nodes kept fp32

keep_io_types=True, so the graphs still take and return float32 and the
browser never learns which precision it loaded.
"""
import os
import sys

import onnx
import onnxruntime as ort
from onnxconverter_common import float16

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rewrite_einsum  # noqa: E402  (needs HERE on the path first)
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, ".work", "moebius-fp32")
DST = os.path.join(ROOT, "models", "moebius")

# Substrings of node names to leave in fp32, per model. Narrow on purpose.
BLOCK = {
    "unet": ("time_proj", "time_embed"),
    "vae_encoder": ("attentions",),
    "vae_decoder": ("attentions",),
}

os.makedirs(DST, exist_ok=True)

for name, keys in BLOCK.items():
    src = f"{SRC}/{name}.onnx"
    dst = f"{DST}/{name}.onnx"
    if os.path.exists(dst):
        print(f"{name}: already converted")
        continue

    print(f"{name}: loading…", flush=True)
    model = onnx.load(src)

    # The UNet's attention is rewritten out of Einsum first. Not an
    # optimisation that can be skipped: fp16 Einsum has no WebGPU kernel at
    # all, so leaving it in puts every attention node on the single-threaded
    # CPU path and the browser spends 2.4s a step with the GPU idle. See
    # rewrite_einsum.py, which proves each contraction against numpy before
    # touching the graph.
    if name == "unet":
        model = rewrite_einsum.rewrite(model)

    blocked = [n.name for n in model.graph.node if any(k in n.name for k in keys)]

    print(f"{name}: converting, {len(blocked)} nodes kept fp32…", flush=True)
    model16 = float16.convert_float_to_float16(
        model, keep_io_types=True, node_block_list=blocked
    )
    onnx.save(model16, dst, save_as_external_data=False)

    # Build a session immediately. A conversion that cannot load is worse than
    # no conversion, and finding out two files later wastes the whole run.
    try:
        ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    except Exception as err:
        os.remove(dst)
        print(f"{name}: FAILED to load after conversion — {err}")
        sys.exit(1)

    a, b = os.path.getsize(src), os.path.getsize(dst)
    print(f"{name}: {a/1e6:.0f} MB -> {b/1e6:.0f} MB ({100*b/a:.0f}%), loads cleanly", flush=True)

served = sum(os.path.getsize(f"{DST}/{n}.onnx") for n in BLOCK)
print(f"\nserved total: {served/1e6:.0f} MB  (1 GB is the Pages site cap)")
