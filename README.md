# verge

Static assets for **[lognjais.github.io/tools](https://lognjais.github.io/tools)**, continued.

There is no code here. Nothing is built. This repo is a bucket that GitHub Pages
happens to serve, exactly like its older sibling
[`edge`](https://github.com/lognjais/edge).

## Why a second bucket

Not because `edge` was organised badly. Because a GitHub Pages **site** may be no
larger than 1 GB, and `edge` already publishes 623 MB.

That limit is often confused with the other one, so both are written down here:

| Limit | Applies to | What gets around it |
|---|---|---|
| 100 MiB | one **file**, refused by `git push` | splitting into `.partN`, rejoined in the browser |
| 1 GB | one published **site** | nothing — a second site |

Splitting does not help the second limit. Ten chunks of 124 MB still add up to
1.24 GB. So the only way to serve a model that does not fit is another Pages
site, which is what this is.

## Why this keeps the promise

The tools app promises that nothing you open in it is ever uploaded, and part of
that promise is that no third party learns you opened it. GitHub serves every
project site of a user from **one host**, so:

```
lognjais.github.io/tools/     the app
lognjais.github.io/edge/      weights
lognjais.github.io/verge/     weights, continued
```

are all the **same origin**. No CORS boundary, no preflight, no third party — the
same reason `edge` exists rather than a jsDelivr or huggingface.co URL. Adding a
repo costs nothing here; adding a CDN would cost the promise.

## The name

`edge`, `verge`. Synonyms, not numbers. A third would be `brink` or `ledge`.
`edge2` would have to be renamed the moment there was an `edge3`.

## What is here

| Path | What | Licence |
|---|---|---|
| `models/moebius/` | Moebius 0.2B inpainting, ONNX, fp16 | Apache-2.0 |

### moebius

The quality tier of `/remove-watermark`. A 0.2B latent diffusion inpainter from
HUST-VL, converted to ONNX by simonw and served here as **fp16**.

fp16 is not a preference, it is the only version that fits. At fp32 the three
files are 864 + 130 + 188 MB = 1.15 GB, which is over the site limit **on its
own**, before anything else is published here. Halved, they are ~591 MB, which
leaves room for whatever comes next.

Upstream's own note is that fp16 is "possible but requires validation", so it is
validated in `fetch.sh` rather than assumed: the fp16 and fp32 graphs are run on
the same input and the outputs compared before anything is committed. The
measured difference is recorded next to the files.

Three graphs, run in this order per denoising step:

```
vae_encoder   the masked image -> a 64x64 latent
unet          20 DDIM steps, 9-channel input, classifier-free guidance
vae_decoder   the finished latent -> 512x512 pixels
```

The scaling factor is **0.13025**, not Stable Diffusion's usual 0.18215. Using
the familiar one produces a washed-out image that looks like a bad seed rather
than a bug.
