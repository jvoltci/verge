#!/usr/bin/env bash
#
# Rebuild everything this repo serves, from scratch.
#
# Nothing here is hand-placed. If a file is in this repo it came from the
# source named below, and this script is the record of which. Same contract as
# the edge repo's fetch.sh, which this is deliberately a sibling of.
#
#   models/moebius/   <- huggingface.co/simonw/Moebius-ONNX, converted to fp16
#
# ── why this repo exists at all ──────────────────────────────────────────────
#
# Two different GitHub limits get confused with each other, and only one of
# them is solved by splitting a file:
#
#   100 MiB   per FILE, refused by git push.  Fixed by .partN + parts.json,
#             rejoined in the browser by the tools repo's model-parts.ts.
#   1 GB      per published Pages SITE.       Not fixed by splitting — ten
#             chunks of 124 MB still add up to 1.24 GB. Only a second site
#             fixes it, and this is that second site.
#
# edge already publishes 623 MB. Moebius at fp32 is 1.16 GB, which is over the
# site limit on its own. Hence: a new site, and fp16.
#
# ── why fp16 is not a preference ─────────────────────────────────────────────
#
#   unet         907 MB -> ~453 MB
#   vae_decoder  198 MB ->  ~99 MB
#   vae_encoder  137 MB ->  ~68 MB
#   total       1242 MB -> ~620 MB
#
# fp32 does not fit under the 1 GB cap even alone. fp16 does, with room to
# spare for whatever is published here next. Upstream's own note is that fp16
# is "possible but requires validation", so validate.py runs both precisions on
# the same input and compares the decoded pixels before anything is committed.
#
# The conversion keeps float32 inputs and outputs (keep_io_types=True) so the
# browser code does not have to know which precision it loaded.

set -euo pipefail
cd "$(dirname "$0")"

HF="https://huggingface.co/simonw/Moebius-ONNX/resolve/main"
FILES=(unet.onnx vae_encoder.onnx vae_decoder.onnx)

# Split anything at or above this, in MiB. Same 90 as edge, and for the same
# reason: ten MiB of headroom under GitHub's 100 MiB refusal.
PART_MIB=90

echo "==> downloading fp32 sources (1.2 GB, not committed)"
mkdir -p .work/moebius-fp32
for f in "${FILES[@]}"; do
  if [ ! -s ".work/moebius-fp32/$f" ]; then
    curl -fL --retry 3 -o ".work/moebius-fp32/$f" "$HF/$f"
  fi
  printf '    %10d  %s\n' "$(wc -c <".work/moebius-fp32/$f")" "$f"
done

echo "==> converting to fp16 and validating"
python3 -m venv .work/venv 2>/dev/null || true
.work/venv/bin/pip install -q onnx onnxruntime onnxconverter-common numpy pillow
.work/venv/bin/python tools/to_fp16.py
.work/venv/bin/python tools/validate.py

echo "==> splitting anything over ${PART_MIB} MiB"
LIMIT=$((PART_MIB * 1024 * 1024))
python3 - "$LIMIT" <<'PY'
import json, os, sys
limit = int(sys.argv[1])
base = "models"
manifest = {}
for root, _, files in os.walk(base):
    for name in sorted(files):
        if name.endswith(".onnx"):
            path = os.path.join(root, name)
            size = os.path.getsize(path)
            if size <= limit:
                continue
            rel = os.path.relpath(path, base)
            parts = 0
            with open(path, "rb") as fh:
                while True:
                    chunk = fh.read(limit)
                    if not chunk:
                        break
                    with open(f"{path}.part{parts}", "wb") as out:
                        out.write(chunk)
                    parts += 1
            manifest[rel] = {"parts": parts, "bytes": size}
            os.remove(path)
            print(f"    {rel}: {parts} parts, {size} bytes")
with open(os.path.join(base, "parts.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)
    fh.write("\n")
PY

echo "==> total $(du -sh models | awk '{print $1}') in models/"
echo "    (the 1 GB Pages site cap is the number to watch)"
