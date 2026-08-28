#!/usr/bin/env python3
"""
Rewrite every Einsum in the UNet into Transpose / Reshape / MatMul.

── why ──────────────────────────────────────────────────────────────────────

The model expresses its attention as 105 Einsum nodes, and that one op choice
costs the browser everything:

  fp16 Einsum   has NO kernel in ONNX Runtime's WebGPU backend, so all of the
                attention runs on single-threaded WASM. Measured on an M5 Pro:
                the GPU does 0.5s of a 2.4s step and idles the rest.
  fp32 Einsum   HAS a WebGPU kernel and it is WRONG for these contractions:
                forcing the Einsums to fp32 made the step 2.7x faster and the
                output garbage (corr 0.079 against the fp32 reference; the
                same file on the CPU provider scores 0.9987).

MatMul, Transpose and Reshape are the ops every WebGPU backend lives on, in
both precisions. So the graph is rewritten to use only those — same maths,
different spelling.

── the six contractions, and what they become ───────────────────────────────

The model is linear attention (contractions of K^T V over 4096 tokens), plus
a positional term. With shapes as inferred from the graph (b = batch):

  1. 'b u k l, b u v l -> b k v'    [b,u,k,l],[b,u,v,l]   u,l contracted
  2. 'b u k m, b u v m -> b k v'    same as 1 with m for l
  3. 'n m k u, b u v m -> b n k v'  [n,m,k,u],[b,u,v,m]   m,u contracted
  4. 'b h k n, b k v -> b h v n'    [b,h,k,n],[b,k,v]     k contracted
  5. 'b h k n, b k v n -> b h v n'  per-token matrix on the right
  6. 'b h k n, b n k v -> b h v n'  per-token matrix, already token-major

  1,2: A -> transpose [b,k,u,l] -> reshape [b,k,u*l]
       B -> transpose [b,u,l,v] -> reshape [b,u*l,v]
       MatMul -> [b,k,v]
  3:   A -> transpose [n,k,u,m] -> reshape [n*k, u*m]        (2D, constant)
       B -> transpose [b,u,m,v] -> reshape [b,u*m,v]
       MatMul broadcasts the 2D left over the batch -> [b,n*k,v]
       -> reshape [b,n,k,v]
  4:   B -> transpose [b,v,k] -> unsqueeze [b,1,v,k]
       MatMul([b,1,v,k],[b,h,k,n]) broadcasts over h -> [b,h,v,n]
  5,6: both sides to token-major [b,n,·,·], MatMul over [h,k]x[k,v],
       then back: A -> [b,n,h,k]; B -> [b,n,k,v] (5 needs a transpose,
       6 is already there); MatMul -> [b,n,h,v] -> transpose [b,h,v,n]

Every decomposition is validated against numpy.einsum on random tensors
BEFORE the graph is touched, and the rewritten model must match the original
on the CPU provider to fp32 exactness afterwards. A rewrite that cannot prove
itself does not get saved.

    .work/venv/bin/python tools/rewrite_einsum.py
"""

from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import onnxsim

HERE = Path(__file__).resolve().parent.parent
WORK = HERE / ".work"
SRC = WORK / "moebius-fp32" / "unet.onnx"
DST = WORK / "unet.noeinsum.onnx"


def dims_of(value_info) -> list:
    return [
        d.dim_value if d.HasField("dim_value") else d.dim_param
        for d in value_info.type.tensor_type.shape.dim
    ]


class Rewriter:
    """Emits the replacement nodes for one Einsum, with unique names."""

    def __init__(self):
        self.counter = 0
        self.nodes = []

    def name(self, op: str) -> str:
        self.counter += 1
        return f"einsum_rw/{op}_{self.counter}"

    def emit(self, op: str, inputs: list, attrs: dict | None = None) -> str:
        out = self.name(op)
        self.nodes.append(helper.make_node(op, inputs, [out], name=out, **(attrs or {})))
        return out

    def transpose(self, x: str, perm: list) -> str:
        return self.emit("Transpose", [x], {"perm": perm})

    def reshape(self, x: str, shape: list, graph_inits: list) -> str:
        shape_name = self.name("shape_const")
        graph_inits.append(
            numpy_helper.from_array(np.array(shape, dtype=np.int64), name=shape_name)
        )
        return self.emit("Reshape", [x, shape_name])

    def matmul(self, a: str, b: str) -> str:
        return self.emit("MatMul", [a, b])

    def unsqueeze(self, x: str, axis: int, graph_inits: list) -> str:
        ax = self.name("axes_const")
        graph_inits.append(
            numpy_helper.from_array(np.array([axis], dtype=np.int64), name=ax)
        )
        return self.emit("Unsqueeze", [x, ax])


def decompose(rw: Rewriter, eq: str, a: str, b: str, sa: list, sb: list, inits: list) -> str:
    """Return the name of the tensor holding the Einsum's result."""
    # Batch stays symbolic; 0 in a Reshape target means "copy that dim".
    if eq in ("b u k l, b u v l -> b k v", "b u k m, b u v m -> b k v"):
        _, u, k, l = sa
        _, _, v, _ = sb
        a2 = rw.reshape(rw.transpose(a, [0, 2, 1, 3]), [0, k, u * l], inits)
        b2 = rw.reshape(rw.transpose(b, [0, 1, 3, 2]), [0, u * l, v], inits)
        return rw.matmul(a2, b2)

    if eq == "n m k u, b u v m -> b n k v":
        n, m, k, u = sa
        _, _, v, _ = sb
        # The left side has no batch: collapse it to 2D and let MatMul's
        # numpy-style broadcasting stretch it over the batch.
        a2 = rw.reshape(rw.transpose(a, [0, 2, 3, 1]), [n * k, u * m], inits)
        b2 = rw.reshape(rw.transpose(b, [0, 1, 3, 2]), [0, u * m, v], inits)
        return rw.reshape(rw.matmul(a2, b2), [0, n, k, v], inits)

    if eq == "b h k n, b k v -> b h v n":
        bt = rw.unsqueeze(rw.transpose(b, [0, 2, 1]), 1, inits)  # [b,1,v,k]
        return rw.matmul(bt, a)  # broadcasts over h -> [b,h,v,n]

    if eq == "b h k n, b k v n -> b h v n":
        a2 = rw.transpose(a, [0, 3, 1, 2])  # [b,n,h,k]
        b2 = rw.transpose(b, [0, 3, 1, 2])  # [b,n,k,v]
        return rw.transpose(rw.matmul(a2, b2), [0, 2, 3, 1])

    if eq == "b h k n, b n k v -> b h v n":
        a2 = rw.transpose(a, [0, 3, 1, 2])  # [b,n,h,k]
        return rw.transpose(rw.matmul(a2, b), [0, 2, 3, 1])

    raise ValueError(f"no decomposition for {eq!r} with shapes {sa} {sb}")


# ── proof before surgery ─────────────────────────────────────────────────────

CASES = [
    # (equation, shape A, shape B) — real shapes from the graph, batch pinned
    # to 3 rather than 2 so a decomposition that hardcodes the batch fails.
    ("b u k l, b u v l -> b k v", (3, 1, 40, 10), (3, 1, 40, 10)),
    ("b u k m, b u v m -> b k v", (3, 1, 40, 96), (3, 1, 40, 96)),
    ("n m k u, b u v m -> b n k v", (96, 10, 40, 1), (3, 1, 40, 10)),
    ("b h k n, b k v -> b h v n", (3, 8, 40, 96), (3, 40, 40)),
    ("b h k n, b k v n -> b h v n", (3, 8, 40, 96), (3, 40, 40, 96)),
    ("b h k n, b n k v -> b h v n", (3, 8, 40, 96), (3, 96, 40, 40)),
]


def prove_each_case() -> None:
    rng = np.random.default_rng(7)
    for eq, sa, sb in CASES:
        A = rng.standard_normal(sa).astype(np.float32)
        B = rng.standard_normal(sb).astype(np.float32)
        want = np.einsum(eq, A, B)

        rw = Rewriter()
        inits = []
        # Symbolic batch in the test graph too, so Reshape's 0-dims are hit.
        sym_a = ["b"] + list(sa[1:]) if eq[0] == "b" else list(sa)
        sym_b = ["b"] + list(sb[1:])
        out = decompose(rw, eq, "A", "B", list(sa), list(sb), inits)
        graph = helper.make_graph(
            rw.nodes,
            "case",
            [
                helper.make_tensor_value_info("A", TensorProto.FLOAT, sym_a),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, sym_b),
            ],
            [helper.make_tensor_value_info(out, TensorProto.FLOAT, list(want.shape))],
            initializer=inits,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"A": A, "B": B})[0]
        diff = float(np.abs(got - want).max())
        rel = diff / float(np.abs(want).max())
        status = "ok" if rel < 1e-5 else "WRONG"
        print(f"  {eq!r:<36} max|d| {diff:.2e}  rel {rel:.2e}  {status}")
        if rel >= 1e-5:
            raise SystemExit(f"decomposition for {eq!r} does not match numpy")


# ── the surgery ──────────────────────────────────────────────────────────────

def rewrite(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return `model` with every Einsum replaced. Used by to_fp16.py."""

    # Pin the batch to 2 before inferring. The batch is not really dynamic —
    # classifier-free guidance always stacks the unconditional and conditional
    # halves — and the graph computes its attention shapes through Shape-op
    # plumbing that inference can only resolve with a real number in place
    # ('data_prop' below propagates those constants through). The rewritten
    # model is therefore batch-2 only, which is the only way it is ever run.
    for value in list(model.graph.input) + list(model.graph.output):
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                dim.ClearField("dim_param")
                dim.dim_value = 2

    # Fold the shape plumbing before rewriting. The graph computes attention
    # shapes with runtime Shape/Gather/Concat chains that static inference
    # cannot see through even with data propagation; onnxsim evaluates the
    # now-constant chains and deletes them (4087 plumbing nodes become ~400),
    # leaving concrete dimensions everywhere an Einsum reads them. Proven
    # separately to keep the output bit-comparable: same checksum on WebGPU
    # before and after folding.
    model, ok = onnxsim.simplify(model)
    if not ok:
        raise SystemExit("onnxsim could not validate the folded model")

    inferred = onnx.shape_inference.infer_shapes(model, data_prop=True)
    shapes = {v.name: dims_of(v) for v in inferred.graph.value_info}
    for v in list(model.graph.input) + list(model.graph.output):
        shapes[v.name] = dims_of(v)
    for init in model.graph.initializer:
        shapes[init.name] = list(init.dims)

    rw = Rewriter()
    inits: list = []
    new_nodes = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Einsum":
            new_nodes.append(node)
            continue
        eq = " ".join(
            [a.s.decode() for a in node.attribute if a.name == "equation"][0].split()
        )
        sa, sb = shapes.get(node.input[0]), shapes.get(node.input[1])
        if sa is None or sb is None:
            raise SystemExit(f"no inferred shape for {node.name} inputs; cannot rewrite")
        out = decompose(rw, eq, node.input[0], node.input[1], sa, sb, inits)
        # The replacement's last node must write the Einsum's own output name,
        # so every consumer downstream is untouched.
        rw.nodes[-1].output[0] = node.output[0]
        new_nodes.extend(rw.nodes)
        rw.nodes = []
        replaced += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(inits)
    onnx.checker.check_model(model, full_check=False)
    print(f"  replaced {replaced} Einsum nodes")
    return model


def rewrite_model() -> None:
    onnx.save(rewrite(onnx.load(str(SRC))), str(DST), save_as_external_data=False)
    print(f"  wrote {DST.name}")


def prove_whole_model() -> None:
    n = 2 * 9 * 64 * 64
    lat = np.cos(np.arange(n, dtype=np.float64) * 0.001).astype(np.float32)
    lat = lat.reshape(2, 9, 64, 64)
    ts = np.array([500, 500], dtype=np.int64)
    ids = np.zeros((2, 10), dtype=np.int64)
    for i in range(10):
        ids[0, i] = 10 + i
        ids[1, i] = i
    feed = {"latent": lat, "timesteps": ts, "input_ids": ids}

    ref = ort.InferenceSession(str(SRC), providers=["CPUExecutionProvider"])
    got = ort.InferenceSession(str(DST), providers=["CPUExecutionProvider"])
    a = ref.run(None, feed)[0].astype(np.float64).ravel()
    b = got.run(None, feed)[0].astype(np.float64).ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    maxd = float(np.abs(a - b).max())
    print(f"  whole model, fp32 vs fp32: corr {corr:.8f}  max|d| {maxd:.2e}")
    # fp32 -> fp32 with reassociated sums: tiny float noise only.
    if corr < 0.999999 or maxd > 1e-3:
        DST.unlink()
        raise SystemExit("rewritten model does not match the original; deleted")


if __name__ == "__main__":
    if not SRC.exists():
        sys.exit(f"no {SRC} — run fetch.sh first")
    print("proving each decomposition against numpy:")
    prove_each_case()
    print("rewriting the graph:")
    rewrite_model()
    print("proving the whole model:")
    prove_whole_model()
    print("done")
