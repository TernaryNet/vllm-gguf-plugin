"""Time PTQ1_0 matmul strategies on Bonsai-27B layer shapes (CUDA-graph timed)."""

import sys
import time

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType as T

import vllm_gguf_plugin.ops as ops
from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton

SHAPES = [
    ("gate_up 5120->34816", 34816, 5120),
    ("down 17408->5120", 5120, 17408),
    ("qkvz 5120->16384", 16384, 5120),
    ("o_proj 6144->5120", 5120, 6144),
]


# Enough copies of each weight that the graph's working set exceeds the L2
# cache, so the kernels stream from HBM like they do in a real decode step.
WORKING_SET_BYTES = 192 << 20


def make_weights(n, k):
    rng = np.random.default_rng(0)
    trits = rng.integers(-1, 2, size=(n, k)).astype(np.float32)
    packed = torch.tensor(gguf.quantize(trits, T.PTQ1_0), device="cuda")
    copies = max(1, -(-WORKING_SET_BYTES // packed.numel()))
    return [packed.clone() for _ in range(copies)]


def bench(fn, weights, iters=10):
    """Milliseconds per call, measured on CUDA-graph replays over ``weights``."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn(weights[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for i in range(iters):
            fn(weights[i % len(weights)])
    graph.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / (5 * iters) * 1e3


def time_strategies(weights, x, n, k):
    def mmvq(w):
        return ops.ggml_mul_mat_vec_a8(w, x, int(T.PTQ1_0), n)

    def mmv_pt(w):
        return ops.ggml_mul_mat_vec_ptq1_0(w, x, n)

    def triton_gemm(w):
        return ggml_mul_mat_a8_triton(w, x, int(T.PTQ1_0), n)

    def dequant_mm(w):
        d = ops.ggml_dequantize(w, int(T.PTQ1_0), n, k, torch.bfloat16)
        return x @ d.T

    m = x.shape[0]
    t_mmvq = bench(mmvq, weights) if m <= 64 else float("nan")
    t_pt = bench(mmv_pt, weights) if m <= ops.PTQ1_0_MMV_MAX_ROWS else float("nan")
    return t_mmvq, t_pt, bench(triton_gemm, weights), bench(dequant_mm, weights)


def main():
    rows = [
        int(v)
        for v in (sys.argv[1:] or ["1", "2", "4", "8", "16", "32", "64", "128", "256"])
    ]
    print(
        f"{'shape':24s} {'M':>5s} {'mmvq ms':>9s} {'pt-mmv ms':>9s} {'triton ms':>10s} "
        f"{'dequant+mm ms':>14s}"
    )
    for name, n, k in SHAPES:
        weights = make_weights(n, k)
        for m in rows:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            t_mmvq, t_pt, t_tri, t_deq = time_strategies(weights, x, n, k)
            print(
                f"{name:24s} {m:5d} {t_mmvq:9.3f} {t_pt:9.3f} "
                f"{t_tri:10.3f} {t_deq:14.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
