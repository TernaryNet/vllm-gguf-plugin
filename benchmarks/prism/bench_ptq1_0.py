"""Time PTQ1_0 matmul strategies on Bonsai-27B layer shapes."""

import sys
import time

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType as T

import vllm_gguf_plugin.ops as ops
from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton


def make_weight(n, k):
    rng = np.random.default_rng(0)
    trits = rng.integers(-1, 2, size=(n, k)).astype(np.float32)
    w = gguf.quantize(trits, T.PTQ1_0)
    return torch.tensor(w, device="cuda")


def bench(fn, iters=10):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


shapes = [
    ("gate_up 5120->34816", 34816, 5120),
    ("down 17408->5120", 5120, 17408),
    ("qkvz 5120->16384", 16384, 5120),
    ("o_proj 6144->5120", 5120, 6144),
]
Ms = [
    int(m)
    for m in (
        sys.argv[1:] or ["1", "2", "4", "8", "16", "32", "64", "128", "256", "1024"]
    )
]
print(
    f"{'shape':24s} {'M':>5s} {'mmvq ms':>9s} {'triton ms':>10s} {'dequant+mm ms':>14s}"
)


def time_strategies(w, x, n, k):
    def mmvq():
        return ops.ggml_mul_mat_vec_a8(w, x, int(T.PTQ1_0), n)

    def triton_gemm():
        return ggml_mul_mat_a8_triton(w, x, int(T.PTQ1_0), n)

    def dequant_mm():
        d = ops.ggml_dequantize(w, int(T.PTQ1_0), n, k, torch.bfloat16)
        return x @ d.T

    m = x.shape[0]
    t_mmvq = bench(mmvq) if m <= 64 else float("nan")
    return t_mmvq, bench(triton_gemm), bench(dequant_mm)


for name, n, k in shapes:
    w = make_weight(n, k)
    for m in Ms:
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        t_mmvq, t_tri, t_deq = time_strategies(w, x, n, k)
        print(
            f"{name:24s} {m:5d} {t_mmvq:9.3f} {t_tri:10.3f} {t_deq:14.3f}", flush=True
        )
