"""GPU checks and micro-bench for the PTQ1_0 int8 tensor-core GEMM (prefill tier).

    PYTHONPATH=<plugin worktree> python tests/run_prefill_checks.py --check [--bench] [--json out.json]

--check: ``ggml_mul_mat_ptq1_0_mma`` (every variant whose N tile fits) against the Triton tier
``ggml_gemm_ptq1_0_triton`` on the same inputs -- expected bit-identical (``torch.equal``) --
and both against a float64 reference of the bf16-dequantized product; plus the dispatcher
(``_fused_mul_mat_gguf``) at 300 rows. Weights are random bytes with a valid fp16 scale: every
byte decodes to some digits with the same formula in every kernel, so equality still tests the
whole data path.

--bench: CUDA-graph timed, weights cycled through enough copies to exceed L2, for the Bonsai
shapes at several M: dequantize + cuBLAS (the current > 256 tier), the Triton tier (M <= 512),
and each mma variant. Prints a table; --json writes the raw numbers.
"""

import argparse
import json
import sys
import time

import torch

import vllm_gguf_plugin.ops as ops
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
from vllm_gguf_plugin.triton.gemm.prism_quant.ptq1_0 import ggml_gemm_ptq1_0_triton

PTQ1_0 = 143
SHAPES = {
    "gate_up": (34816, 5120),
    "qkvz": (16384, 5120),
    "qkv": (14336, 5120),
    "down": (5120, 17408),
    "out": (5120, 6144),
}
VARIANTS = [0, 1, 2, 3, 4]
# + 16: one activation scale per 128 elements (not the Triton tier's numerics; error compared
# against the float64 reference instead of equality)
SB128_VARIANTS = [16, 17, 18, 19, 20]


def random_ptq1_0(n, k, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    blocks = torch.randint(
        0, 256, (n, k // 128, 28), dtype=torch.uint8, device=device, generator=gen
    )
    scale = (
        torch.rand((n, k // 128), device=device, generator=gen) * 0.04 + 0.01
    ).to(torch.float16)
    blocks[:, :, 26:28] = scale.view(torch.uint8).view(n, k // 128, 2)
    return blocks.view(n, k // 128 * 28).contiguous()


def variant_fits(variant, n):
    return n % (256 if variant % 16 == 4 else 128) == 0


def check(args, report):
    torch.manual_seed(0)
    cases = [
        (256, 5120, [16, 100, 257, 300, 512, 2048]),
        (128, 6144, [257, 1000]),
        (384, 17408, [257, 777]),
        (5120, 5120, [2048]),
        (512, 1024, [129, 4096]),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    fails = 0
    rows = []
    for n, k, ms in cases:
        w = random_ptq1_0(n, k, seed=n + k)
        dense64 = ops.ggml_dequantize(w, PTQ1_0, n, k, torch.float32).double()
        for m in ms:
            for dtype in dtypes:
                x = torch.randn((m, k), device="cuda", dtype=torch.float32)
                # a few large outliers per row, like real activations
                x[:, :: 997] *= 20.0
                x = x.to(dtype)
                tri = ggml_gemm_ptq1_0_triton(w, x, n)
                ref = x.double() @ dense64.T
                scale = ref.abs().max().item()
                err_tri = (tri.double() - ref).abs().max().item() / scale
                for v in args.variants:
                    if not variant_fits(v, n):
                        continue
                    out = ops.ggml_mul_mat_ptq1_0_mma(w, x, n, v)
                    equal = torch.equal(out, tri) if v < 16 else None
                    nbad = int((out != tri).sum().item())
                    err = (out.double() - ref).abs().max().item() / scale
                    row = {
                        "n": n, "k": k, "m": m, "dtype": str(dtype).split(".")[-1],
                        "variant": v, "equal_triton": equal, "mismatches": nbad,
                        "max_rel_err_vs_f64": err, "triton_max_rel_err_vs_f64": err_tri,
                    }
                    rows.append(row)
                    if v < 16 and not equal:
                        fails += 1
                        print(f"MISMATCH {row}", flush=True)
                    if v >= 16 and err > 3 * err_tri + 1e-6:
                        fails += 1
                        print(f"SB128 ERROR TOO LARGE {row}", flush=True)
        print(f"check n={n} k={k}: done ({fails} failing so far)", flush=True)
    # the dispatcher at 300 rows takes the mma tier and matches the Triton tier
    w = random_ptq1_0(512, 5120, seed=7)
    x = torch.randn((300, 5120), device="cuda", dtype=torch.bfloat16)
    disp = _fused_mul_mat_gguf(x, w, PTQ1_0)
    disp_equal = torch.equal(disp, ggml_gemm_ptq1_0_triton(w, x, 512))
    print(f"dispatcher 300 rows equal to Triton tier: {disp_equal}")
    fails += 0 if disp_equal else 1
    worst = max((r["max_rel_err_vs_f64"] for r in rows if r["variant"] < 16), default=0.0)
    worst_128 = max((r["max_rel_err_vs_f64"] for r in rows if r["variant"] >= 16), default=0.0)
    worst_tri = max(r["triton_max_rel_err_vs_f64"] for r in rows)
    ratios = [r["max_rel_err_vs_f64"] / r["triton_max_rel_err_vs_f64"] for r in rows if r["variant"] >= 16]
    if ratios:
        ratios.sort()
        print(f"SB128 error / Triton-tier error: median {ratios[len(ratios) // 2]:.2f}, max {ratios[-1]:.2f}")
    print(
        f"CHECK {'PASS' if fails == 0 else 'FAIL'}: {len(rows)} kernel cases, "
        f"{sum(bool(r['equal_triton']) for r in rows)} of {sum(r['variant'] < 16 for r in rows)} per-32 cases "
        f"bit-identical to the Triton tier; max rel err vs float64: mma {worst:.3e}, per-128 {worst_128:.3e}, "
        f"Triton {worst_tri:.3e}"
    )
    report["check"] = {"fails": fails, "rows": rows, "dispatcher_equal": disp_equal}
    return fails == 0


def bench_ms(fn, weights, iters):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for i in range(2):
            fn(weights[i % len(weights)])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for i in range(iters):
            fn(weights[i % len(weights)])
    graph.replay()
    torch.cuda.synchronize()
    reps = 5
    t = time.perf_counter()
    for _ in range(reps):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t) / (reps * iters) * 1e3
    del graph
    torch.cuda.empty_cache()
    return elapsed


def bench(args, report):
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    results = []
    names = args.shapes.split(",")
    ms = [int(v) for v in args.rows.split(",")]
    print(f"device {torch.cuda.get_device_name(0)}, L2 {l2 >> 20} MB")
    variants = args.variants
    print("| shape | M | dequant+cuBLAS ms | Triton ms | " + " | ".join(f"mma v{v} ms" for v in variants) + " | best v32 TOPS | best v32/old | best v128/old |")
    print("|---|---|---|---|" + "---|" * len(variants) + "---|---|---|")
    for name in names:
        n, k = SHAPES[name]
        base = random_ptq1_0(n, k, seed=1)
        copies = max(2, -(-3 * l2 // base.numel()))
        weights = [base] + [base.clone() for _ in range(copies - 1)]
        for m in ms:
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            iters = max(3, min(20, int(2e10 / (2 * m * n * k) * 20)))

            def old(w):
                d = ops.ggml_dequantize(w, PTQ1_0, n, k, torch.bfloat16)
                return x @ d.T

            def tri(w):
                return ggml_gemm_ptq1_0_triton(w, x, n)

            row = {"shape": name, "n": n, "k": k, "m": m}
            row["old_ms"] = bench_ms(old, weights, iters) if not args.no_old else float("nan")
            row["triton_ms"] = bench_ms(tri, weights, iters) if m <= 512 and not args.no_old else None
            for v in variants:
                if variant_fits(v, n):
                    row[f"mma{v}_ms"] = bench_ms(
                        lambda w, v=v: ops.ggml_mul_mat_ptq1_0_mma(w, x, n, v),
                        weights,
                        iters,
                    )
            t32 = [row[f"mma{v}_ms"] for v in variants if v < 16 and f"mma{v}_ms" in row]
            t128 = [row[f"mma{v}_ms"] for v in variants if v >= 16 and f"mma{v}_ms" in row]
            best = min(t32) if t32 else float("nan")
            best128 = min(t128) if t128 else float("nan")
            row["best_mma_tops"] = 2 * m * n * k / (best * 1e-3) / 1e12
            row["best_over_old"] = best / row["old_ms"]
            row["best128_over_old"] = best128 / row["old_ms"]
            row["best128_tops"] = 2 * m * n * k / (best128 * 1e-3) / 1e12
            results.append(row)
            cells = [f"{row['old_ms']:.3f}", "-" if row["triton_ms"] is None else f"{row['triton_ms']:.3f}"]
            cells += [f"{row[f'mma{v}_ms']:.3f}" if f"mma{v}_ms" in row else "-" for v in variants]
            print(f"| {name} | {m} | " + " | ".join(cells) + f" | {row['best_mma_tops']:.0f} | {row['best_over_old']:.2f} | {row['best128_over_old']:.2f} |", flush=True)
        del weights, base
        torch.cuda.empty_cache()
    report["bench"] = results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--shapes", default="gate_up,qkvz,down,out")
    parser.add_argument("--rows", default="128,256,257,512,1024,2048")
    parser.add_argument("--variants", default="0,1,2,3,4")
    parser.add_argument("--no-old", action="store_true", help="skip the dequant + cuBLAS and Triton timings")
    parser.add_argument("--json")
    args = parser.parse_args()
    args.variants = [int(v) for v in args.variants.split(",")]
    if not ops.ptq1_0_mma_available():
        print("REFUSED: ggml_mul_mat_ptq1_0_mma not in this build", file=sys.stderr)
        return 2
    report = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    ok = True
    if args.check:
        ok = check(args, report)
    if args.bench:
        bench(args, report)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
