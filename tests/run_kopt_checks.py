# SPDX-License-Identifier: Apache-2.0
"""GPU checks and micro-benchmarks for the Hadacore rotation, the planar PTQ1_0
mat-vec and the fused rotate + quantize mat-vec, without pytest.

Run in the engine venv with this checkout first on the path (not with -I, which
would drop PYTHONPATH):

    PYTHONPATH=$HOME/k_plugin /home/ubuntu/vllm-env/bin/python \\
        $HOME/k_plugin/tests/run_kopt_checks.py --hadamard --planar --fused --bench --out B.json

--hadamard  Hadacore vs the dense ``x @ H`` rotation (and both vs an fp64 reference) on
            N(0,1) activations of every folded width (5120 / 6144 / 17408) at 1-2048
            rows, bf16 and fp16; the embedding inverse; rotate + PTQ1_0 mat-vec end to
            end; CUDA-graph capture and torch.compile(fullgraph) of the layout.
--planar    planar PTQ1_0 == GGML-order PTQ1_0, bit for bit: mat-vec at 1-16 columns and
            every lanes-per-row variant, Triton GEMM 9-256 rows, dequantize, embedding
            lookup, the dispatcher's tiers; repack/unpack round trip.
--fused     the fused sign + FWHT + Q8 quantize + mat-vec against rotate-then-mat-vec.
--bench     the 27B shapes with weights streaming (copies cycled past the L2), CUDA-graph
            replay: us per call and weight GB/s, GGML vs planar (lanes per row 0/8/16/32)
            at 1/2/4/8 columns; rotation us per call dense vs Hadacore (vs fused).
Exit 0 = every check passed; 1 = a check failed; 2 = refused (missing kernel/op).
"""

import argparse
import json
import os
import sys
import time

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType

import vllm_gguf_plugin
import vllm_gguf_plugin.ops as ops
from vllm_gguf_plugin.quantization import hadamard
from vllm_gguf_plugin.quantization import linear
from vllm_gguf_plugin.quantization import utils as qutils
from vllm_gguf_plugin.quantization import vocal_embeds

PTQ = int(GGMLQuantizationType.PTQ1_0)
PLANAR = qutils.PTQ1_0_PLANAR
SHAPES = {  # name: (output rows, K)
    "gate_up": (34816, 5120),
    "qkvz": (16384, 5120),
    "attn_qkv": (14336, 5120),
    "out/o_proj": (5120, 6144),
    "down": (5120, 17408),
    "lm_head": (248320, 5120),
}
FOLDED_WIDTHS = (5120, 6144, 17408)
# planar mat-vec variants: 0 = auto, lanes per row 8 / 16 / 32, +1000 = prefetching loop,
# +2000 = L1-bypassing weight loads
LPR_VARIANTS = (0, 8, 16, 32, 1008, 1016, 1032, 2008, 2016, 2032, 3016, 3032)
FAILURES = []


def expect(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(' ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def ternary(rows, k, seed=0):
    rng = np.random.default_rng(seed)
    trits = rng.integers(-1, 2, size=(rows, k // 128, 128)).astype(np.float32)
    trits[..., 0] = 1
    scales = rng.uniform(0.25, 1.5, size=(rows, k // 128, 1)).astype(np.float16).astype(np.float32)
    return (trits * scales).reshape(rows, k)


def packed_ptq(rows, k, seed=0):
    return torch.tensor(gguf.quantize(ternary(rows, k, seed), GGMLQuantizationType.PTQ1_0), device="cuda")


def planar_copy(w):
    return linear.repack_ptq1_0_planar(w.clone())


def random_signs(width, seed):
    rng = np.random.default_rng(seed)
    return np.where(rng.integers(0, 2, size=width) == 1, 1.0, -1.0).astype(np.float32)


def layouts(width, seed, forward=True, inverse=False):
    """(hadacore layout, dense layout) over the same signs."""
    signs = random_signs(width, seed)
    # the exact device of the inputs (cuda:N): _apply re-prepares -- and so
    # re-resolves Hadacore -- on any device mismatch, "cuda" vs "cuda:0" included
    device = torch.device("cuda", torch.cuda.current_device())
    hc = hadamard.GGUFHadamardLayout(1024, width, signs, forward=forward, inverse=inverse)
    hc.prepare(device)
    dense = hadamard.GGUFHadamardLayout(1024, width, signs, forward=forward, inverse=inverse)
    dense.prepare(device)
    dense._hadacore = None
    return hc, dense, torch.tensor(signs, dtype=torch.float64, device="cuda")


def rot_ref(x, signs, inverse=False):
    h = hadamard.hadamard_matrix(1024, "cuda", torch.float64)
    xd = x.double()
    if not inverse:
        xd = xd * signs
    out = (xd.reshape(-1, 1024) @ h).reshape(x.shape)
    if inverse:
        out = out * signs
    return out


# --- K1: Hadacore rotation ------------------------------------------------------


def check_hadamard():
    if hadamard.resolve_hadacore(1024) is None:
        print("REFUSED: vLLM's hadacore_transform is not available in this venv")
        return 2
    torch.manual_seed(0)
    worst = {}
    for dtype in (torch.bfloat16, torch.float16):
        for width in FOLDED_WIDTHS:
            hc, dense, signs = layouts(width, seed=width)
            assert hc._hadacore is not None and dense._hadacore is None
            for m in (1, 2, 4, 8, 64, 2048):
                x = torch.randn((m, width), dtype=dtype, device="cuda")
                a = hc.input_to_gguf(x).float()
                b = dense.input_to_gguf(x).float()
                ref = rot_ref(x, signs)
                max_rel = ((a - b).abs().max() / b.abs().max()).item()
                mean_rel = ((a - b).abs().mean() / b.pow(2).mean().sqrt()).item()
                err_hc = ((a.double() - ref).norm() / ref.norm()).item()
                err_dense = ((b.double() - ref).norm() / ref.norm()).item()
                worst[str(dtype)] = max(worst.get(str(dtype), 0.0), max_rel)
                expect(
                    f"rotation {dtype} width={width} m={m}",
                    max_rel <= 2**-6 and mean_rel <= 4e-3,
                    f"max|hc-dense|/max|dense|={max_rel:.2e} mean/rms={mean_rel:.2e} "
                    f"relL2 vs fp64: hadacore {err_hc:.2e} dense {err_dense:.2e}",
                )
    # embedding inverse: s * (H z)
    hc, dense, signs = layouts(5120, seed=7, forward=False, inverse=True)
    z = torch.randn((3, 5120), dtype=torch.bfloat16, device="cuda")
    a, b = hc.output_to_vllm(z).float(), dense.output_to_vllm(z).float()
    ref = rot_ref(z, signs, inverse=True)
    max_rel = ((a - b).abs().max() / b.abs().max()).item()
    expect("embedding inverse width=5120", max_rel <= 2**-6,
           f"max rel {max_rel:.2e} relL2 vs fp64 hadacore {((a.double() - ref).norm() / ref.norm()).item():.2e}")
    # the input tensor is never modified (out-of-place transform)
    x = torch.randn((2, 5120), dtype=torch.bfloat16, device="cuda")
    keep = x.clone()
    nosign = hadamard.GGUFHadamardLayout(1024, 5120, None, forward=True, inverse=False)
    nosign.prepare(torch.device("cuda", torch.cuda.current_device()))
    nosign.input_to_gguf(x)
    expect("input untouched (identity signs)", torch.equal(x, keep))
    # rotate + PTQ1_0 mat-vec end to end, vs the dense-rotation path and fp64
    for width in FOLDED_WIDTHS:
        hc, dense, signs = layouts(width, seed=width + 1)
        w = ternary(256, width, seed=width)
        wq = torch.tensor(gguf.quantize(w, GGMLQuantizationType.PTQ1_0), device="cuda")
        wd = torch.tensor(w, dtype=torch.float64, device="cuda")
        for m in (1, 4, 8):
            x = torch.randn((m, width), dtype=torch.bfloat16, device="cuda")
            ya = linear._fused_mul_mat_gguf(hc.input_to_gguf(x), wq, PTQ).double()
            yb = linear._fused_mul_mat_gguf(dense.input_to_gguf(x), wq, PTQ).double()
            yr = rot_ref(x, signs) @ wd.T
            rel = ((ya - yb).norm() / yb.norm()).item()
            expect(f"folded linear width={width} m={m}", rel <= 5e-3,
                   f"relL2 hc vs dense {rel:.2e}; vs fp64: hc {((ya - yr).norm() / yr.norm()).item():.2e} "
                   f"dense {((yb - yr).norm() / yr.norm()).item():.2e}")
    # CUDA graph capture and torch.compile(fullgraph=True)
    hc, _, _ = layouts(17408, seed=3)
    x = torch.randn((4, 17408), dtype=torch.bfloat16, device="cuda")
    eager = hc.input_to_gguf(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        hc.input_to_gguf(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = hc.input_to_gguf(x)
    graph.replay()
    torch.cuda.synchronize()
    expect("CUDA graph replay == eager", torch.equal(captured, eager))
    compiled = torch.compile(lambda t: hc.input_to_gguf(t) * 1.0, fullgraph=True, dynamic=True)
    out = compiled(x)
    expect("torch.compile(fullgraph) == eager", torch.equal(out, eager))
    print(f"hadamard worst max-rel per dtype: {worst}")
    return 0


# --- K2: planar PTQ1_0 ---------------------------------------------------------------


def check_planar():
    if not ops.ptq1_0_planar_available():
        print("REFUSED: this plugin build has no planar PTQ1_0 kernels (wrong PYTHONPATH or stale .so)")
        return 2
    torch.manual_seed(0)
    for k in (512, 1024, 5120, 6144, 17408):
        for rows in (3, 37, 515):
            w = packed_ptq(rows, k, seed=rows + k)
            wp = planar_copy(w)
            expect(f"round trip k={k} rows={rows}", torch.equal(linear.unpack_ptq1_0_planar(wp), w))
            for n in range(1, 17):
                for dt in (torch.bfloat16, torch.float16, torch.float32):
                    if dt != torch.bfloat16 and n not in (1, 3, 8, 16):
                        continue
                    x = torch.randn((n, k), dtype=dt, device="cuda")
                    ref = ops.ggml_mul_mat_vec_ptq1_0(w, x, rows)
                    for lpr in LPR_VARIANTS:
                        out = ops.ggml_mul_mat_vec_ptq1_0_planar(wp, x, rows, lpr)
                        ok = torch.equal(out, ref)
                        if not ok or (n in (1, 8) and dt == torch.bfloat16 and rows == 515):
                            diff = (out.float() - ref.float()).abs().max().item()
                            expect(f"mat-vec k={k} rows={rows} n={n} {dt} lpr={lpr}", ok, f"max|diff|={diff:.3g}")
            for dt in (torch.bfloat16, torch.float32):
                expect(f"dequant k={k} rows={rows} {dt}", torch.equal(
                    ops.ggml_dequantize(wp, PLANAR, rows, k, dt), ops.ggml_dequantize(w, PTQ, rows, k, dt)))
        # the Triton GEMM and the dispatcher on one shape per K
        w = packed_ptq(300, k, seed=k)
        wp = planar_copy(w)
        for n in (9, 16, 33, 256):
            x = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
            expect(f"triton gemm k={k} n={n}", torch.equal(
                ops.ggml_mul_mat_a8(wp, x, PLANAR, 300), ops.ggml_mul_mat_a8(w, x, PTQ, 300)))
        for n in (1, 8, 9, 300):
            x = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
            expect(f"dispatch k={k} n={n}", torch.equal(
                linear._fused_mul_mat_gguf(x, wp, PLANAR), linear._fused_mul_mat_gguf(x, w, PTQ)))
        ids = torch.tensor([[0, 299], [150, 3]], device="cuda")
        expect(f"embedding k={k}", torch.equal(
            vocal_embeds._apply_gguf_embedding(ids, wp, PLANAR, k, dtype=torch.bfloat16),
            vocal_embeds._apply_gguf_embedding(ids, w, PTQ, k, dtype=torch.bfloat16)))
    # CUDA graph capture of the planar op
    w = planar_copy(packed_ptq(1000, 5120, seed=1))
    x = torch.randn((2, 5120), dtype=torch.bfloat16, device="cuda")
    eager = ops.ggml_mul_mat_vec_ptq1_0_planar(w, x, 1000)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ops.ggml_mul_mat_vec_ptq1_0_planar(w, x, 1000)
    graph.replay()
    torch.cuda.synchronize()
    expect("planar mat-vec CUDA graph replay == eager", torch.equal(captured, eager))
    # refusals: no silent zeros from the generic kernels
    for name, fn in (("generic mmvq", lambda: torch.ops._C_gguf.ggml_mul_mat_vec_a8(w, x, PLANAR, 1000)),
                     ("generic dequant", lambda: torch.ops._C_gguf.ggml_dequantize(w, PLANAR, 1000, 5120, None))):
        try:
            fn()
            torch.cuda.synchronize()
            expect(f"{name} refuses the planar id", False)
        except RuntimeError:
            expect(f"{name} refuses the planar id", True)
    return 0


# --- K1b: fused rotate + quantize ------------------------------------------------------


def check_fused():
    if not hasattr(ops, "fused_rot_available") or not ops.fused_rot_available():
        print("REFUSED: this plugin build has no fused rotate+quantize mat-vec")
        return 2
    torch.manual_seed(0)
    for width in FOLDED_WIDTHS:
        hc, dense, signs = layouts(width, seed=width + 2)
        sg = hc._signs[torch.float32]
        w = packed_ptq(515, width, seed=width)
        wd = torch.tensor(ternary(515, width, seed=width), dtype=torch.float64, device="cuda")
        for planar in (False, True):
            wq, wt = (planar_copy(w), PLANAR) if planar else (w, PTQ)
            for n in (1, 2, 3, 4, 8):
                x = torch.randn((n, width), dtype=torch.bfloat16, device="cuda")
                fused = ops.ggml_mul_mat_vec_ptq1_0_rot(wq, x, sg, 515, planar).double()
                yd = linear._fused_mul_mat_gguf(dense.input_to_gguf(x), wq, wt).double()
                yh = linear._fused_mul_mat_gguf(hc.input_to_gguf(x), wq, wt).double()
                yr = rot_ref(x, signs) @ wd.T
                rel = ((fused - yd).norm() / yd.norm()).item()
                expect(f"fused width={width} planar={planar} n={n}", rel <= 5e-3,
                       f"relL2 fused vs dense-rotation path {rel:.2e}; vs fp64: fused "
                       f"{((fused - yr).norm() / yr.norm()).item():.2e} dense {((yd - yr).norm() / yr.norm()).item():.2e} "
                       f"hadacore {((yh - yr).norm() / yr.norm()).item():.2e}")
        # the op as the linear method calls it: fused at <= 8 rows, dense rotation above
        for n in (1, 8, 9, 64):
            x = torch.randn((n, width), dtype=torch.bfloat16, device="cuda")
            op = linear._fused_rot_mul_mat_gguf(x, w, PTQ, sg, dense._signs[torch.bfloat16],
                                                dense._matrices[torch.bfloat16])
            yd = linear._fused_mul_mat_gguf(dense.input_to_gguf(x), w, PTQ)
            expect(f"fused op width={width} n={n} == dense-rotation path", torch.equal(op, yd),
                   f"max|diff|={(op.float() - yd.float()).abs().max().item():.3g}")
    return 0


# --- benches -------------------------------------------------------------------


def _time_graph(fn, reps, warm):
    for _ in range(warm):
        fn(0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(reps):
            fn(i)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000.0 / (5 * reps)
    del graph
    return us


def bench_matvec(results, cols=(1, 2, 4, 8), reps=20, l2_bytes=200 << 20, shapes=None):
    for name, (rows, k) in SHAPES.items():
        if shapes and name not in shapes:
            continue
        base_rows = min(rows, 4096)
        base = packed_ptq(base_rows, k, seed=1)
        first = base.repeat(-(-rows // base_rows), 1)[:rows].contiguous()
        del base
        nbytes = first.numel()
        ncopies = max(1, -(-l2_bytes // nbytes))
        variants = [("ggml", None)] + [(f"planar{l}", l) for l in LPR_VARIANTS]
        for vname, lpr in variants:
            if lpr is None:
                copies = [first] + [first.clone() for _ in range(ncopies - 1)]
            else:
                p = planar_copy(first)
                copies = [p] + [p.clone() for _ in range(ncopies - 1)]
            for n in cols:
                x = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
                if lpr is None:
                    fn = lambda i: ops.ggml_mul_mat_vec_ptq1_0(copies[i % len(copies)], x, rows)  # noqa: E731
                else:
                    fn = lambda i: ops.ggml_mul_mat_vec_ptq1_0_planar(copies[i % len(copies)], x, rows, lpr)  # noqa: E731
                us = _time_graph(fn, reps, len(copies))
                results.append({"bench": "matvec", "shape": name, "rows": rows, "k": k, "variant": vname,
                                "cols": n, "us": us, "weight_bytes": nbytes, "gb_per_s": nbytes / us / 1e3,
                                "copies": len(copies)})
                print(f"{name:10s} {vname:8s} n={n} {us:8.1f} us {nbytes / us / 1e3:7.1f} GB/s", flush=True)
            del copies
            torch.cuda.empty_cache()
        del first
        torch.cuda.empty_cache()


def bench_rotation(results, reps=200):
    for width in FOLDED_WIDTHS:
        hc, dense, _ = layouts(width, seed=width)
        for m in (1, 2, 4, 8):
            x = torch.randn((m, width), dtype=torch.bfloat16, device="cuda")
            for vname, lay in (("dense", dense), ("hadacore", hc)):
                us = _time_graph(lambda i, lay=lay: lay.input_to_gguf(x), reps, 3)
                results.append({"bench": "rotation", "width": width, "m": m, "variant": vname, "us": us})
                print(f"rotation width={width} m={m} {vname:8s} {us:6.2f} us (incl. sign multiply)", flush=True)


def bench_small_m(results, reps=200):
    """BF16 in_proj_ba-shaped GEMMs ([M, 5120] x [5120, 96]) at 1-8 rows, call variants."""
    w = torch.randn((96, 5120), dtype=torch.bfloat16, device="cuda") / 64
    for m in (1, 2, 3, 4, 8):
        x = torch.randn((m, 5120), dtype=torch.bfloat16, device="cuda")
        ref = (x.float() @ w.float().T)
        variants = {
            "x@w.T": lambda: x @ w.T,
            "plugin small_m_gemv": (lambda: ops.small_m_gemv(w, x)) if m >= 2 else (lambda: x @ w.T),
            "(w@x.T).T.contig": lambda: (w @ x.T).T.contiguous(),
            "per-row mv": lambda: torch.stack([torch.mv(w, x[i]) for i in range(m)]),
            "bmm-rows": lambda: torch.bmm(w.unsqueeze(0).expand(m, -1, -1), x.unsqueeze(-1)).squeeze(-1),
        }
        for name, fn in variants.items():
            out = fn()
            err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
            if name == "plugin small_m_gemv" and m >= 2:
                expect(f"small_m_gemv m={m} vs fp32", err <= 1e-2, f"max rel err {err:.2e}")
            us = _time_graph(lambda i, fn=fn: fn(), reps, 3)
            results.append({"bench": "small_m", "m": m, "variant": name, "us": us, "max_rel_err": err})
            print(f"in_proj_ba m={m} {name:18s} {us:7.2f} us max rel err {err:.2e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hadamard", action="store_true")
    parser.add_argument("--planar", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--bench-shapes", nargs="*", help="subset of the bench shapes")
    parser.add_argument("--bench-small-m", action="store_true", help="in_proj_ba-shaped BF16 GEMM variants")
    parser.add_argument("--out", help="write bench results JSON here")
    args = parser.parse_args()
    print(f"run_kopt_checks plugin={vllm_gguf_plugin.__file__} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name()} utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
          f"env VLLM_GGUF_HADACORE={os.environ.get('VLLM_GGUF_HADACORE', '(unset)')} "
          f"VLLM_GGUF_PTQ1_0_PLANAR={os.environ.get('VLLM_GGUF_PTQ1_0_PLANAR', '(unset)')}", flush=True)
    if not (args.hadamard or args.planar or args.fused or args.bench or args.bench_small_m):
        parser.error("pass --hadamard, --planar, --fused, --bench and/or --bench-small-m")
    if args.out and not (args.bench or args.bench_small_m):
        parser.error("--out needs --bench or --bench-small-m")
    for flag, fn in ((args.hadamard, check_hadamard), (args.planar, check_planar), (args.fused, check_fused)):
        if flag and fn() == 2:
            return 2
    if args.bench or args.bench_small_m:
        results = []
        if args.bench_small_m:
            bench_small_m(results)
        if args.bench:
            bench_rotation(results)
            bench_matvec(results, shapes=args.bench_shapes)
        if args.out:
            with open(args.out, "w") as handle:
                json.dump(results, handle, indent=1)
            print(f"bench results: {args.out}")
    print(f"failures: {len(FAILURES)}{(' ' + ', '.join(FAILURES[:20])) if FAILURES else ''}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
