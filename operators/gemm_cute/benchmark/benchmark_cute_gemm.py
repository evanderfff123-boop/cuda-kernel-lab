#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
CUTLASS_ROOT = Path("/home/evanderfan/workspace/third_party/cutlass-v3.2.2")
BUILD_DIR = ROOT / ".torch_extensions" / "cute_gemm_v1"


def load_extension():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    return load(
        name="cute_gemm_v1",
        sources=[
            str(ROOT / "src" / "bindings.cpp"),
            str(ROOT / "src" / "cute_gemm_v1.cu"),
        ],
        build_directory=str(BUILD_DIR),
        extra_include_paths=[
            str(CUTLASS_ROOT / "include"),
            str(CUTLASS_ROOT / "tools" / "util" / "include"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-ccbin=/usr/bin/g++-10",
        ],
        verbose=False,
    )


def cuda_time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def tflops(m: int, n: int, k: int, ms: float) -> float:
    return 2.0 * m * n * k / ms / 1.0e9


def check_close(name: str,
                out: torch.Tensor,
                ref: torch.Tensor,
                rtol: float = 1.0e-4,
                atol: float = 1.0e-4) -> None:
    max_abs = (out - ref).abs().max().item()
    max_ref = ref.abs().max().item()
    rel = max_abs / max(max_ref, 1.0e-6)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    print(f"  {name:<8s} check max_abs={max_abs:.3e} rel={rel:.3e}")


def benchmark_shape(module,
                    m: int,
                    n: int,
                    k: int,
                    warmup: int,
                    iters: int,
                    do_check: bool) -> None:
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    kernels = [
        ("v1", lambda: module.forward(a, b)),
        ("v2", lambda: module.forward_v2(a, b)),
        ("v3", lambda: module.forward_v3(a, b)),
        ("v4", lambda: module.forward_v4(a, b)),
        ("torch", lambda: a @ b),
    ]

    if do_check:
        ref = a @ b
        torch.cuda.synchronize()
        check_close("v1", module.forward(a, b), ref)
        check_close("v2", module.forward_v2(a, b), ref)
        check_close("v3", module.forward_v3(a, b), ref)
        check_close("v4", module.forward_v4(a, b), ref)

    print(f"\nM={m} N={n} K={k}  warmup={warmup} iters={iters}")
    print(f"{'kernel':<10s} {'ms':>12s} {'TFLOPS':>12s}")
    print(f"{'-' * 10} {'-' * 12} {'-' * 12}")

    for name, fn in kernels:
        ms = cuda_time_ms(fn, warmup, iters)
        print(f"{name:<10s} {ms:12.6f} {tflops(m, n, k, ms):12.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark CuTe GEMM v1-v4 against torch.matmul")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    if not (CUTLASS_ROOT / "include" / "cute" / "tensor.hpp").exists():
        raise RuntimeError(
            f"CuTe header not found under {CUTLASS_ROOT}. "
            "Expected third_party/cutlass-v3.2.2/include/cute/tensor.hpp"
        )

    module = load_extension()

    if args.m is not None or args.n is not None or args.k is not None:
        if args.m is None or args.n is None or args.k is None:
            raise ValueError("--m, --n, and --k must be provided together")
        shapes = [(args.m, args.n, args.k)]
    else:
        shapes = [
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (127, 131, 64),
        ]

    device = torch.cuda.get_device_name()
    print(f"Device: {device}")

    for m, n, k in shapes:
        benchmark_shape(module,
                        m,
                        n,
                        k,
                        args.warmup,
                        args.iters,
                        args.check)


if __name__ == "__main__":
    main()
