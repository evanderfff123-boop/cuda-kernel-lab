#!/usr/bin/env python3
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
CUTLASS_ROOT = Path("/home/evanderfan/workspace/third_party/cutlass-v3.2.2")
BUILD_DIR = ROOT / ".torch_extensions" / "gemm_cute"


def load_extension():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    return load(
        name="gemm_cute_cuda_core",
        sources=[
            str(ROOT / "src" / "bindings.cpp"),
            str(ROOT / "src" / "cuda_core_gemm.cu"),
            str(ROOT / "src" / "cute_style_gemm.cu"),
            str(ROOT / "src" / "cute_gemmv1.cu"),
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
        verbose=True,
    )


def run_case(fn, m: int, n: int, k: int) -> tuple[float, float]:
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    out = fn(a, b)
    ref = a @ b

    torch.cuda.synchronize()
    max_abs = (out - ref).abs().max().item()
    max_ref = ref.abs().max().item()
    rel = max_abs / max(max_ref, 1.0e-6)

    torch.testing.assert_close(out, ref, rtol=1.0e-4, atol=1.0e-4)
    return max_abs, rel


def run_group(title: str, module, kernels, shapes) -> None:
    print(f"\n{title}")

    for m, n, k in shapes:
        print(f"\nM={m} N={n} K={k}")
        print(f"{'kernel':<12s} {'max_abs':>12s} {'rel':>12s}")
        print(f"{'-' * 12} {'-' * 12} {'-' * 12}")

        for name, fn in kernels:
            max_abs, rel = run_case(fn, m, n, k)
            print(f"{name:<12s} {max_abs:12.3e} {rel:12.3e}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")

    if not (CUTLASS_ROOT / "include" / "cute" / "tensor.hpp").exists():
        raise RuntimeError(
            f"CuTe header not found under {CUTLASS_ROOT}. "
            "Expected third_party/cutlass/include/cute/tensor.hpp"
        )

    module = load_extension()

    cuda_core_shapes = [
        (16, 16, 16),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 512),
        (127, 131, 64),
        (256, 129, 33),
        (511, 509, 257),
    ]
    cuda_core_kernels = [
        ("tensor", module.tensor_index),
        ("cta_tile", module.cta_tile),
        ("k_tile", module.k_tile),
        ("smem", module.smem_tile),
        ("cta_style", module.cta_tiler_style),
        ("smem_tensor", module.smem_tensor_style),
        ("copy_part", module.copy_partition_style),
        ("math_part", module.math_partition_style),
    ]
    run_group("CUDA-core / CuTe-style kernels", module, cuda_core_kernels, cuda_core_shapes)

    cute_source_shapes = [
        (128, 128, 128),
        (256, 256, 128),
        (512, 512, 256),
        (1024, 1024, 512),
    ]
    cute_source_kernels = [
        ("cute_v1", module.cute_gemmv1),
    ]
    run_group("CuTe source-study kernels", module, cute_source_kernels, cute_source_shapes)

    print("\nPASS")


if __name__ == "__main__":
    main()
