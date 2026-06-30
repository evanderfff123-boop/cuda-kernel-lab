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


def run_case(module, name: str, fn, m: int, n: int, k: int) -> None:
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    out = fn(a, b)
    ref = a @ b

    torch.cuda.synchronize()
    max_abs = (out - ref).abs().max().item()
    max_ref = ref.abs().max().item()
    rel = max_abs / max(max_ref, 1.0e-6)

    print(
        f"{name:>10s}  M={m:4d} N={n:4d} K={k:4d} "
        f"max_abs={max_abs:.6e} rel={rel:.6e}"
    )
    torch.testing.assert_close(out, ref, rtol=1.0e-4, atol=1.0e-4)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")

    if not (CUTLASS_ROOT / "include" / "cute" / "tensor.hpp").exists():
        raise RuntimeError(
            f"CuTe header not found under {CUTLASS_ROOT}. "
            "Expected third_party/cutlass/include/cute/tensor.hpp"
        )

    module = load_extension()

    for m, n, k in [
        (16, 16, 16),
        (128, 128, 128),
        (127, 131, 64),
        (256, 129, 33),
    ]:
        run_case(module, "tensor", module.tensor_index, m, n, k)
        run_case(module, "cta_tile", module.cta_tile, m, n, k)
        run_case(module, "k_tile", module.k_tile, m, n, k)
        run_case(module, "smem", module.smem_tile, m, n, k)
        run_case(module, "cta_style", module.cta_tiler_style, m, n, k)
        run_case(module, "smem_tensor", module.smem_tensor_style, m, n, k)

    print("PASS")


if __name__ == "__main__":
    main()
