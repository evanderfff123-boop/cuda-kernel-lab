#!/usr/bin/env python3
import math
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / ".torch_extensions" / "fa_v1"


def load_extension():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    return load(
        name="flash_attention_v1",
        sources=[
            str(ROOT / "src" / "bindings.cpp"),
            str(ROOT / "src" / "fa_v1.cu"),
        ],
        build_directory=str(BUILD_DIR),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )


def reference_attention(q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor) -> torch.Tensor:
    d = q.shape[-1]
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(d)
    probs = torch.softmax(scores, dim=-1)
    return probs @ v.float()


def run_case(module, n: int, d: int) -> None:
    torch.manual_seed(0)
    q = torch.randn((n, d), device="cuda", dtype=torch.float32)
    k = torch.randn((n, d), device="cuda", dtype=torch.float32)
    v = torch.randn((n, d), device="cuda", dtype=torch.float32)

    out = module.forward(q, k, v)
    ref = reference_attention(q, k, v)

    torch.cuda.synchronize()
    max_abs = (out.float() - ref.float()).abs().max().item()
    max_ref = ref.float().abs().max().item()
    rel = max_abs / max(max_ref, 1.0e-6)

    print(f"N={n:4d} D={d:3d} max_abs={max_abs:.6e} rel={rel:.6e}")
    torch.testing.assert_close(
        out,
        ref,
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")

    module = load_extension()

    # The current v1 interface is the user's tiled kernel contract:
    # float32 CUDA tensors with shape [N, 64], N divisible by 64.
    for n, d in [
        (64, 64),
        (128, 64),
        (512, 64),
    ]:
        run_case(module, n, d)

    print("PASS")


if __name__ == "__main__":
    main()
