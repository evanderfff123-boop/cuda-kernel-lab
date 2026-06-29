# FlashAttention

This operator starts with the smallest useful problem:

```text
B = 1, H = 1
Q, K, V: [N, D] fp32 CUDA tensors
O:       [N, D] fp32 CUDA tensor
```

The current first implementation targets the user's v1 tiled contract:

```text
Q, K, V, O: torch.float32 CUDA tensors with shape [N, 64]
N:          divisible by 64
Br:         64
Bc:         64
```

It computes standard attention without materializing the full `[N, N]` matrix
in global memory:

```text
O = softmax(Q @ K^T / sqrt(D)) @ V
```

For `v1`, each CUDA block owns one query tile of 64 rows, and each thread owns
one query row inside that tile.

## Python JIT test

From this directory:

```bash
python3 tests/test_fa_v1.py
```

The test uses `torch.utils.cpp_extension.load` to compile:

```text
src/bindings.cpp
src/fa_v1.cu
```

Default test shape:

```text
N = 512
D = 64
dtype = torch.float32
```

## Current files

```text
operators/flash_attention/
├── README.md
├── src/
│   ├── bindings.cpp
│   └── fa_v1.cu
└── tests/
    └── test_fa_v1.py
```

## Next steps

1. Keep this naive CUDA implementation as a correctness baseline.
2. Add benchmark scripts comparing PyTorch reference and this CUDA v1 kernel.
3. Improve memory access and reduce repeated global/shared-memory work.
4. Only after the `[N, D]` version is stable, extend the API to `[B, H, N, D]`.
