# CUDA Kernel Lab

A personal CUDA kernel lab for implementing, validating, benchmarking, and
documenting GPU operator optimizations.

The repository is organized around independent operators. Each operator owns
its kernels, correctness tests, benchmarks, and notes, while shared utilities
can gradually move into `common/` as patterns stabilize.

## Operators

| Operator | Status | Notes |
|---|---|---|
| `transpose` | ready | FP32 square matrix transpose with naive, shared-memory, padded, tiled, and cuBLAS baseline variants |
| `gemm_cute` | learning | FP32 GEMM variants for CUDA-core optimization and CuTe source-study kernels |

## Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_ARCH=75
cmake --build build --parallel
```

The default architecture is `sm_75`, matching GTX 1650 Ti. Override
`GPU_ARCH` for other GPUs.

## Test

```bash
ctest --test-dir build --output-on-failure
```

CuTe GEMM uses PyTorch JIT extensions for fast iteration:

```bash
python3 operators/gemm_cute/tests/test_cuda_core_gemm.py
```

## Benchmark

```bash
./build/operators/transpose/transpose_benchmark --check
./build/operators/transpose/transpose_benchmark --size 4096 --check
```

For CuTe GEMM:

```bash
python3 operators/gemm_cute/benchmark/benchmark_cute_gemm.py --check
python3 operators/gemm_cute/benchmark/benchmark_cute_gemm.py --m 1024 --n 1024 --k 512 --warmup 10 --iters 100 --check
```

Or run the full build/test/benchmark flow:

```bash
./scripts/run_all.sh -DGPU_ARCH=75
```

## Layout

```text
.
├── CMakeLists.txt
├── common/
├── operators/
│   ├── transpose/
│   │   ├── include/
│   │   ├── src/
│   │   ├── tests/
│   │   └── benchmark/
│   └── gemm_cute/
│       ├── src/
│       ├── tests/
│       └── benchmark/
├── docs/
└── scripts/
```

## CuTe GEMM notes

`operators/gemm_cute` is a learning-focused area for GEMM kernels:

- `src/cuda_core_gemm.cu`: CUDA-core GEMM variants using progressively richer CuTe tensor/layout indexing.
- `src/cute_style_gemm.cu`: CuTe-style variants for CTA tiling, SMEM tensors, copy partitioning, and math partitioning.
- `src/cute_gemmv1.cu`: source-study implementation modeled after CuTe tutorial SGEMM.
- `tests/test_cuda_core_gemm.py`: correctness tests against `torch.matmul`.
- `benchmark/benchmark_cute_gemm.py`: timing and TFLOPS comparison against `torch.matmul`.

The CuTe source-study kernel contains disabled tensor-layout print helpers:

```cpp
#if 0
    if(thread0()) {
        print("  mA : "); print(mA); print("\n");
        // ...
    }
#endif
```

To inspect CuTe tensors while learning, temporarily change one block from
`#if 0` to `#if 1`, then run:

```bash
python3 operators/gemm_cute/tests/test_cuda_core_gemm.py
```

Enable only one print block at a time; otherwise the output becomes noisy.

## Development pattern

For each new operator:

1. Add `operators/<name>/`.
2. Keep kernel variants in `src/`.
3. Keep public launchers and registry in `include/`.
4. Add a correctness test and benchmark before treating the operator as done.
5. Record interesting optimization notes in that operator's docs.
