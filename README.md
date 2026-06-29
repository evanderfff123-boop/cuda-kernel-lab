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
| `flash_attention` | scaffold | PyTorch JIT extension for single-batch/single-head `[N, D]` FP16 forward attention |

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

## Benchmark

```bash
./build/operators/transpose/transpose_benchmark --check
./build/operators/transpose/transpose_benchmark --size 4096 --check
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
│   └── transpose/
│       ├── include/
│       ├── src/
│       ├── tests/
│       └── benchmark/
├── docs/
└── scripts/
```

## Development pattern

For each new operator:

1. Add `operators/<name>/`.
2. Keep kernel variants in `src/`.
3. Keep public launchers and registry in `include/`.
4. Add a correctness test and benchmark before treating the operator as done.
5. Record interesting optimization notes in that operator's docs.
