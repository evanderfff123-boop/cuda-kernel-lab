# Matrix Transpose

FP32 square matrix-transpose kernels.

The input and output are row-major `[m, m]` matrices:

```text
output[col * m + row] = input[row * m + col]
```

## Kernels

| Kernel | Block | Shared-memory tile | Description |
|---|---:|---:|---|
| `cublas` | library managed | library managed | `cublasSgeam` reference and performance baseline |
| `naive` | 32x32 | none | direct global-memory transpose |
| `smem` | 32x32 | 32x32 | coalesced global access, with shared-memory bank conflicts |
| `padding` | 32x32 | 32x33 | adds one padding column to remove bank conflicts |
| `tiled` | 32x8 | 32x33 | each thread handles four rows, reducing the block to 256 threads |

All kernels support arbitrary positive `m`, including sizes that are not
multiples of 32.

## Run

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGPU_ARCH=75
cmake --build build --parallel
ctest --test-dir build --output-on-failure

./build/operators/transpose/transpose_benchmark --check
./build/operators/transpose/transpose_benchmark --size 4096 --check
```

## Suggested next experiments

1. Support rectangular `[rows, cols]` transpose.
2. Compare tile/block shapes such as `16x16`, `32x8`, and `32x16`.
3. Add vectorized aligned load/store variants with scalar edge paths.
4. Try diagonal block reordering for partition camping.
5. Use Nsight Compute to inspect global load/store efficiency, bank conflicts,
   occupancy, and achieved memory bandwidth.
