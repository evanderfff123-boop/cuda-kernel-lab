#include "transpose/kernels.cuh"

namespace transpose {
namespace {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

__global__ void transpose_tiled_kernel(int m,
                                       float* input,
                                       float* output) {
    // The extra column removes the 32-way shared-memory bank conflict while
    // reading a tile by columns.
    __shared__ float tile[kTileDim][kTileDim + 1];

    int input_col = blockIdx.x * kTileDim + threadIdx.x;
    int input_row = blockIdx.y * kTileDim + threadIdx.y;

#pragma unroll
    for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
        if (input_col < m && input_row + offset < m) {
            tile[threadIdx.y + offset][threadIdx.x] =
                input[(input_row + offset) * m + input_col];
        }
    }

    __syncthreads();

    // Swap block coordinates. Writes are now coalesced in the transposed
    // [m, m] output matrix.
    const int output_col = blockIdx.y * kTileDim + threadIdx.x;
    const int output_row = blockIdx.x * kTileDim + threadIdx.y;

#pragma unroll
    for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
        if (output_row + offset < m && output_col < m) {
            output[(output_row + offset) * m + output_col] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

}  // namespace

void launch_tiled(int m,
                  float* input,
                  float* output,
                  cudaStream_t stream) {
    const dim3 block(kTileDim, kBlockRows);
    const dim3 grid((m + kTileDim - 1) / kTileDim,
                    (m + kTileDim - 1) / kTileDim);
    transpose_tiled_kernel<<<grid, block, 0, stream>>>(
        m, input, output);
}

}  // namespace transpose
