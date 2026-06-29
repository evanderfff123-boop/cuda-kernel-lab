#include "transpose/kernels.cuh"
#define INDX(row, col, ld)  (((row) * (ld)) + (col))

namespace transpose {
namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 32;

__global__ void transpose_naive_kernel(int m, float *a, float *c)
{
    int myCol = blockIdx.x * blockDim.x + threadIdx.x;
    int myRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (myRow < m && myCol < m) {
        c[INDX(myCol, myRow, m)] = a[INDX(myRow, myCol, m)];
    }
    return;
}

}  // namespace

void launch_naive(int m, float *a, float *c, cudaStream_t stream) {
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((m + block.x - 1) / block.x,
                    (m + block.y - 1) / block.y);
    transpose_naive_kernel<<<grid, block, 0, stream>>>(m, a, c);
}

}  // namespace transpose
