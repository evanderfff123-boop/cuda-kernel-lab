#include "transpose/kernels.cuh"
#define INDX(row, col, ld) (((row) * (ld)) + (col))

namespace transpose {
namespace {

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

__global__ void transpose_smem_kernel(int m, float* a, float* c) {
    __shared__ float smemArray[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X];

    const int myCol = blockDim.x * blockIdx.x + threadIdx.x;
    const int myRow = blockDim.y * blockIdx.y + threadIdx.y;

    const int tileX = blockDim.x * blockIdx.x;
    const int tileY = blockDim.y * blockIdx.y;

    if(myRow < m && myCol < m)
    {
        // 读取a的时候访存合并
        smemArray[threadIdx.y][threadIdx.x] = a[INDX(tileY + threadIdx.y, tileX + threadIdx.x, m)];
    }
    __syncthreads();  
    const int outputRow = tileX + threadIdx.y;
    const int outputCol = tileY + threadIdx.x;
    if(outputRow < m && outputCol < m)
    {
        // 写c的时候访存合并
        c[INDX(outputRow, outputCol, m)] = smemArray[threadIdx.x][threadIdx.y];
    }
    return;
}
}  // namespace

void launch_smem(int m, float* a, float* b, cudaStream_t stream) {
    const dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    const dim3 grid((m + block.x - 1) / block.x,
                    (m + block.y - 1) / block.y);
    transpose_smem_kernel<<<grid, block, 0, stream>>>(m, a, b);
}

}  // namespace transpose
