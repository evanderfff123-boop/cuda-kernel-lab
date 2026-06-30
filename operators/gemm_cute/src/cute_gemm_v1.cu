#include <cuda_runtime.h>

#include <cute/tensor.hpp>

namespace {

// v1 goal:
//   - Use CuTe Tensor to express row-major A/B/C indexing.
//   - Keep the CUDA execution model intentionally simple:
//       one CUDA thread computes one C[row, col].
//
// Matrix contract:
//   A: [M, K], row-major
//   B: [K, N], row-major
//   C: [M, N], row-major
__global__ void cute_gemm_v1_kernel(const float* a,
                                    const float* b,
                                    float* c,
                                    int m,
                                    int n,
                                    int k) {
    using namespace cute;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) {
        return;
    }

    // CuTe Tensor = pointer + shape + stride.
    //
    // Row-major [M, K]:
    //   physical offset for A(i, kk) = i * K + kk
    //   stride = (K, 1)
    auto A = make_tensor(make_gmem_ptr(a),
                         make_shape(m, k),
                         make_stride(k, 1));

    // Row-major [K, N]:
    //   physical offset for B(kk, j) = kk * N + j
    //   stride = (N, 1)
    auto B = make_tensor(make_gmem_ptr(b),
                         make_shape(k, n),
                         make_stride(n, 1));

    // Row-major [M, N]:
    //   physical offset for C(i, j) = i * N + j
    //   stride = (N, 1)
    auto C = make_tensor(make_gmem_ptr(c),
                         make_shape(m, n),
                         make_stride(n, 1));

    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        acc += A(row, kk) * B(kk, col);
    }
    C(row, col) = acc;
}

// v2 goal:
//   - Keep the same simple computation as v1.
//   - Introduce CuTe tiling for the output matrix C.
//   - Each CTA owns one C tile with shape [BM, BN].
//   - Each thread owns one element inside that C tile.
//
// This is the first step toward the CuTe GEMM mental model:
//   global tensor -> CTA tile -> thread-local coordinate inside tile
template <int BM, int BN>
__global__ void cute_gemm_v2_kernel(const float* a,
                                    const float* b,
                                    float* c,
                                    int m,
                                    int n,
                                    int k) {
    using namespace cute;

    // Tile-local thread coordinates.
    const int tid_m = threadIdx.y;
    const int tid_n = threadIdx.x;

    // Global coordinates corresponding to this thread's element.
    const int global_m = blockIdx.y * BM + tid_m;
    const int global_n = blockIdx.x * BN + tid_n;

    if (global_m >= m || global_n >= n) {
        return;
    }

    auto A = make_tensor(make_gmem_ptr(a),
                         make_shape(m, k),
                         make_stride(k, 1));

    auto B = make_tensor(make_gmem_ptr(b),
                         make_shape(k, n),
                         make_stride(n, 1));

    auto C = make_tensor(make_gmem_ptr(c),
                         make_shape(m, n),
                         make_stride(n, 1));

    // Take the CTA's [BM, BN] tile from the global C tensor.
    //
    // blockIdx.y selects the M tile, blockIdx.x selects the N tile.
    // c_tile has tile-local coordinates:
    //
    //   c_tile(tid_m, tid_n) == C(blockIdx.y * BM + tid_m,
    //                             blockIdx.x * BN + tid_n)
    auto c_tile = local_tile(C,
                             make_shape(Int<BM>{}, Int<BN>{}),
                             make_coord(blockIdx.y, blockIdx.x));

    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        acc += A(global_m, kk) * B(kk, global_n);
    }

    c_tile(tid_m, tid_n) = acc;
}

template<int BM, int BN, int BK>
__global__ void cute_gemm_v3_kernel(const float* a,
                                    const float* b,
                                    float* c,
                                    int m,
                                    int n,
                                    int k) {
    using namespace cute;
    const int tid_m = threadIdx.y;
    const int tid_n = threadIdx.x;

    const int global_m = blockIdx.y * BM + tid_m;
    const int global_n = blockIdx.x * BN + tid_n;

    if (global_m >= m || global_n >= n) {
        return;
    }
    
    auto A = make_tensor(make_gmem_ptr(a),
                         make_shape(m, k),
                         make_stride(k, 1));
    auto B = make_tensor(make_gmem_ptr(b),
                         make_shape(k, n),
                         make_stride(n, 1));
    auto C = make_tensor(make_gmem_ptr(c),
                         make_shape(m, n),
                         make_stride(n, 1));

    auto C_tile = local_tile(C,
                             make_shape(Int<BM>{}, Int<BN>{}),
                             make_coord(blockIdx.y, blockIdx.x));

    float acc = 0.0f;

    const int num_k_tiles = (k + BK - 1) / BK;
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        auto A_tile = local_tile(A,
                                 make_shape(Int<BM>{}, Int<BK>{}),
                                 make_coord(blockIdx.y, k_tile));

        auto B_tile = local_tile(B,
                                 make_shape(Int<BK>{}, Int<BN>{}),
                                 make_coord(k_tile, blockIdx.x));

        for (int kk = 0; kk < BK; ++kk) {
            int global_k = k_tile * BK + kk;
            if (global_k < k) {
                acc += A_tile(tid_m, kk) * B_tile(kk, tid_n);
            }
        }
    }

    C_tile(tid_m, tid_n) = acc;
}

}  // namespace

void launch_cute_gemm_v1(const float* a,
                         const float* b,
                         float* c,
                         int m,
                         int n,
                         int k,
                         cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x,
                    (m + block.y - 1) / block.y);

    cute_gemm_v1_kernel<<<grid,
                          block,
                          0,
                          stream>>>(a, b, c, m, n, k);
}

void launch_cute_gemm_v2(const float* a,
                         const float* b,
                         float* c,
                         int m,
                         int n,
                         int k,
                         cudaStream_t stream) {
    constexpr int bm = 16;
    constexpr int bn = 16;

    const dim3 block(bn, bm);
    const dim3 grid((n + bn - 1) / bn,
                    (m + bm - 1) / bm);

    cute_gemm_v2_kernel<bm, bn><<<grid,
                                  block,
                                  0,
                                  stream>>>(a, b, c, m, n, k);
}

void launch_cute_gemm_v3(const float* a,
                         const float* b,
                         float* c,
                         int m,
                         int n,
                         int k,
                         cudaStream_t stream) {
    constexpr int bm = 16;
    constexpr int bn = 16;
    constexpr int bk = 8;

    const dim3 block(bn, bm);
    const dim3 grid((n + bn - 1) / bn,
                    (m + bm - 1) / bm);

    cute_gemm_v3_kernel<bm, bn, bk><<<grid,
                                      block,
                                      0,
                                      stream>>>(a, b, c, m, n, k);
}
