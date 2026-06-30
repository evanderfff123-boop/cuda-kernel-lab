// Reserved for the next learning stage: CuTe-style GEMM.
//
// The kernels in cuda_core_gemm.cu are still mostly "CUDA core GEMM with CuTe
// Tensor/Layout indexing". This file is intended for the next step where the
// implementation starts to use more CuTe-native concepts such as:
//
//   - CTA tiler as a first-class object
//   - logical B layout [N, K]
//   - thread layout
//   - tiled copy
//   - eventually TiledMMA / Tensor Core MMA
//
// Keep this file empty until the first CuTe-style kernel is introduced.

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

namespace {

template <int BM, int BN, int BK>
__global__ void cta_tiler_gemm_kernel(const float* a,
                                 const float* b,
                                 float* c,
                                 int m,
                                 int n,
                                 int k){
    using namespace cute;

    const int tid_m = threadIdx.y;
    const int tid_n = threadIdx.x;

    const int global_m = blockIdx.y * BM + tid_m;
    const int global_n = blockIdx.x * BN + tid_n;
    
    auto A = make_tensor(make_gmem_ptr(a),
                         make_shape(m, k),
                         make_stride(k, Int<1>{}));
    auto B = make_tensor(make_gmem_ptr(b),
                         make_shape(n, k),
                         make_stride(Int<1>{}, n));
    auto C = make_tensor(make_gmem_ptr(c),
                         make_shape(m, n),
                         make_stride(n, Int<1>{}));
    auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});

    auto C_tile = local_tile(C,
                             select<0, 1>(cta_tiler),
                             make_coord(blockIdx.y, blockIdx.x));
    __shared__ float smem_a[BM * BK];
    __shared__ float smem_b[BN * BK];
    auto sA = make_tensor(make_smem_ptr(smem_a),
                          make_shape(Int<BM>{}, Int<BK>{}));
    auto sB = make_tensor(make_smem_ptr(smem_b),
                          make_shape(Int<BN>{}, Int<BK>{}));
    float acc = 0.0f;
    int num_k_tiles = (k + BK - 1) / BK;
    for(int k_tile = 0; k_tile < num_k_tiles; ++k_tile){
        auto A_tile = local_tile(A,
                                 select<0, 2>(cta_tiler),
                                 make_coord(blockIdx.y, k_tile));
        auto B_tile = local_tile(B,
                                 select<1, 2>(cta_tiler),
                                 make_coord(blockIdx.x, k_tile));
        if(tid_n < BK){
            int global_k = k_tile * BK + tid_n;
            sA(tid_m, tid_n) = (global_m < m && global_k < k)
                ? A_tile(tid_m, tid_n)
                : 0.0f;
        }

        if(tid_m < BK){
            int global_k = k_tile * BK + tid_m;
            sB(tid_n, tid_m) = (global_n < n && global_k < k)
                ? B_tile(tid_n, tid_m)
                : 0.0f;
        }

        __syncthreads();

        for(int kk = 0; kk < BK; ++kk){
            acc += sA(tid_m, kk) * sB(tid_n, kk);
        }

        __syncthreads();
    }
    if(global_m < m && global_n < n){
        C_tile(tid_m, tid_n) = acc;
    }
}

}  // namespace

void launch_cta_tiler_style_gemm(const float* a,
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

    cta_tiler_gemm_kernel<bm, bn, bk><<<grid,
                                        block,
                                        0,
                                        stream>>>(a, b, c, m, n, k);
}
