// Learning stage: CuTe-style GEMM.
//
// The kernels in cuda_core_gemm.cu are still mostly "CUDA core GEMM with CuTe
// Tensor/Layout indexing". This file is intended for the next step where the
// implementation starts to use more CuTe-native concepts such as:
//
//   - CTA tiler as a first-class object
//   - logical B layout [N, K]
//   - SMEM tensors
//   - thread layout
//   - tiled copy
//   - eventually TiledMMA / Tensor Core MMA

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

namespace {

template <int BM, int BN, int BK>
__global__ void cta_tiler_gemm_kernel(const float* a,
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

template <int BM, int BN, int BK, class ASmemLayout, class BSmemLayout>
__global__ void smem_tensor_gemm_kernel(const float* a,
                                        const float* b,
                                        float* c,
                                        int m,
                                        int n,
                                        int k,
                                        ASmemLayout sA_layout,
                                        BSmemLayout sB_layout) {
    using namespace cute;
    // 共享内存在编译期必须确定大小。这个断言确保了传入的 ASmemLayout 它的 Shape 和 Stride 都是编译期常量（即使用 Int<N> 而不是普通的 int 声明），
    // 从而保证 cosize_v 可以在编译期计算出具体的数组大小。
    CUTE_STATIC_ASSERT(is_static<ASmemLayout>::value);
    CUTE_STATIC_ASSERT(is_static<BSmemLayout>::value);

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

    // cosize_v是Cute提供的一个元函数，用于计算一个Layout在物理内存中所占用的最大空间
    __shared__ float smem_a[cosize_v<ASmemLayout>];
    __shared__ float smem_b[cosize_v<BSmemLayout>];
    // 将指针与布局绑定，形成tensor
    auto sA = make_tensor(make_smem_ptr(smem_a),
                          sA_layout);
    auto sB = make_tensor(make_smem_ptr(smem_b),
                          sB_layout);
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

void launch_smem_tensor_style_gemm(const float* a,
                                   const float* b,
                                   float* c,
                                   int m,
                                   int n,
                                   int k,
                                   cudaStream_t stream) {
    using namespace cute;

    constexpr int bm = 16;
    constexpr int bn = 16;
    constexpr int bk = 8;

    auto sA_layout = make_layout(make_shape(Int<bm>{}, Int<bk>{}));
    auto sB_layout = make_layout(make_shape(Int<bn>{}, Int<bk>{}));

    const dim3 block(bn, bm);
    const dim3 grid((n + bn - 1) / bn,
                    (m + bm - 1) / bm);

    smem_tensor_gemm_kernel<bm, bn, bk><<<grid,
                                         block,
                                         0,
                                         stream>>>(a, b, c, m, n, k,
                                                   sA_layout,
                                                   sB_layout);
}
