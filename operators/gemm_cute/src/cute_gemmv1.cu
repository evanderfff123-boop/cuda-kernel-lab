#include <cute/tensor.hpp>

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
            Alpha alpha, Beta beta)
{
    using namespace cute;
    using X = Underscore;

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);  // (m, n, k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    Tensor tAgA = local_partition(gA, tA, threadIdx.x);
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);

    Tensor tBgB = local_partition(gB, tB, threadIdx.x);
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);

    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});
    Tensor tCrC = make_tensor_like(tCgC);

    clear(tCrC);

    auto K_TILE_MAX = size<2>(tAgA);
    for(int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        copy(tAgA(_,_,k_tile), tAsA);
        copy(tBgB(_,_,k_tile), tBsB);

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        gemm(tCsA, tCsB, tCrC);
        __syncthreads();
    }

    axpby(alpha, tCrC, beta, tCgC);
}

void launch_cute_gemmv1(const float* a,
                        const float* b,
                        float* c,
                        int m,
                        int n,
                        int k,
                        cudaStream_t stream) {
    using namespace cute;

    auto shape_mnk = make_shape(m, n, k);

    auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<8>{});

    auto a_stride = make_stride(k, Int<1>{});
    auto b_stride = make_stride(Int<1>{}, n);
    auto c_stride = make_stride(n, Int<1>{});

    auto sA_layout = make_layout(make_shape(Int<128>{}, Int<8>{}));
    auto sB_layout = make_layout(make_shape(Int<128>{}, Int<8>{}));
    auto sC_layout = make_layout(make_shape(Int<128>{}, Int<128>{}));

    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
    auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

    const dim3 block(size(tC));
    const dim3 grid((n + 127) / 128,
                    (m + 127) / 128);

    gemm_device<<<grid, block, 0, stream>>>(
        shape_mnk,
        cta_tiler,
        a, a_stride, sA_layout, tA,
        b, b_stride, sB_layout, tB,
        c, c_stride, sC_layout, tC,
        1.0f, 0.0f);
}
