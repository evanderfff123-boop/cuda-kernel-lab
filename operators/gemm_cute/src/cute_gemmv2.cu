#include <cute/tensor.hpp>

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta)
{
    using namespace cute;
    using X = Underscore;

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    auto thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tAsA = thr_copy_a.partition_D(sA);
    Tensor tArA = make_fragment_like(tAsA);

    auto thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);
    Tensor tBrB = make_fragment_like(tBsB);

    copy(copy_a, tAgA(_,_,_,0), tArA);
    copy(copy_b, tBgB(_,_,_,0), tBrB);

    auto thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

    clear(tCrC);

    auto K_TILE_MAX = size<3>(tAgA);

    for(int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        __syncthreads();
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads();

        int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
        copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);

        gemm(mma, tCsA, tCsB, tCrC);
    }
    axpby(alpha, tCrC, beta, tCgC);
}

void launch_cute_gemmv2(const float* a,
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

    auto copyA = make_tiled_copy(Copy_Atom<UniversalCopy<float>, float>{},
                                 Layout<Shape<_32,_8>,Stride<_8, _1>>{},
                                 Layout<Shape<_1,_1>>{});
    auto copyB = make_tiled_copy(Copy_Atom<UniversalCopy<float>, float>{},
                                 Layout<Shape<_32,_8>,Stride<_8, _1>>{},
                                 Layout<Shape<_1,_1>>{});
    auto mmaC = make_tiled_mma(UniversalFMA<float, float, float>{},
                               Layout<Shape<_16,_16,_1>>{});
    const dim3 block(size(mmaC));
    const dim3 grid((n + 127) / 128,
                    (m + 127) / 128);
    
     gemm_device<<<grid, block, 0, stream>>>(
        shape_mnk,
        cta_tiler,
        a, a_stride, sA_layout, copyA,
        b, b_stride, sB_layout, copyB,
        c, c_stride, sC_layout, mmaC,
        1.0f, 0.0f);
}
