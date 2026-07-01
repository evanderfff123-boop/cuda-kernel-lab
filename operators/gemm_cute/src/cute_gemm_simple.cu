// 用于学习以及快速上手
#include <cute/tensor.hpp>

__global__ 
void
gemm_device(const float* A, const float* B, float* C, int M, int N, int K){
    using namespace cute;

    using bM = Int<128>;
    using bN = Int<128>;
    using bK = Int<8>;

    auto shape_MNK = make_shape(M, N, K);
    auto cta_tiler = make_shape(bM, bN, bK);

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), make_stride(K, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), make_stride(Int<1>{}, N));
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), make_stride(N, Int<1>{}));

    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);  // coord中使得A的M被blockIdx.y 切片分走了
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // BLK_M, BLK_K, k; 当你想问M没被分完的去哪里，请看上面的coord
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    auto sA_layout = make_layout(make_shape(bM{}, bK{}));
    auto sB_layout = make_layout(make_shape(bN{}, bK{}));

    __shared__ float smemA[cosize_v<decltype(sA_layout)>];
    __shared__ float smemB[cosize_v<decltype(sB_layout)>];

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);  // BLK_M, BLK_K
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    auto copy_a = make_tiled_copy(Copy_Atom<UniversalCopy<float>, float>{},
                                 Layout<Shape<_32, _8>, Stride<_8, _1>>{},
                                 Layout<Shape<_1, _1>{});
    auto copy_b = make_tiled_copy(Copy_Atom<UniversalCopy<float>, float>{},
                                 Layout<Shape<_32, _8>, Stride<_8, _1>>{},
                                 Layout<Shape<_1, _1>{});
    auto mma = make_tile_mma(UniversalFMA<float, float, float>{},
                             Layout<Shape<_16, _16, _1>>{});

    auto thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);  // CPY, CPY_M, CPY_K, k; 这里CPY是(_1, _1), CPY_M是4，表示需要在M维度上重复4次
    Tensor tAsA = thr_copy_a.partition_D(sA);  // CPY, CPY_M, CPY_K

    auto thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);

    auto thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    int k_tile_max = size<3>(tAgA);
    for(int k_tile = 0; k_tile < k_tile_max; k_tile++){
        copy(copy_a, tAgA(_,_,_,k_tile), tAsA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBsB);

        __syncthreads();

        gemm(mma, tCsA, tCsB, tCrC);

        __syncthreads();
    }

    axpby(1.0f, tCrC, 0.0f, tCgC);
}

void launch_gemm_simple(const float* a, const float* b, float* c, int m, int n, int k, cudaStream_t stream){
    dim3 block(256);
    dim3 grid((n + 127) / 128, (m + 127) / 128);

    gemm_simple<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}