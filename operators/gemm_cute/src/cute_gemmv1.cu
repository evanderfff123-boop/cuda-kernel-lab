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

    // 全局内存张量的创建
    // make_gmem_ptr: 将原始指针包装为CUTE识别的全局内存指针
    // select<0, 2>(shape_MNK):从(M, N, K)的元组中选择维度M和K
    // make_tensor: 结合数据指针，形状，步长，构建出一个完整的全局内存Tensor对象
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);
    
    // 线程块层面的分块
    // cta_tiler: 在外层定义的形状，表示一个线程块（CTA）处理的子矩阵的大小
    // cta_coord: 当前线程块在网格中的坐标
    // local_tile: 将全局大矩阵分割成当前线程块需要处理的局部块
    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);  // (m, n, k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});
    
    // 共享内存的声明与Tensor包装
    // cosize_v: 计算一个共享内存Layout所需的连续物理内存大小，用于静态声明共享内存的大小
    // make_smem_ptr: 包装共享内存的指针
    // 将共享内存地址与外层传入的共享内存布局绑定
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    // 线程级划分--数据搬运部分
    // tA或者tB表示搬运数据时的线程布局
    // local_partition: 依据线程布局和当前线程ID，计算出当前线程具体负责搬运哪一部分数据
    // tAgA：当前线程在全局内存负责读的子区域
    // tAsA: 当前线程在共享内存负责写的子区域
    Tensor tAgA = local_partition(gA, tA, threadIdx.x);
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);

    Tensor tBgB = local_partition(gB, tB, threadIdx.x);
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);

    // 线程级划分--计算和累加部分
    // tC表示计算时使用的线程布局
    // local_partition: tCsA/ tCsB:根据计算布局tC划分共享内存，得到每个线程在做乘法时需要读取的A和B的片段
    //                  tCgC：当前线程负责写回的C矩阵全局内存区域
    // make_tensor_like: 在寄存器中创建一个与tCgC结构相同的Tensor tCrC，这个Tensor被当作累加器
    // clear: 将寄存器的累加器清零
    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});
    Tensor tCrC = make_tensor_like(tCgC);

    clear(tCrC);

    // 主循环--K轴迭代计算
    // K_TILE_MAX：沿着K轴需要迭代的总步数
    // 数据搬运：将第k_tile块的A和B数据从全局内存拷贝到共享内存
    //           在Ampere架构及以上，会变为异步拷贝指令：cp.async
    // cp_async_fence(),cp_async_wait<0>(): 用于确保当前迭代发起的异步拷贝操作全部完成
    // __syncthreads(): 屏障同步，确保线程块内的所有线程都已将数据写入共享内存，之后才能开始计算。
    // gemm：计算：每个线程调用线程级的矩阵乘法，将共享内存的片段tCsA和tCsB相乘，并累加到寄存器中
    // __syncthreads()：计算后同步
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

    // 写回全局内存
    // axpby：计算D = alpha * Y + beta * X
    // 由于alpha = 1.0f， beta = 0.0f
    // 所以等价于直接将结果写入全局内存
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
