#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

namespace {

__global__ void flash_attn_fwd_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, float scale, int Tc, int Tr, int Bc, int Br
) {
    (void)N;
    (void)Tr;

    int tx = threadIdx.x; 
    int bx = blockIdx.x;  

    // 动态共享内存切分
    extern __shared__ float s_mem[]; 
    float* s_Q = s_mem;              
    float* s_K = s_Q + Br * d;       
    float* s_V = s_K + Bc * d;       

    // 初始化寄存器状态
    float m_old = -INFINITY;
    float l_old = 0.0f;
    float O_row[64] = {0.0f}; 

    // 一次性载入 Q 分块到共享内存
    int global_q_row = bx * Br + tx; 
    for (int col = 0; col < d; ++col) {
        s_Q[tx * d + col] = Q[global_q_row * d + col];
    }
    __syncthreads(); 

    // 外层循环：遍历 K, V 的所有分块
    for (int j = 0; j < Tc; ++j) {
        int global_kv_row = j * Bc + tx; 
        
        // 协作载入 K, V 到共享内存
        for (int col = 0; col < d; ++col) {
            s_K[tx * d + col] = K[global_kv_row * d + col];
            s_V[tx * d + col] = V[global_kv_row * d + col];
        }
        __syncthreads();

        // 1. 计算 QK^T 点积得分
        float S_row[64] = {0.0f}; 
        float m_block = -INFINITY;
        for (int k = 0; k < Bc; ++k) {
            float sum = 0.0f;
            for (int col = 0; col < d; ++col) {
                sum += s_Q[tx * d + col] * s_K[k * d + col];
            }
            S_row[k] = sum * scale;
            if (S_row[k] > m_block) {
                m_block = S_row[k];
            }
        }

        // 2. 计算指数和 l_block
        float l_block = 0.0f;
        for (int k = 0; k < Bc; ++k) {
            S_row[k] = expf(S_row[k] - m_block);
            l_block += S_row[k];
        }

        // 3. 计算递推新状态
        float m_new = fmaxf(m_old, m_block);
        float exp_old = expf(m_old - m_new);
        float exp_block = expf(m_block - m_new);
        float l_new = exp_old * l_old + exp_block * l_block;

        // 4. 更新局部输出 O_row
        for (int col = 0; col < d; ++col) {
            float pv_sum = 0.0f;
            for (int k = 0; k < Bc; ++k) {
                pv_sum += S_row[k] * s_V[k * d + col];
            }
            O_row[col] = (exp_old * l_old * O_row[col] + exp_block * pv_sum) / l_new;
        }

        // 5. 更新状态参数，进入下一次迭代
        m_old = m_new;
        l_old = l_new;

        __syncthreads(); 
    }

    // 写回全局显存
    for (int col = 0; col < d; ++col) {
        O[global_q_row * d + col] = O_row[col];
    }
}

}  // namespace

void launch_flash_attn_fwd_cuda(const float* q,
                                const float* k,
                                const float* v,
                                float* o,
                                int n,
                                int d,
                                float scale,
                                cudaStream_t stream) {
    constexpr int br = 64;
    constexpr int bc = 64;

    const int tr = n / br;
    const int tc = n / bc;

    const dim3 grid(tr);
    const dim3 block(br);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(br * d + bc * d + bc * d) * sizeof(float);

    flash_attn_fwd_kernel<<<grid, block, shared_bytes, stream>>>(
        q,
        k,
        v,
        o,
        n,
        d,
        scale,
        tc,
        tr,
        bc,
        br);
}
