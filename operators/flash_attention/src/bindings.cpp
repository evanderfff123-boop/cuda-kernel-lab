#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cmath>

void launch_flash_attn_fwd_cuda(const float* q,
                                const float* k,
                                const float* v,
                                float* o,
                                int n,
                                int d,
                                float scale,
                                cudaStream_t stream);

namespace {

void check_input(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                name, " must have dtype torch.float32");
    TORCH_CHECK(tensor.dim() == 2, name, " must have shape [N, D]");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor fa_v1_forward(torch::Tensor q,
                            torch::Tensor k,
                            torch::Tensor v) {
    check_input(q, "q");
    check_input(k, "k");
    check_input(v, "v");

    TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");
    TORCH_CHECK(q.sizes() == v.sizes(), "q and v must have the same shape");

    const int64_t n = q.size(0);
    const int64_t d = q.size(1);
    TORCH_CHECK(n > 0, "N must be positive");
    TORCH_CHECK(n % 64 == 0,
                "flash attention v1 currently requires N to be divisible by 64");
    TORCH_CHECK(d == 64, "flash attention v1 currently requires D == 64");

    auto o = torch::empty_like(q);
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(d));
    launch_flash_attn_fwd_cuda(q.data_ptr<float>(),
                               k.data_ptr<float>(),
                               v.data_ptr<float>(),
                               o.data_ptr<float>(),
                               static_cast<int>(n),
                               static_cast<int>(d),
                               softmax_scale,
                               at::cuda::getCurrentCUDAStream());
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",
          &fa_v1_forward,
          "FlashAttention v1 forward: q/k/v [N, 64] fp32 -> o [N, 64]");
}
