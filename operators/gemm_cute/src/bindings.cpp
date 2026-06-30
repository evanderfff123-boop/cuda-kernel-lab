#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

void launch_cute_gemm_v1(const float* a,
                         const float* b,
                         float* c,
                         int m,
                         int n,
                         int k,
                         cudaStream_t stream);

void launch_cute_gemm_v2(const float* a,
                         const float* b,
                         float* c,
                         int m,
                         int n,
                         int k,
                         cudaStream_t stream);

void launch_cute_gemm_v3(const float* a,
                         const float* b,
                         float* c,
                         int m,
                         int n,
                         int k,
                         cudaStream_t stream);

namespace {

void check_input(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                name, " must have dtype torch.float32");
    TORCH_CHECK(tensor.dim() == 2, name, " must be a rank-2 tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda(cudaError_t error) {
    TORCH_CHECK(error == cudaSuccess,
                "CUDA error: ",
                cudaGetErrorString(error));
}

}  // namespace

torch::Tensor cute_gemm(torch::Tensor a,
                        torch::Tensor b,
                        int version) {
    check_input(a, "a");
    check_input(b, "b");

    TORCH_CHECK(a.size(1) == b.size(0),
                "shape mismatch: a is [M, K], b must be [K, N]");

    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b.size(1));

    auto c = torch::empty({m, n}, a.options());

    if (version == 1) {
        launch_cute_gemm_v1(a.data_ptr<float>(),
                            b.data_ptr<float>(),
                            c.data_ptr<float>(),
                            m,
                            n,
                            k,
                            at::cuda::getCurrentCUDAStream());
    } else if (version == 2) {
        launch_cute_gemm_v2(a.data_ptr<float>(),
                            b.data_ptr<float>(),
                            c.data_ptr<float>(),
                            m,
                            n,
                            k,
                            at::cuda::getCurrentCUDAStream());
    } else if (version == 3) {
        launch_cute_gemm_v3(a.data_ptr<float>(),
                            b.data_ptr<float>(),
                            c.data_ptr<float>(),
                            m,
                            n,
                            k,
                            at::cuda::getCurrentCUDAStream());
    } else {
        TORCH_CHECK(false, "unsupported CuTe GEMM version: ", version);
    }

    check_cuda(cudaGetLastError());
    return c;
}

torch::Tensor cute_gemm_v1(torch::Tensor a, torch::Tensor b) {
    return cute_gemm(a, b, 1);
}

torch::Tensor cute_gemm_v2(torch::Tensor a, torch::Tensor b) {
    return cute_gemm(a, b, 2);
}

torch::Tensor cute_gemm_v3(torch::Tensor a, torch::Tensor b) {
    return cute_gemm(a, b, 3);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",
          &cute_gemm_v1,
          "CuTe GEMM v1: naive fp32 GEMM using CuTe Tensor indexing");
    m.def("forward_v2",
          &cute_gemm_v2,
          "CuTe GEMM v2: naive fp32 GEMM with CuTe C output tiling");
    m.def("forward_v3",
          &cute_gemm_v3,
          "CuTe GEMM v3: naive fp32 GEMM with CuTe A, B input and C output tiling");
}
