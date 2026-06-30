#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

void launch_tensor_index_gemm(const float* a,
                         const float* b,
                         float* c,
                         int m,
                         int n,
                         int k,
                         cudaStream_t stream);

void launch_cta_tile_gemm(const float* a,
                         const float* b,
                         float* c,
                         int m,
                         int n,
                         int k,
                         cudaStream_t stream);

void launch_k_tile_gemm(const float* a,
                         const float* b,
                         float* c,
                         int m,
                         int n,
                         int k,
                         cudaStream_t stream);

void launch_smem_tile_gemm(const float* a,
                         const float* b,
                         float* c,
                         int m,
                         int n,
                         int k,
                         cudaStream_t stream);

void launch_cta_tiler_style_gemm(const float* a,
                                 const float* b,
                                 float* c,
                                 int m,
                                 int n,
                                 int k,
                                 cudaStream_t stream);

void launch_smem_tensor_style_gemm(const float* a,
                                   const float* b,
                                   float* c,
                                   int m,
                                   int n,
                                   int k,
                                   cudaStream_t stream);

void launch_copy_partition_style_gemm(const float* a,
                                      const float* b,
                                      float* c,
                                      int m,
                                      int n,
                                      int k,
                                      cudaStream_t stream);

void launch_math_partition_style_gemm(const float* a,
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

torch::Tensor cute_gemm_dispatch(torch::Tensor a,
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
        launch_tensor_index_gemm(a.data_ptr<float>(),
                            b.data_ptr<float>(),
                            c.data_ptr<float>(),
                            m,
                            n,
                            k,
                            at::cuda::getCurrentCUDAStream());
    } else if (version == 2) {
        launch_cta_tile_gemm(a.data_ptr<float>(),
                            b.data_ptr<float>(),
                            c.data_ptr<float>(),
                            m,
                            n,
                            k,
                            at::cuda::getCurrentCUDAStream());
    } else if (version == 3) {
        launch_k_tile_gemm(a.data_ptr<float>(),
                            b.data_ptr<float>(),
                            c.data_ptr<float>(),
                            m,
                            n,
                            k,
                            at::cuda::getCurrentCUDAStream());
    } else if (version == 4) {
        launch_smem_tile_gemm(a.data_ptr<float>(),
                            b.data_ptr<float>(),
                            c.data_ptr<float>(),
                            m,
                            n,
                            k,
                            at::cuda::getCurrentCUDAStream());
    } else if (version == 5) {
        launch_cta_tiler_style_gemm(a.data_ptr<float>(),
                                    b.data_ptr<float>(),
                                    c.data_ptr<float>(),
                                    m,
                                    n,
                                    k,
                                    at::cuda::getCurrentCUDAStream());
    } else if (version == 6) {
        launch_smem_tensor_style_gemm(a.data_ptr<float>(),
                                      b.data_ptr<float>(),
                                      c.data_ptr<float>(),
                                      m,
                                      n,
                                      k,
                                      at::cuda::getCurrentCUDAStream());
    } else if (version == 7) {
        launch_copy_partition_style_gemm(a.data_ptr<float>(),
                                         b.data_ptr<float>(),
                                         c.data_ptr<float>(),
                                         m,
                                         n,
                                         k,
                                         at::cuda::getCurrentCUDAStream());
    } else if (version == 8) {
        launch_math_partition_style_gemm(a.data_ptr<float>(),
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

torch::Tensor tensor_index_gemm(torch::Tensor a, torch::Tensor b) {
    return cute_gemm_dispatch(a, b, 1);
}

torch::Tensor cta_tile_gemm(torch::Tensor a, torch::Tensor b) {
    return cute_gemm_dispatch(a, b, 2);
}

torch::Tensor k_tile_gemm(torch::Tensor a, torch::Tensor b) {
    return cute_gemm_dispatch(a, b, 3);
}

torch::Tensor smem_tile_gemm(torch::Tensor a, torch::Tensor b) {
    return cute_gemm_dispatch(a, b, 4);
}

torch::Tensor cta_tiler_style_gemm(torch::Tensor a, torch::Tensor b) {
    return cute_gemm_dispatch(a, b, 5);
}

torch::Tensor smem_tensor_style_gemm(torch::Tensor a, torch::Tensor b) {
    return cute_gemm_dispatch(a, b, 6);
}

torch::Tensor copy_partition_style_gemm(torch::Tensor a, torch::Tensor b) {
    return cute_gemm_dispatch(a, b, 7);
}

torch::Tensor math_partition_style_gemm(torch::Tensor a, torch::Tensor b) {
    return cute_gemm_dispatch(a, b, 8);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_index",
          &tensor_index_gemm,
          "CuTe GEMM: naive fp32 GEMM using CuTe Tensor indexing");
    m.def("cta_tile",
          &cta_tile_gemm,
          "CuTe GEMM: fp32 GEMM with CuTe C output tiling");
    m.def("k_tile",
          &k_tile_gemm,
          "CuTe GEMM: fp32 GEMM with CuTe A/B/C K tiling");
    m.def("smem_tile",
          &smem_tile_gemm,
          "CuTe GEMM: fp32 GEMM with CuTe shared-memory A/B tiles");
    m.def("cta_tiler_style",
          &cta_tiler_style_gemm,
          "CuTe-style GEMM: CTA tiler with logical B [N, K]");
    m.def("smem_tensor_style",
          &smem_tensor_style_gemm,
          "CuTe-style GEMM: shared-memory tensors from explicit SMEM layouts");
    m.def("copy_partition_style",
          &copy_partition_style_gemm,
          "CuTe-style GEMM: copy partitioning for gmem-to-smem tile loads");
    m.def("math_partition_style",
          &math_partition_style_gemm,
          "CuTe-style GEMM: math partitioning with cute::gemm");

    // Backward-compatible aliases while the tutorial is still evolving.
    m.def("forward", &tensor_index_gemm);
    m.def("forward_v2", &cta_tile_gemm);
    m.def("forward_v3", &k_tile_gemm);
    m.def("forward_v4", &smem_tile_gemm);
    m.def("forward_v5", &cta_tiler_style_gemm);
    m.def("forward_v6", &smem_tensor_style_gemm);
    m.def("forward_v7", &copy_partition_style_gemm);
    m.def("forward_v8", &math_partition_style_gemm);
}
