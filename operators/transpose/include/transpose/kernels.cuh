#pragma once

#include <cuda_runtime.h>

#include <vector>

namespace transpose {

// Input and output are row-major [m, m] matrices, so
// output[col * m + row] = input[row * m + col].
using KernelFn = void (*)(int m,
                          float* input,
                          float* output,
                          cudaStream_t stream);

struct KernelInfo {
    const char* name;
    const char* description;
    KernelFn launch;
};

void launch_naive(int m,
                  float* input,
                  float* output,
                  cudaStream_t stream = nullptr);

void launch_smem(int m,
                 float* input,
                 float* output,
                 cudaStream_t stream = nullptr);

void launch_padding(int m,
                    float* input,
                    float* output,
                    cudaStream_t stream = nullptr);

void launch_cublas(int m,
                   float* input,
                   float* output,
                   cudaStream_t stream = nullptr);

void launch_tiled(int m,
                  float* input,
                  float* output,
                  cudaStream_t stream = nullptr);

// Keep benchmark and tests independent from the number of kernel versions.
const std::vector<KernelInfo>& kernels();

}  // namespace transpose
