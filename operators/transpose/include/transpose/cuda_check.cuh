#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace transpose {

inline void cuda_check(cudaError_t error,
                       const char* expression,
                       const char* file,
                       int line) {
    if (error == cudaSuccess) {
        return;
    }

    std::fprintf(stderr,
                 "CUDA error at %s:%d: %s failed: %s\n",
                 file,
                 line,
                 expression,
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

}  // namespace transpose

#define CUDA_CHECK(expression) \
    ::transpose::cuda_check((expression), #expression, __FILE__, __LINE__)
