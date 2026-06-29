#include "transpose/kernels.cuh"

#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>

namespace transpose {
namespace {

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "cuBLAS error: %s failed with status %d\n",
                     operation,
                     static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

class CublasHandle {
public:
    CublasHandle() {
        check_cublas(cublasCreate(&handle_), "cublasCreate");
    }

    ~CublasHandle() {
        cublasDestroy(handle_);
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const {
        return handle_;
    }

private:
    cublasHandle_t handle_{};
};

cublasHandle_t cublas_handle() {
    static CublasHandle handle;
    return handle.get();
}

}  // namespace

void launch_cublas(int m,
                   float* input,
                   float* output,
                   cudaStream_t stream) {
    cublasHandle_t handle = cublas_handle();
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS treats the row-major input buffer as the column-major transpose
    // of the same matrix. Transposing that view produces the desired
    // row-major transposed output in memory.
    check_cublas(
        cublasSgeam(handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_T,
                    m,
                    m,
                    &alpha,
                    input,
                    m,
                    &beta,
                    input,
                    m,
                    output,
                    m),
        "cublasSgeam");
}

}  // namespace transpose
