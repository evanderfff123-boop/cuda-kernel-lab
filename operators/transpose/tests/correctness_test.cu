#include "transpose/cuda_check.cuh"
#include "transpose/kernels.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

void initialize(std::vector<float>& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(static_cast<int>(i % 251) - 125);
    }
}

bool validate(int m,
              const std::vector<float>& input,
              const std::vector<float>& output) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < m; ++col) {
            const std::size_t input_index =
                static_cast<std::size_t>(row) * m + col;
            const std::size_t output_index =
                static_cast<std::size_t>(col) * m + row;
            if (output[output_index] != input[input_index]) {
                std::fprintf(
                    stderr,
                    "  first mismatch: input[%d, %d]=%.9g, output[%d, %d]=%.9g\n",
                    row,
                    col,
                    input[input_index],
                    col,
                    row,
                    output[output_index]);
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    const std::vector<int> sizes = {
        1,
        7,
        31,
        32,
        33,
        127,
        513,
        1024,
    };

    int passed = 0;
    int total = 0;

    for (const int m : sizes) {
        const std::size_t elements =
            static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
        const std::size_t bytes = elements * sizeof(float);

        std::vector<float> input(elements);
        std::vector<float> output(elements);
        initialize(input);

        float* device_input = nullptr;
        float* device_output = nullptr;
        CUDA_CHECK(cudaMalloc(&device_input, bytes));
        CUDA_CHECK(cudaMalloc(&device_output, bytes));
        CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes,
                              cudaMemcpyHostToDevice));

        for (const auto& kernel : transpose::kernels()) {
            ++total;
            CUDA_CHECK(cudaMemset(device_output, 0xFF, bytes));
            kernel.launch(m, device_input, device_output, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(output.data(), device_output, bytes,
                                  cudaMemcpyDeviceToHost));

            const bool ok = validate(m, input, output);
            passed += ok ? 1 : 0;
            std::printf("[%s] %dx%d: %s\n",
                        kernel.name, m, m, ok ? "PASS" : "FAIL");
        }

        CUDA_CHECK(cudaFree(device_input));
        CUDA_CHECK(cudaFree(device_output));
    }

    std::printf("\n%d / %d tests passed\n", passed, total);
    return passed == total ? EXIT_SUCCESS : EXIT_FAILURE;
}
