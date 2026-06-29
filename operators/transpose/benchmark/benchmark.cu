#include "transpose/cuda_check.cuh"
#include "transpose/kernels.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
    int warmup = 10;
    int iterations = 100;
    bool csv = false;
    bool check = false;
    bool list = false;
    std::string kernel_filter;
    std::vector<int> sizes = {
        256,
        1024,
        2048,
        4096,
        1023,
    };
};

void print_usage(const char* program) {
    std::printf(
        "Usage: %s [options]\n"
        "  --size N             benchmark one N x N matrix\n"
        "  --kernel NAME        run kernels whose names contain NAME\n"
        "  --warmup N            warmup iterations (default: 10)\n"
        "  --iters N             measured iterations (default: 100)\n"
        "  --check               verify every measured kernel\n"
        "  --csv                 emit machine-readable CSV\n"
        "  --list                list registered kernels\n"
        "  --help                show this message\n",
        program);
}

int parse_positive_int(const char* text, const char* option) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (text[0] == '\0' || *end != '\0' || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "Invalid value for %s: %s\n", option, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

const char* require_value(int& index,
                          int argc,
                          char** argv,
                          const char* option) {
    if (index + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", option);
        std::exit(EXIT_FAILURE);
    }
    return argv[++index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    int size = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--size") {
            size = parse_positive_int(
                require_value(i, argc, argv, "--size"), "--size");
        } else if (arg == "--warmup") {
            options.warmup = parse_positive_int(
                require_value(i, argc, argv, "--warmup"), "--warmup");
        } else if (arg == "--iters") {
            options.iterations = parse_positive_int(
                require_value(i, argc, argv, "--iters"), "--iters");
        } else if (arg == "--kernel") {
            options.kernel_filter =
                require_value(i, argc, argv, "--kernel");
        } else if (arg == "--csv") {
            options.csv = true;
        } else if (arg == "--check") {
            options.check = true;
        } else if (arg == "--list") {
            options.list = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    if (size > 0) {
        options.sizes = {size};
    }
    return options;
}

class Event {
public:
    Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~Event() { cudaEventDestroy(event_); }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{};
};

float finish_measurement(Event& start, Event& stop, int iterations) {
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start.get(), stop.get()));
    return total_ms / iterations;
}

float measure_kernel(transpose::KernelFn launch,
                     int m,
                     float* input,
                     float* output,
                     int warmup,
                     int iterations) {
    for (int i = 0; i < warmup; ++i) {
        launch(m, input, output, nullptr);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Event start;
    Event stop;
    CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < iterations; ++i) {
        launch(m, input, output, nullptr);
    }
    return finish_measurement(start, stop, iterations);
}

void initialize(std::vector<float>& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        // Exactly representable values make transpose validation strict and
        // deterministic without spending time on random-number generation.
        values[i] = static_cast<float>(static_cast<int>(i % 4093) - 2046);
    }
}

bool validate(int m,
              const std::vector<float>& input,
              const std::vector<float>& output) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < m; ++col) {
            const float expected = input[static_cast<std::size_t>(row) * m + col];
            const float actual = output[static_cast<std::size_t>(col) * m + row];
            if (actual != expected) {
                std::fprintf(stderr,
                             "Mismatch at output[%d, %d]: got %.9g, expected %.9g\n",
                             col,
                             row,
                             actual,
                             expected);
                return false;
            }
        }
    }
    return true;
}

double effective_bandwidth_gbps(std::size_t matrix_bytes, float milliseconds) {
    // One full input read plus one full output write.
    return 2.0 * static_cast<double>(matrix_bytes) /
           (static_cast<double>(milliseconds) * 1.0e6);
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    const auto& registry = transpose::kernels();
    const transpose::KernelInfo* cublas_kernel = nullptr;

    for (const auto& kernel : registry) {
        if (std::string(kernel.name) == "cublas") {
            cublas_kernel = &kernel;
            break;
        }
    }
    if (cublas_kernel == nullptr) {
        std::fprintf(stderr, "cuBLAS baseline is not registered\n");
        return EXIT_FAILURE;
    }

    if (options.list) {
        for (const auto& kernel : registry) {
            std::printf("%-16s %s\n", kernel.name, kernel.description);
        }
        return EXIT_SUCCESS;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    if (!options.csv) {
        std::printf("Device: %s (sm_%d%d)\n", properties.name,
                    properties.major, properties.minor);
        std::printf("Timing: %d warmup, %d measured iterations\n\n",
                    options.warmup, options.iterations);
        std::printf("%8s  %-16s %11s %13s %11s %7s\n",
                    "size", "kernel", "time(ms)", "GB/s",
                    "cublas_eff", "check");
        std::printf("%8s  %-16s %11s %13s %11s %7s\n",
                    "--------", "----------------", "-----------",
                    "-------------", "-----------", "-------");
    } else {
        std::printf(
            "size,kernel,time_ms,bandwidth_gbps,cublas_efficiency_pct,correct\n");
    }

    bool all_correct = true;
    for (const int m : options.sizes) {
        const std::size_t elements =
            static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
        const std::size_t bytes = elements * sizeof(float);

        std::vector<float> host_input(elements);
        std::vector<float> host_output;
        initialize(host_input);
        if (options.check) {
            host_output.resize(elements);
        }

        float* device_input = nullptr;
        float* device_output = nullptr;
        CUDA_CHECK(cudaMalloc(&device_input, bytes));
        CUDA_CHECK(cudaMalloc(&device_output, bytes));
        CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), bytes,
                              cudaMemcpyHostToDevice));

        const float cublas_ms = measure_kernel(
            cublas_kernel->launch,
            m,
            device_input,
            device_output,
            options.warmup,
            options.iterations);

        for (const auto& kernel : registry) {
            if (!options.kernel_filter.empty() &&
                std::string(kernel.name).find(options.kernel_filter) ==
                    std::string::npos) {
                continue;
            }
            const bool is_cublas = std::string(kernel.name) == "cublas";
            const float elapsed_ms =
                is_cublas
                    ? cublas_ms
                    : measure_kernel(kernel.launch,
                                     m,
                                     device_input,
                                     device_output,
                                     options.warmup,
                                     options.iterations);
            const double bandwidth = effective_bandwidth_gbps(bytes, elapsed_ms);
            const double cublas_efficiency =
                elapsed_ms > 0.0f ? 100.0 * cublas_ms / elapsed_ms : 0.0;

            bool correct = true;
            if (options.check) {
                CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, bytes,
                                      cudaMemcpyDeviceToHost));
                correct = validate(m, host_input, host_output);
                all_correct = all_correct && correct;
            }

            const char* check_text =
                options.check ? (correct ? "PASS" : "FAIL") : "-";
            if (options.csv) {
                std::printf("%d,%s,%.6f,%.3f,%.2f,%s\n",
                            m, kernel.name, elapsed_ms, bandwidth,
                            cublas_efficiency, check_text);
            } else {
                std::printf("%8d  %-16s %11.6f %13.3f %10.2f%% %7s\n",
                            m, kernel.name, elapsed_ms, bandwidth,
                            cublas_efficiency, check_text);
            }
        }

        CUDA_CHECK(cudaFree(device_input));
        CUDA_CHECK(cudaFree(device_output));
    }

    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
