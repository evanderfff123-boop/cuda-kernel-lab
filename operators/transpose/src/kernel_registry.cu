#include "transpose/kernels.cuh"

namespace transpose {

const std::vector<KernelInfo>& kernels() {
    static const std::vector<KernelInfo> registry = {
        {
            "cublas",
            "cuBLAS GEAM transpose",
            launch_cublas,
        },
        {
            "naive",
            "direct global-memory transpose",
            launch_naive,
        },
        {
            "smem",
            "32x32 shared-memory transpose",
            launch_smem,
        },
        {
            "padding",
            "32x33 padded shared-memory transpose",
            launch_padding,
        },
        {
            "tiled",
            "32x32 padded tile processed by a 32x8 thread block",
            launch_tiled,
        },
    };
    return registry;
}

}  // namespace transpose
