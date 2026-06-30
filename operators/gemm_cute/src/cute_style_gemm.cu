// Reserved for the next learning stage: CuTe-style GEMM.
//
// The kernels in cuda_core_gemm.cu are still mostly "CUDA core GEMM with CuTe
// Tensor/Layout indexing". This file is intended for the next step where the
// implementation starts to use more CuTe-native concepts such as:
//
//   - CTA tiler as a first-class object
//   - logical B layout [N, K]
//   - thread layout
//   - tiled copy
//   - eventually TiledMMA / Tensor Core MMA
//
// Keep this file empty until the first CuTe-style kernel is introduced.
