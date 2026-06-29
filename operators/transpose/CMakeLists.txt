add_library(transpose_kernels STATIC
    src/kernel_registry.cu
    src/transpose_cublas.cu
    src/transpose_naive.cu
    src/transpose_padding.cu
    src/transpose_smem.cu
    src/transpose_tiled.cu
)

target_include_directories(transpose_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(transpose_kernels
    PUBLIC
        CUDA::cudart
        CUDA::cublas
)

if(BUILD_TESTS)
    add_executable(transpose_correctness_test
        tests/correctness_test.cu
    )

    target_link_libraries(transpose_correctness_test
        PRIVATE
            transpose_kernels
            CUDA::cudart
    )

    add_test(NAME transpose_correctness
             COMMAND transpose_correctness_test)
endif()

if(BUILD_BENCHMARK)
    add_executable(transpose_benchmark
        benchmark/benchmark.cu
    )

    target_link_libraries(transpose_benchmark
        PRIVATE
            transpose_kernels
            CUDA::cudart
    )
endif()
