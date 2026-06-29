cmake_minimum_required(VERSION 3.18)
project(cuda_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(GPU_ARCH "75" CACHE STRING
    "CUDA architecture to build for, for example 75, 80, or 86")
set(CMAKE_CUDA_ARCHITECTURES "${GPU_ARCH}")

option(BUILD_BENCHMARK "Build benchmark executables" ON)
option(BUILD_TESTS "Build correctness tests" ON)

find_package(CUDAToolkit REQUIRED)

if(BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(operators/transpose)
