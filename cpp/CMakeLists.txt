cmake_minimum_required(VERSION 3.20)
project(phaseflow_native LANGUAGES CXX)

add_library(phaseflow_native STATIC src/model.cpp)
target_include_directories(phaseflow_native PUBLIC include)
target_compile_features(phaseflow_native PUBLIC cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(phaseflow_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(phaseflow_cli src/phaseflow_cli.cpp)
target_link_libraries(phaseflow_cli PRIVATE phaseflow_native)

# The shared .cuh header contains allocation-free __host__ __device__ numerical
# routines. This build intentionally needs no CUDA toolkit. A renderer must
# compile and validate its CUDA/OptiX integration on its own target hardware.
