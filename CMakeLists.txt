cmake_minimum_required(VERSION 3.17)

project(GPGPU LANGUAGES CXX CUDA)
include(FetchContent)

FetchContent_Declare(
   GoogleBenchmark
   URL https://github.com/google/benchmark/archive/v1.5.5.tar.gz
   )

FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.8.0.tar.gz
  )

if (NOT GoogleBenchmark_POPULATED)
    FetchContent_Populate(GoogleBenchmark)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
    add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

FetchContent_MakeAvailable(CLI11)

include_directories(include)

set(CMAKE_CXX_FlAGS "-g")
set(CMAKE_DEBUG_CXX_FlAGS "-fsanitize=address")

find_package(OpenCV REQUIRED)

add_library(lbp_lib SHARED src/gpu_lbp.cu)
target_compile_features(lbp_lib PUBLIC cxx_std_17)
set_target_properties(lbp_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test
        src/main.cc
        src/render.cc
        src/cpu_lbp.cc
        src/train_kmeans.cc
        src/nearest_neighbour.cc
        src/utils.cc
        )
target_compile_features(test PUBLIC cxx_std_17)
target_link_libraries(test
        lbp_lib
        "opencv_core;opencv_imgcodecs;opencv_highgui"
        CLI11::CLI11
        )

add_executable(bench
        src/benchmark.cc
        src/render.cc
        src/cpu_lbp.cc
        src/train_kmeans.cc
        src/nearest_neighbour.cc
        src/utils.cc
        )
target_compile_features(bench PUBLIC cxx_std_17)
target_link_libraries(bench
        lbp_lib
        "opencv_core;opencv_imgcodecs;opencv_highgui"
        benchmark
        )
