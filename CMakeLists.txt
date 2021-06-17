cmake_minimum_required(VERSION 3.17)

# project name
project(CUDA_CMAKE LANGUAGES CXX CUDA)
include(FetchContent)
#FetchContent_Declare(
#   GoogleBenchmark
#   URL https://github.com/google/benchmark/archive/v1.4.1.tar.gz
#   )
  
 FetchContent_Declare(
   CLI11
   URL https://github.com/CLIUtils/CLI11/archive/v1.8.0.tar.gz
   )
  
 FetchContent_Declare(
   spdlog
   URL https://github.com/gabime/spdlog/archive/v1.4.2.tar.gz
   )

 #if (NOT GoogleBenchmark_POPULATED)
 # FetchContent_Populate(GoogleBenchmark)
 # set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
 # set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
 # add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
 #endif()

FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)

#include directories
include_directories(include)

set(CMAKE_CXX_FlAGS "-g")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

# find dependencies
FIND_PACKAGE(OpenCV REQUIRED)

# build a shared library with our CUDA code
add_library(myCudaLib
        SHARED
        src/gpu_lbp.cu
        )

target_compile_features(myCudaLib PUBLIC cxx_std_14)
set_target_properties(myCudaLib
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(myCudaLib)

# build the C++ code and link with the CUDA code
ADD_EXECUTABLE(gpgpu
        src/main.cc
        src/lbp.cc
        src/train_kmeans.cc
        src/nearest_neighbour.cc
        src/cpu_lbp.cc
        src/utils.cc
        )

TARGET_LINK_LIBRARIES(gpgpu
        myCudaLib ${OpenCV_LIBS}
        spdlog::spdlog
        CLI11::CLI11
        )
