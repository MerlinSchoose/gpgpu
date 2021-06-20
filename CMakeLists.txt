cmake_minimum_required(VERSION 3.17)

project(GPGPU LANGUAGES CXX CUDA)
include(FetchContent)

find_package(Boost 1.75.0 REQUIRED COMPONENTS serialization) 


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

include_directories(include ${Boost_INCLUDE_DIRS}) 

set(CMAKE_DEBUG_CXX_FlAGS "-fsanitize=address")

find_package(OpenCV REQUIRED)

add_library(lbp_lib SHARED src/gpu_lbp.cu)
target_compile_features(lbp_lib PUBLIC cxx_std_17)
set_target_properties(lbp_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(SOURCES
  src/cpu_lbp.cc
  src/render.cc
  src/train_kmeans.cc
  src/nearest_neighbour.cc
  src/serialize.cc
  src/utils.cc)

add_executable(train
        src/train.cc
        ${SOURCES}
        )
target_compile_features(train PUBLIC cxx_std_17)
target_link_libraries(train
        lbp_lib
        "opencv_core;opencv_imgcodecs;opencv_highgui"
        ${Boost_LIBRARIES}
        )



add_executable(test
        src/main.cc
        ${SOURCES}
        )
target_compile_features(test PUBLIC cxx_std_17)
target_link_libraries(test
        lbp_lib
        "opencv_core;opencv_imgcodecs;opencv_highgui"
        CLI11::CLI11
        ${Boost_LIBRARIES}
        )

add_executable(bench
        src/benchmark.cc
        ${SOURCES}
        )
target_compile_features(bench PUBLIC cxx_std_17)
target_link_libraries(bench
        lbp_lib
        "opencv_core;opencv_imgcodecs;opencv_highgui"
        benchmark
        ${Boost_LIBRARIES}
        )

