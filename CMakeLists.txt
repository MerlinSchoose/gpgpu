cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)

include(FetchContent)

FetchContent_Declare(
  GoogleBenchmark
  URL https://github.com/google/benchmark/archive/v1.4.1.tar.gz
  )

FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.8.0.tar.gz
  )

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.4.2.tar.gz
  )


if (NOT GoogleBenchmark_POPULATED)
  FetchContent_Populate(GoogleBenchmark)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
  add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")


add_library(gpu_lbp)
target_compile_features(gpu_lbp PUBLIC cxx_std_17)
target_link_libraries(gpu_lbp PRIVATE spdlog::spdlog "opencv_core;opencv_imgcodecs;opencv_highgui") 


target_sources(gpu_lbp PRIVATE include/lbp.hh src/gpu_lbp.cu)
set_target_properties(gpu_lbp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#add_library(cpu_ref SHARED IMPORTED)
#set_target_properties(render_cpu_ref PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/librender_cpu_ref.so)

set(SOURCES
  src/main.cc
  src/cpu_lbp.cc
  src/lbp.cc
  src/nearest_neighbour.cc
  src/train_kmeans.cc)

add_executable(gpgpu ${SOURCES})
target_compile_features(gpgpu PUBLIC cxx_std_17)
target_link_libraries(gpgpu PRIVATE "opencv_core;opencv_imgcodecs;opencv_highgui" CLI11::CLI11 spdlog::spdlog gpu_lbp)

#add_executable(bench src/bench.cpp)
#target_compile_features(bench PUBLIC cxx_std_17)
#target_link_libraries(bench PRIVATE render_cpu_ref spdlog::spdlog benchmark renderer)
