cmake_minimum_required(VERSION 3.17)
project(CUDA_CMAKE LANGUAGES CXX CUDA)
include(FetchContent)

FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.8.0.tar.gz
  )

FetchContent_MakeAvailable(CLI11)

find_package(OpenCV REQUIRED)

add_library(CudaLib SHARED src/gpu_lbp.cu)
target_compile_features(CudaLib PUBLIC cxx_std_17)
set_target_properties(CudaLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(CudaLib)

add_executable(gpgpu
    src/main.cc
    src/lbp.cc
    src/train_kmeans.cc
    src/nearest_neighbour.cc
    src/cpu_lbp.cc)

# target_include_directories(gpgpu PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(gpgpu
    CudaLib
    "opencv_core;opencv_imgcodecs;opencv_highgui"
    CLI11::CLI11)
