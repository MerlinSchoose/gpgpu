cmake_minimum_required(VERSION 3.17)
project(CUDA_CMAKE LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

add_library(CudaLib SHARED src/library.cu)
target_compile_features(CudaLib PUBLIC cxx_std_17)
set_target_properties(CudaLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(CudaLib)

add_executable(gpgpu
    src/main.cc
    src/lbp.cc
    src/train_kmeans.cc
    src/nearest_neighbour.cc)

# target_include_directories(gpgpu PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(gpgpu
    CudaLib
    "opencv_core;opencv_imgcodecs;opencv_highgui")
