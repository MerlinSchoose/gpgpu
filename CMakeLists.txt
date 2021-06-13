cmake_minimum_required(VERSION 3.17)
project(gpgpu)

set(CMAKE_CXX_STANDARD 17)

add_executable(gpgpu src/main.cc src/lbp.cc src/train_kmeans.cc)

find_package(OpenCV REQUIRED)
target_include_directories(gpgpu PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(gpgpu ${OpenCV_LIBS})
