cmake_minimum_required(VERSION 3.8)

project(stencil LANGUAGES CXX CUDA)
include_directories(include)

add_library(dacelib SHARED src/kernels_dace.cu)
add_library(optimizedlib SHARED src/kernels_optimized.cu)

add_executable(dace src/main.cu)
add_executable(optimized src/main.cu)

target_link_libraries(dace dacelib)
target_link_libraries(optimized optimizedlib)
