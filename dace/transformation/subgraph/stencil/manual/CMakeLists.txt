project(stencil)
include_directories(include)

add_library(dacelib SHARED kernels_dace.cpp)
add_library(optimizedlib SHARED kernels_optimiized.cpp)

add_executable(dace src/main.cpp)
add_executable(optimized src/main.cpp)

target_link_libraries(dace dacelib)
target_link_libraries(optimized optimizedlib)
