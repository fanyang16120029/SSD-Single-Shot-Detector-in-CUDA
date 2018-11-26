cmake_minimum_required(VERSION 2.6)
project(ssd_cuda)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(Main Main.h Main.cpp DNNLIB.h DNNLIB.cpp kernel.cu)  

# C++ list
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")

# CUDA list
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -use_fast_math)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)


