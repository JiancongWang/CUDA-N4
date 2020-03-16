cmake_minimum_required(VERSION 3.1)
project( CUDAN4 )

# import library
SET(ITK_DIR "/home/jiancong/Desktop/CUDA_N4/ITK_build" CACHE PATH "ITK directory override" FORCE)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# CUDA version of LBGFS
# find_package(CudaLBFGS REQUIRED)
# include_directories(${CUDALBDFS_INCLUDE_DIRS})

include_directories(../common/inc/)

# NVCC config
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61; -lcufft;--std=c++11;)

# Compile cuda objects
CUDA_ADD_LIBRARY(cuda_obj STATIC 
                 reducer.cu
                 histogram.cu
                 cudaN4.cu
                 )

CUDA_ADD_CUBLAS_TO_TARGET(cuda_obj)

# Link FFTW to cuda object
target_link_libraries(cuda_obj fftw3)
target_link_libraries(cuda_obj fftw3f)

# Compiler config - make sure this comes after the nvcc compile to prevent interference to nvcc
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -lpthread")

add_executable(CUDAN4 main.cpp) # only used one c++ class for this test

# Link FFTW to main
target_link_libraries(CUDAN4 fftw3)
target_link_libraries(CUDAN4 fftw3f)

# Link ITK/VTK to main
target_link_libraries(CUDAN4 ${ITK_LIBRARIES} )

# Link CUDA to main
target_link_libraries(CUDAN4 ${CUDA_LIBRARIES})

# Link CUDA obj to main
target_link_libraries(CUDAN4 cuda_obj)





