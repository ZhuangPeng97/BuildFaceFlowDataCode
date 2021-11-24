cmake_minimum_required(VERSION 2.8)
project(CLOEST)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")


#find_package(CUDA REQUIRED)
#cuda_add_library(cuda_knn SHARED src/cuda_knn.cu)

#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
#include_directories("/usr/local/cuda-10.0/samples/common/inc")

# nanoflann
include_directories("./include")

# eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# src
include_directories("./src")
file(GLOB HEADERS "./src/*")
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} 
    OpenMeshCore OpenMeshTools
    #cuda_knn -lcuda -lcublas
)

execute_process(COMMAND ln -f -s ${CMAKE_SOURCE_DIR}/data WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
