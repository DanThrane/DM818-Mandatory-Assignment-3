cmake_minimum_required(VERSION 2.8)
project(dm818_dns)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES comm.cpp comm.h main.cpp matrix.cpp matrix.h matrix_mul_imada.cpp matrix_mul.h)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_PATH})

add_executable(dnsmat ${SOURCE_FILES})
target_link_libraries(dnsmat ${MPI_LIBRARIES} ${BLASS_LIBRARIES})

if(MPI_COMPILE_FLAGS)
    set_target_properties(dnsmat PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(dnsmat PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()


