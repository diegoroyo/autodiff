cmake_minimum_required(VERSION 3.10)

project(autodiff VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DAUTODIFF_DEBUG")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/ext/libcpp-common/include)
include_directories(${CMAKE_SOURCE_DIR}/ext/libcpp-common/src)

add_subdirectory(ext/libcpp-common/)

# Examples

# file(GLOB_RECURSE libautodiff-SRC "src/*.cpp")

add_executable(karpathy "examples/karpathy.cpp")
target_link_libraries(karpathy PRIVATE libcpp-common)

add_executable(matrixvector "examples/matrixvector.cpp")
target_link_libraries(matrixvector PRIVATE libcpp-common)

add_executable(andgate "examples/andgate.cpp")
target_link_libraries(andgate PRIVATE libcpp-common)

add_executable(nerf "examples/nerf.cpp")
target_link_libraries(nerf PRIVATE libcpp-common)