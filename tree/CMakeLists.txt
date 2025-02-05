cmake_minimum_required(VERSION 3.10)
project(treeCXX)
# Setup flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra  -march=native -O2 -g")

find_package(OpenMP)
if (OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Include directories and libraries
include_directories(include
    ${PROJECT_SOURCE_DIR}/../puffinn/include
    ${PROJECT_SOURCE_DIR}/../puffinn/test/include
)
add_library(puffinn
    SHARED
    ${PROJECT_SOURCE_DIR}/../puffinn/include/puffinn.hpp
    ${PROJECT_SOURCE_DIR}/../puffinn/test/include/catch.hpp
)

# Executables
add_executable(test source/test.cpp)


if (OpenMP_FOUND)
    target_link_libraries(test OpenMP::OpenMP_CXX)
endif()

set_target_properties(puffinn PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test puffinn)


