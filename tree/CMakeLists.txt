cmake_minimum_required(VERSION 3.10)
project(tree CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra  -march=native -O2 -g")

find_package(OpenMP)
if (OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories("include")

add_executable(test source/test.cpp)

if (OpenMP_FOUND)
    target_link_libraries(test OpenMP::OpenMP_CXX)
endif()


