cmake_minimum_required(VERSION 3.1.0)
project(Nearest_neighbor_search VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 11)
include(CTest)
enable_testing()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
include_directories(.)

add_executable(Nearest_neighbor_search main.cpp engine.h engine.cpp tests.h unit_tests.h)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
