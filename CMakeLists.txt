cmake_minimum_required(VERSION 3.1.0)
project(Nearest_neighbor_search VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 11)
include(CTest)
enable_testing()

include_directories(.)

add_executable(Nearest_neighbor_search main.cpp engine.h engine.cpp tests.h unit_tests.h)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
