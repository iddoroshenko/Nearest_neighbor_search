cmake_minimum_required(VERSION 3.1.0)
project(revenge_project VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 11)
include(CTest)
enable_testing()

add_executable(revenge_project main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
