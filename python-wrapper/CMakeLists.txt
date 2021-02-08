cmake_minimum_required(VERSION 3.1.0)
project(Nearest_neighbor_search VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 11)
include(CTest)
enable_testing()

find_package(PythonInterp REQUIRED)
include_directories(. ${PYTHON_INCLUDE_DIRS} ./.. build)

add_subdirectory(pybind11)
pybind11_add_module(enginewrapper build/enginewrapper.cpp ../engine.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


