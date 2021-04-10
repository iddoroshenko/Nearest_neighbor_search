 cmake_minimum_required(VERSION 3.4...3.18)
 project(cmake_example)

 include_directories(. ${PYTHON_INCLUDE_DIRS} ./.. build)
 add_subdirectory(pybind11)
 pybind11_add_module(cmake_example build/enginewrapper.cpp ../engine.cpp)

 # EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
 # define (VERSION_INFO) here.
 target_compile_definitions(cmake_example PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})