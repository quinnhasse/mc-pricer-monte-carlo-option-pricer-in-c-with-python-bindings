cmake_minimum_required(VERSION 3.20)
project(mc_pricer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# OpenMP
find_package(OpenMP)

# pybind11 — locate via pip-installed package
execute_process(
  COMMAND python3 -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# Catch2
find_package(Catch2 3 REQUIRED)

# Google Benchmark
find_package(benchmark REQUIRED)

# Header-only library target
add_library(mc_headers INTERFACE)
target_include_directories(mc_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mc_headers INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(mc_headers INTERFACE MC_OPENMP_ENABLED)
endif()

# Python extension module
pybind11_add_module(mc_pricer src/bindings.cpp)
target_link_libraries(mc_pricer PRIVATE mc_headers)

# Tests
enable_testing()
add_subdirectory(tests)

# Benchmarks
add_subdirectory(benchmarks)

# Install the Python module next to python/ scripts
install(TARGETS mc_pricer DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/python)
