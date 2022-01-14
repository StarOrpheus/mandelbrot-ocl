cmake_minimum_required(VERSION 3.21)
project(mandelbrot_ocl)

set(CMAKE_CXX_STANDARD 17)

add_executable(mandelbrot_ocl main.cpp)
target_link_libraries(mandelbrot_ocl OpenCL)