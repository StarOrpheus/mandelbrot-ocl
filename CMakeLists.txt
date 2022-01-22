cmake_minimum_required(VERSION 3.21)
project(mandelbrot_ocl)

set(CMAKE_CXX_STANDARD 17)

find_path(VULKAN_HPP_INCLUDE_DIRS "vulkan/vulkan.hpp")
find_package(Vulkan REQUIRED FATAL_ERROR)
find_package(OpenCL REQUIRED)

add_executable(mandelbrot_ocl opencl_main.cpp)
target_link_libraries(mandelbrot_ocl OpenCL)

add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/kernel.spv"
        COMMAND $ENV{VULKAN_SDK}/bin/dxc -T cs_6_0 -E "main" -spirv -fspv-target-env=vulkan1.1 -Fo "${CMAKE_BINARY_DIR}/kernel.spv" "kernel.hlsl"
        DEPENDS "kernel.hlsl"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Buiding Shaders"
)
add_custom_target(ComputeGlslShader DEPENDS "${CMAKE_BINARY_DIR}/kernel.spv")

add_executable(mandelbrot_vulkan vulkan_main.cpp)
target_include_directories(mandelbrot_vulkan PRIVATE ${VULKAN_HPP_INCLUDE_DIRS})
target_link_libraries(mandelbrot_vulkan PRIVATE ${Vulkan_LIBRARIES})
add_dependencies(mandelbrot_vulkan ComputeGlslShader)
