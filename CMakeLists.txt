#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from NVIDIA CORPORATION is strictly prohibited.
#

cmake_minimum_required(VERSION 3.18)

project(BiCGStab_sample LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

# Check if we can use --threads for faster compile
if(${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11)
  if(${CUDAToolkit_VERSION_MINOR} GREATER_EQUAL 2)
    set(CMAKE_CUDA_FLAGS "--threads=0")
  endif()
endif()

if(CMAKE_CUDA_ARCHITECTURES LESS 60)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(${PROJECT_NAME} main.cu mmio.c)

target_compile_options(
  ${PROJECT_NAME}
  PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")

target_include_directories(${PROJECT_NAME}
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart CUDA::cusparse
                                             CUDA::cublas)
