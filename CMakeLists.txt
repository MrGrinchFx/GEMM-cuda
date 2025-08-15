cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
cmake_policy(SET CMP0146 NEW)

project(cuda-gemm LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17) # Or 20, 23 etc., depending on your project's needs
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE MAIN_SOURCES
  main.cu
  GEMM.cu
  GEMM_kernels.cu
)
# --- Create Main Executable Target ---
# This executable will contain all the core application logic.
add_executable(${PROJECT_NAME} ${MAIN_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${COMMON_INCLUDE_DIRS})
# Link against the CUDA runtime library, which is essential for CUDA code.
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})
set(CUDA_ARCH_CODE "86")

add_executable(matrix_generator matrix_generator.cu)

set_property(TARGET ${PROJECT_NAME} cuda-gemm PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_CODE})
set_property(TARGET matrix_generator cuda-gemm PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_CODE})
