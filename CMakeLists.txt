# CMake entry point.
cmake_minimum_required(VERSION 3.18)

# We need CUDA.
project(GaussianRender LANGUAGES CUDA CXX)

# cmake_policy: CMP0104 requires CUDA_ARCHITECTURES to be set on targets (enforced in 3.18+).
# CMP0072 prefers GLVND OpenGL libraries over legacy ones.
cmake_policy(SET CMP0072 NEW)

find_package(CUDAToolkit REQUIRED)

# We need OpenGL.
find_package(OpenGL REQUIRED)

# Compile external dependencies.
add_subdirectory(external)

# On Visual Studio, this module can set the debug working directory.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/rpavlik-cmake-modules-fe2273")
include(CreateLaunchers)
include(MSVCMultipleProcessCompile) # /MP

add_executable(GaussianRender
    src/Demo.cpp
    src/Consts.h
    src/Utilities.h
    src/DeviceBuffer.cu
    src/GaussianRender.cuh
    src/GaussianRender.cu
    src/CudaErrorCheck.cpp
    src/PlyParser.h
    src/PlyParser.cpp
    src/CameraControls.h
    src/CameraControls.cpp
)

# Use target-scoped include directories instead of global include_directories().
target_include_directories(GaussianRender PRIVATE
    external/glfw-3.1.2/include/
    external/glm-1.0.1/
    external/glew-1.13.0/include/
    .
)

# Use target-scoped compile definitions instead of global add_definitions().
target_compile_definitions(GaussianRender PRIVATE
    TW_STATIC
    TW_NO_LIB_PRAGMA
    TW_NO_DIRECT3D
    GLEW_STATIC
    _CRT_SECURE_NO_WARNINGS
)

set_target_properties(GaussianRender PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # CUDA_ARCHITECTURES replaces the deprecated set_target_properties + "75" string in 3.18+.
    # Use 75 for Turing (RTX 20xx/Quadro), add more as needed e.g. "75;86;89".
    CUDA_ARCHITECTURES 75
)

target_link_libraries(GaussianRender PRIVATE
    OpenGL::GL      # Modern IMPORTED target replacing ${OPENGL_LIBRARY}
    glfw
    GLEW_1130
    CUDA::cudart    # Explicit CUDA runtime linkage via CUDAToolkit
)

# Fix GLM compiler warning.
target_compile_options(GaussianRender PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=2977>
)

# NVCC debug flags.
target_compile_options(GaussianRender PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-g;-O0>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3;--use_fast_math>
)