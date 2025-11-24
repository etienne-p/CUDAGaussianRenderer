# CMake entry point.
cmake_minimum_required (VERSION 3.0)

# We need CUDA.
project (GaussianRender LANGUAGES CUDA CXX)
set(CMAKE_CUDA_COMPILER "nvcc")
find_package(CUDAToolkit REQUIRED)

# We need OpenGL.
find_package(OpenGL REQUIRED)

# Compile external dependencies.
add_subdirectory (external)

# On Visual 2005 and above, this module can set the debug working directory
# cmake_policy(SET CMP0026 OLD)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/rpavlik-cmake-modules-fe2273")
include(CreateLaunchers)
include(MSVCMultipleProcessCompile) # /MP

include_directories(
	external/glfw-3.1.2/include/
	external/glm-1.0.1/
	external/glew-1.13.0/include/
	.
)

set(ALL_LIBS
	${OPENGL_LIBRARY}
	glfw
	GLEW_1130
)

add_definitions(
	-DTW_STATIC
	-DTW_NO_LIB_PRAGMA
	-DTW_NO_DIRECT3D
	-DGLEW_STATIC
	-D_CRT_SECURE_NO_WARNINGS
)

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

set_target_properties(GaussianRender PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(GaussianRender PROPERTIES CUDA_ARCHITECTURES "75")

target_link_libraries(GaussianRender
	${ALL_LIBS}
)

# Fix GLM compiler warning (try -Xcudafe --display_error_number).
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe=\"--diag_suppress=2977\"")

# Add NVCC debug flags.
target_compile_options(GaussianRender PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-g;-O0>")
target_compile_options(GaussianRender PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3;--use_fast_math>")
