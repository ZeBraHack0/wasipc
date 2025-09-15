cmake_minimum_required(VERSION 3.20)
project(wasipc LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

find_package(CUDAToolkit QUIET)
if (NOT CUDAToolkit_FOUND)
  find_package(CUDA REQUIRED)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(wasipc STATIC src/wasipc.cpp)
target_include_directories(wasipc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(wasipc PUBLIC cxx_std_17)

set_target_properties(wasipc PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (TARGET CUDAToolkit::cudart)
  set(WASIPC_CUDART CUDAToolkit::cudart)
elseif (TARGET CUDA::cudart)
  set(WASIPC_CUDART CUDA::cudart)
else()
  message(FATAL_ERROR "cudart target not found; set CUDAToolkit_ROOT or install CUDA Toolkit.")
endif()

target_link_libraries(wasipc PUBLIC ${WASIPC_CUDART})

pybind11_add_module(pywasipc src/pywasipc.cpp)
target_link_libraries(pywasipc PRIVATE wasipc ${WASIPC_CUDART})
target_compile_features(pywasipc PRIVATE cxx_std_17)

include(GNUInstallDirs)
install(TARGETS pywasipc
  LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}
  RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}
  ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}
)
