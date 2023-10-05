cmake_minimum_required(VERSION 3.0)
project(torchscatter)
set(CMAKE_CXX_STANDARD 14)
set(TORCHSCATTER_VERSION 2.1.2)

option(WITH_CUDA "Enable CUDA support" OFF)
option(WITH_PYTHON "Link to Python when building" ON)

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

if (WITH_PYTHON)
  add_definitions(-DWITH_PYTHON)
  find_package(Python3 COMPONENTS Development)
endif()
find_package(Torch REQUIRED)

file(GLOB HEADERS csrc/*.h)
file(GLOB OPERATOR_SOURCES csrc/cpu/*.h csrc/cpu/*.cpp csrc/*.cpp)
if(WITH_CUDA)
  file(GLOB OPERATOR_SOURCES ${OPERATOR_SOURCES} csrc/cuda/*.h csrc/cuda/*.cu)
endif()

add_library(${PROJECT_NAME} SHARED ${OPERATOR_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
if (WITH_PYTHON)
  target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME TorchScatter)

target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${HEADERS}>"
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(TORCHSCATTER_CMAKECONFIG_INSTALL_DIR "share/cmake/TorchScatter" CACHE STRING "install path for TorchScatterConfig.cmake")

configure_package_config_file(cmake/TorchScatterConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/TorchScatterConfig.cmake"
  INSTALL_DESTINATION ${TORCHSCATTER_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/TorchScatterConfigVersion.cmake
  VERSION ${TORCHSCATTER_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TorchScatterConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/TorchScatterConfigVersion.cmake
  DESTINATION ${TORCHSCATTER_CMAKECONFIG_INSTALL_DIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT TorchScatterTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(EXPORT TorchScatterTargets
  NAMESPACE TorchScatter::
  DESTINATION ${TORCHSCATTER_CMAKECONFIG_INSTALL_DIR})

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
install(FILES
  csrc/cpu/scatter_cpu.h
  csrc/cpu/segment_coo_cpu.h
  csrc/cpu/segment_csr_cpu.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cpu)
if(WITH_CUDA)
  install(FILES
    csrc/cuda/scatter_cuda.h
    csrc/cuda/segment_coo_cuda.h
    csrc/cuda/segment_csr_cuda.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cuda)
endif()

if(WITH_CUDA)
  set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
  set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
endif()
