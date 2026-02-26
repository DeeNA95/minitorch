cmake_minimum_required(VERSION 3.28)
project(
  minitorchlib
  VERSION 0.0.0
  LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)
add_subdirectory(tests)

# 1. Include CMake's built-in package helper module
include(CMakePackageConfigHelpers)

# 1. Tell CMake to install the actual exported target definitions into the
#   standard cmake folder
install(
  EXPORT minitorchlibtargets
  FILE minitorchlibTargets.cmake
  NAMESPACE minitorchlib::
  DESTINATION lib/cmake/minitorchlib)

# 1. Create a basic minimal config file on-the-fly inside the build folder
file(
  WRITE ${CMAKE_CURRENT_BINARY_DIR}/minitorchlibConfig.cmake.in
  "@PACKAGE_INIT@\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/minitorchlibTargets.cmake\")\n"
)

# 1. Configure that file to be ready for installation
configure_package_config_file(
  ${CMAKE_CURRENT_BINARY_DIR}/minitorchlibConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/minitorchlibConfig.cmake
  INSTALL_DESTINATION lib/cmake/minitorchlib)

# 1. Tell CMake to actually install the config file we just created
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/minitorchlibConfig.cmake
        DESTINATION lib/cmake/minitorchlib)
