cmake_minimum_required(VERSION 3.28)
project(minitorchlib VERSION 0.0.0 LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)
add_subdirectory(tests)
target_include_directories(minitorchlib PUBLIC include)
