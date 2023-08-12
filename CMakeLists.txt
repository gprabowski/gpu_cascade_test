cmake_minimum_required(VERSION 3.22)
project(filter_test LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_BUILD_TYPE Release)

find_package(CUDAToolkit REQUIRED)


add_executable(filter_test
        cascade_test.cu)

set_property(TARGET filter_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(filter_test PROPERTIES CUDA_ARCHITECTURES 86)

target_include_directories(filter_test PUBLIC includes/)

target_link_libraries(filter_test CUDA::cufft CUDA::cudart)

add_custom_command(TARGET filter_test PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_SOURCE_DIR}/test.json $<TARGET_FILE_DIR:filter_test>/test.json)
