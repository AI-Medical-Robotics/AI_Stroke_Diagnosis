cmake_minimum_required(VERSION 3.15)
project(itk_unit_tests CXX)
set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_BINARY_DIR}/generators/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

find_package(ITK REQUIRED)

set(SRCS
    src/test_itk.cpp
)

add_executable(${PROJECT_NAME} ${SRCS})

message("ITK_LIBRARIES = ${ITK_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} PUBLIC
    CONAN_PKG::gtest
    ${ITK_LIBRARIES}
)

# message("VTK_INCLUDE_DIRS = ${VTK_INCLUDE_DIRS}")
message("CONAN_INCLUDE_DIRS_SJSU_ITK = ${CONAN_INCLUDE_DIRS_SJSU_ITK}")
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CONAN_INCLUDE_DIRS}
    ${CONAN_INCLUDE_DIRS_SJSU_ITK}
    ${CONAN_INCLUDE_DIRS_SJSU_ITK}/itk-5.3/
)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
