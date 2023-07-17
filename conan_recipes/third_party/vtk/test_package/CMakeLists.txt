cmake_minimum_required(VERSION 3.15)
project(vtk_unit_tests CXX)
set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_BINARY_DIR}/generators/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

find_package(VTK REQUIRED)
# include(${VTK_USE_FILE})

set(SRCS
    src/test_vtk.cpp
)

add_executable(${PROJECT_NAME} ${SRCS})

message("VTK_LIBRARIES = ${VTK_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} PUBLIC
    CONAN_PKG::gtest
    ${VTK_LIBRARIES}
)

# message("VTK_INCLUDE_DIRS = ${VTK_INCLUDE_DIRS}")
message("CONAN_INCLUDE_DIRS_SJSU_VTK = ${CONAN_INCLUDE_DIRS_SJSU_VTK}")
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CONAN_INCLUDE_DIRS}
    ${CONAN_INCLUDE_DIRS_SJSU_VTK}
    ${CONAN_INCLUDE_DIRS_SJSU_VTK}/vtk-9.2/
    ${CONAN_INCLUDE_DIRS_SJSU_VTK}/Common/Core/
)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
