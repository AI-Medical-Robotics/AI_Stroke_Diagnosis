cmake_minimum_required(VERSION 3.15)

project(test_mri_exploration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

find_package(VTK 9 REQUIRED)
find_package(GTest)

set(SRCS
    TestMRIDataLoader.cpp
)

ENABLE_TESTING()

add_executable(${PROJECT_NAME} ${SRCS})

message("VTK_LIBRARIES = ${VTK_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} 
    ${VTK_LIBRARIES} 
    CONAN_PKG::gtest
)

# message("SJSU_VTK_INCLUDE_DIRS = ${SJSU_VTK_INCLUDE_DIRS}")
message("CONAN_INCLUDE_DIRS_SJSU_VTK = ${CONAN_INCLUDE_DIRS_SJSU_VTK}")
# message("GTest_INCLUDE_DIRS = ${GTest_INCLUDE_DIRS}")
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CONAN_INCLUDE_DIRS_GTEST}
    ${CONAN_INCLUDE_DIRS_SJSU_VTK}
    ${CONAN_INCLUDE_DIRS_SJSU_VTK}/vtk-9.2/
    ${CONAN_INCLUDE_DIRS_SJSU_VTK}/Common/Core/
)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
