cmake_minimum_required(VERSION 3.15)

project(mri_eda_app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SRCS
    src/main.cpp
)

add_executable(${PROJECT_NAME} ${SRCS})

target_link_libraries(${PROJECT_NAME}
    # mri_exploration
    # CONAN_PKG::opencv
    # CONAN_PKG::sjsu_vtk
    ${VTK_LIBRARIES}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    # ${CONAN_INCLUDE_DIRS_OPENCV}
    ${CONAN_INCLUDE_DIRS_SJSU_VTK}
    ${CONAN_INCLUDE_DIRS_SJSU_VTK}/vtk-9.2/
    ${CONAN_INCLUDE_DIRS_SJSU_VTK}/Common/Core/
)
