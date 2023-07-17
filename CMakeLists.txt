cmake_minimum_required(VERSION 3.15)

project(ai_stroke_diagnosis)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add_compile_options(-fPIC)
# add_link_options("-Wl,--no-undefined")

# ENABLE_TESTING()

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)
include_directories(${CMAKE_BINARY_DIR})

# set(SRCS
#     main.cpp
# )

# add_executable(${PROJECT_NAME} ${SRCS})

add_subdirectory(DataExploration)
# add_subdirectory(test)
