cmake_minimum_required(VERSION 3.15)

# if(UNIX)
#     find_package(OpenCV 4)
# endif()

find_package(VTK 9 REQUIRED)
# include(${VTK_USE_FILE})

# add_subdirectory(lib)
add_subdirectory(app)
