from conans import ConanFile, CMake, tools
import os, subprocess, shutil
from glob import glob

# Build VTK
#   - Release mode doesnt work on Bizon AI computer
#   - Debug mode works on Bizon AI computer

class VtkConan(ConanFile):
    name = "sjsu_vtk"
    version = "9.2.6"
    requires = "gtest/1.11.0"
    license = "BSD-3-Clause"
    url = "https://vtk.org/"
    description = "VTK: Process images and create 3D computer graphics with the Visualization Toolkit."
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake", "cmake_find_package"
    options = {
        "shared": [True, False],
        "fPIC": [True, False]
    }
    default_options = {
        "shared": False,
        "fPIC": True
    }
    short_paths = True

    source_subdir = "VTK"
    build_subdir = "build"

    def source(self):
        self.run("git clone --recursive -b v9.2.6 git@github.com:Kitware/VTK.git")

    def build(self):
        cmake = CMake(self)
        cmake.definitions["VTK_BUILD_EXAMPLES"] = "ON"
        cmake.definitions["VTK_QT_VERSION:STRING"] = "5"
        cmake.definitions["VTK_Group_Qt:BOOL"] = "ON"
        cmake.definitions["QT_QMAKE_EXECUTABLE:PATH"] = os.path.join(os.path.expanduser("~"), "/Qt5.12.12/5.12.12/gcc_64/bin/qmake")
        cmake.definitions["CMAKE_PREFIX_PATH:PATH"] = os.path.join(os.path.expanduser("~"), "/Qt5.12.12/5.12.12/gcc_64/lib/cmake")
        cmake.definitions["VTK_GROUP_ENABLE_Qt-STRINGS"] = "YES"
        cmake.definitions["VTK_MODULE_ENABLE_VTK_GUISupportQt"] = "YES"
        cmake.configure(source_folder=self.source_subdir, build_folder=self.build_subdir)
        cmake.build()
        cmake.install()

    def package(self):
        self.copy(pattern="*.h", dst="include", src=self.source_subdir, keep_path=True)
        self.copy(pattern="*.dll", dst="bin", src=self.build_subdir, keep_path=False)
        self.copy(pattern="*.lib", dst="lib", src=self.build_subdir, keep_path=False)
        self.copy(pattern="*.a", dst="lib", src=self.build_subdir, keep_path=False)
        self.copy(pattern="*.so", dst="lib", src=self.build_subdir, keep_path=False)

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
