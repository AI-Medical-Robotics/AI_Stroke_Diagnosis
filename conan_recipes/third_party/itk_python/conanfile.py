from conans import ConanFile, CMake, tools
import os, subprocess, shutil
from glob import glob
import sys

class ItkConan(ConanFile):
    name = "sjsu_itk"
    version = "5.3.0"
    requires = "gtest/1.11.0"
    license = "Apache 2.0 License"
    url = "https://github.com/InsightSoftwareConsortium/ITKPythonPackage"
    description = "A setup script to generate ITK Python Wheels"
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

    source_subdir = "ITK"
    build_subdir = "build"

    def source(self):
        self.run("git clone --depth 1 -b v5.3.0 git@github.com:InsightSoftwareConsortium/ITK.git")

    def build(self):
        cmake = CMake(self)
        cmake.definitions["ITK_BUILD_ALL_MODULES"] = "ON"
        cmake.definitions["ITK_WRAP_PYTHON:BOOL"] = "ON"
        cmake.definitions["PYTHON_EXECUTABLE"] = "{}".format(sys.executable)
        cmake.definitions["BUILD_TESTING"] = "ON"
        cmake.definitions["DISABLE_MODULE_TESTS"] = "OFF"
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
