from conans import ConanFile, CMake, tools
import os, subprocess

shared_options = {
    "shared": [True, False],
    "fPIC": [True, False]
}

shared_default_options = {
    "shared": False,
    "fPIC": True
}

class SjsuAIStrokeDiagnosis(ConanFile):
    name = "sjsu_ai_stroke"
    version = "0.0.1"
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake", "cmake_paths", "cmake_find_package"
    options = shared_options
    default_options = shared_default_options

    def configure(self):
        self.options.shared = False

    def requirements(self):
        self.requires("gtest/1.11.0")
        self.requires("zlib/1.2.13")
        self.requires("openssl/3.1.1")
        self.requires("libpng/1.6.40")
        self.requires("libtiff/4.5.1")
        self.requires("libjpeg/9d")
        self.requires("openjpeg/2.5.0")
        self.requires("opencv/4.5.5")
        self.requires("itk/5.1.0")
        # self.requires("sjsu_vtk/9.2.6@debug/latest")
        # self.requires("cppzmq/4.10.0")
        # self.requires("protobuf/3.21.9")
        # self.requires("nlohmann_json/3.11.2")
        # self.requires("pybind11/2.10.4")
