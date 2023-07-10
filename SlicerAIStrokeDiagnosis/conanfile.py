from conans import ConanFile, CMake, tools
import os, subprocess

shared_options = {
    "shared": [True, False]
}

shared_default_options = {
    "shared": False
}

class SlicerAIStrokeDiagnosis(ConanFile):
    name = "slicer_ai_stroke"
    version = "0.0.1"
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake", "cmake_paths", "cmake_find_package"
    options = shared_options
    default_options = shared_default_options

    def configure(self):
        self.options.shared = False

    def requirements(self):
        # self.requires("grpc/1.50.1")
        self.requires("cppzmq/4.10.0")
        self.requires("protobuf/3.21.9")
        self.requires("nlohmann_json/3.11.2")
        # self.requires("pybind11/2.10.4")

    


