# AI Stroke Diagnosis

SJSU Master Thesis on AI Medical Imaging for Stroke Diagnosis

For building 3D slicer, here are the commands:

~~~bash
cd ~/src
git clone https://github.com/Slicer/Slicer.git --recursive
pushd Slicer
git checkout v5.2.2
popd
mkdir Slicer522-SuperBuild-debug
~/src/open_source/Slicer522-SuperBuild-debug

# Note: 3D Slicer may complain about QT5 not being found, so install it first

~~~

For building VTK C++ lib and VTK python wheel file:

~~~bash
cd VTK
mkdir build
cd build
cmake .. -D VTK_WRAP_PYTHON=ON -D VTK_WHEEL_BUILD=ON -D PYTHON_EXECUTABLE=$(which python) -D VTK_QT_VERSION:STRING="5" -D VTK_Group_Qt:BOOL=ON -D QT_QMAKE_EXECUTABLE:PATH="~/Qt5.12.12/5.12.12/gcc_64/bin/qmake" -D CMAKE_PREFIX_PATH:PATH="~/Qt5.12.12/5.12.12/gcc_64/lib/cmake/" -D VTK_GROUP_ENABLE_Qt-STRINGS=YES -D VTK_MODULE_ENABLE_VTK_GUISupportQt=YES

make -j $(nproc)

python setup.py bdist_wheel

pip install dist/vtk-9.2.20230723.dev0-cp38-cp38-linux_x86_64.whl
~~~

For building ITK C++ lib and ITK python wheel file:

~~~bash
cd ITK
mkdir build
cd build
# Update CMAKE_INSTALL_PREFIX, so later when creating python wheel file, it'll refer to $HOME
cmake .. -D ITK_BUILD_ALL_MODULES=ON \
-D ITK_WRAP_PYTHON:BOOL=ON \
-D PYTHON_EXECUTABLE=$(which python) \
-D BUILD_TESTING=ON \
-D DISABLE_MODULE_TESTS=OFF \
-D CMAKE_INSTALL_PREFIX=$HOME

make -j $(nproc)

cd ../../ITKPythonPackage

# modified manylinux-build-wheels.sh to have cmake args similar to itk above; builds wheel for each py version
./scripts/dockcross-manylinux-build-wheels.sh

# mkdir build
# cd build
# cmake .. -D ITK_SOURCE_DIR=../../ITK -D ITK_BINARY_DIR="../../ITK/build/" -D ITKPythonPackage_BUILD_PYTHON=ON -D ITKPythonPackage_USE_TBB=OFF
# make -j $(nproc)

# mv ../setup.py ../setup.py.bk01

# ./scripts/dockcross-manylinux-build-wheels.sh

# pushd ../scripts


# Configure setup.py for custom itk
# SETUP_PY_CONFIGURE="../scripts/setup_py_configure.py"
# python setup_py_configure.py "itk" --output-dir="../"

# popd

# cd ../

# Generate setup.py for custom itk
# python setup.py bdist_wheel

# Enter sudo
# sudo su
# /home/ubuntu/miniconda3/condabin/conda init bash
# close and reopen shell
# conda activate stroke_ai

# ITK_REPO_SOURCE_DIR=/home/ubuntu/src/open_source/ITK
# ITK_REPO_BINARY_DIR=$ITK_REPO_SOURCE_DIR/build


# Complex generate setup.py custom itk
# python setup.py bdist_wheel build -G 'Unix Makefiles' -- \
#     -D ITK_SOURCE_DIR=${ITK_REPO_SOURCE_DIR} \
#     -D ITK_BINARY_DIR=${ITK_REPO_BINARY_DIR} \
#     -D ITKPythonPackage_ITK_BINARY_REUSE:BOOL=ON \
#     -D ITKPythonPackage_WHEEL_NAME:STRING="itk" \
#     -D CMAKE_BUILD_TYPE:STRING="Release" \
#     -D ITKPythonPackage_USE_TBB=OFF \
#     -D Module_ITKTBB:BOOL=OFF \
#     -D ITK_WRAP_DOC:BOOL=ON

~~~

Here is my plan:

- Client - Publisher: Simulate Stroke MRI Scanner in Unity where we load the MRI images while running the scanner simulator and then send those images over Zmq using C# Zmq Publisher.
    - First work on a simple Python Zmq publisher to send stroke MRIs to 3D Slicer
    - Second convert Python Zmq publisher to C++.

- Server - Subscriber: "This an AI Stroke Diagnosis 3D Slicer extension using SuperBuild to build a project comprised of multiple modules that doctors and radiologists can leverage to load stroke data over Zmq using Python/C++ Zmq Subscriber, display stroke data dasbhoard analytics with ITK/VTK, display stroke lesion segmentation performed with PyTorch/CUDA's custom UNet model and display stroke medical image captioning performed with PyTorch/CUDA's custom CapGAN (or Transformer, LLM) model."
    - Initially work on simple Python Zmq subscriber in 3D Slicer custom extension.

## Contributors

James Guzman (SJSU MS AI, Medical Imaging), Dr. Magdalini Eirinaki (SJSU MS AI Director & Project Advisor)

## 3D Slicer Extension Name

SlicerAIStrokeDiagnosis

## 3D Slicer Extension Category

SjsuAIStrokeDiagnosis
