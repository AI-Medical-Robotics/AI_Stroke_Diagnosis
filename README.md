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
