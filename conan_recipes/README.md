# SJSU Conan Recipes

Create a new conan package

~~~bash
cd AI_Stroke_Diagnosis/conan_recipes/third_party/vtk

conan create . debug/latest --profile ../../../profiles/debug-linux
~~~

## Third Party

- SJSU_VTK: conan package built in debug mode
    - conan package built in release mode when running doesnt output expected results
