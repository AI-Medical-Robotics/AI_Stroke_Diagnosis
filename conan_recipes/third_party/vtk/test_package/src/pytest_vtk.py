import vtk

def test_vtk_functionality():
    # Create a sphere
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(1.0)
    sphere_source.Update()

    # Check if the sphere has been created successfully
    assert sphere_source.Get_Output() is not None
    