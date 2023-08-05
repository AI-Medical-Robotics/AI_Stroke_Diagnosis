import os
import vtk
import itk
import itkwidgets
import pandas as pd
import numpy as np
from tqdm import tqdm

def generate_itk_3d_image():
    # Define image properties
    image_type = itk.Image[itk.UC, 3]
    size = itk.Size[3]()
    size.Fill(256)
    origin = itk.Point[itk.D, 3]()
    origin.Fill(0.0)
    spacing = itk.Vector[itk.D, 3]()
    spacing.Fill(1.0)

    # Create the ITK image
    image = itk.Image[itk.UC, 3].New()
    image.SetRegions(itk.ImageRegion[3](size))
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.Allocate()

    # Set pixel values
    image.FillBuffer(255)

    return image

def itk_load_nifti(nifti_path):
    # define pixel_type, dimension and image_type
    pixel_type = itk.F
    dimension = 3
    image_type = itk.Image[pixel_type, dimension]

    # # Load MRI data using ITK
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(nifti_path)
    reader.Update()

    # Get the loaded ITK imaage
    itk_image = reader.GetOutput()
    itk_array = itk.GetArrayViewFromImage(itk_image)

    print("Type = {}".format(type(itk_image)))

    # Print the shape of the image
    size = itk_image.GetLargestPossibleRegion().GetSize()
    print("Brain Mask Shape of voxel = ", size[0], size[1], size[2])
    return itk_image, itk_array

def itk_extract_nifti_slice(itk_image, slice_index):
    size = itk_image.GetLargestPossibleRegion().GetSize()
    print("size[2] = {}".format(size[2]))
    size[2] = slice_index

    index = itk_image.GetLargestPossibleRegion().GetIndex()
    print("index[2] = {}".format(index[2]))
    index[2] = slice_index

    region = itk.ImageRegion[3]()
    region.SetSize(size)
    region.SetIndex(index)

    itk_slice = itk.extract_image_filter(itk_image, region)
    return itk_slice

# Convert ITK to VTK image without vtk.util and ImageToVTKImageFilter
def itk_to_vtk_3d_image(image):
    # GEt the image properties
    size = image.GetLargestPossibleRegion().GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()

    # Create a VTK image data object
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(size[0], size[1], size[2])
    vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
    vtk_image.SetOrigin(origin[0], origin[1], origin[2])

    # Get the ITK image data array
    itk_array = itk.GetArrayViewFromImage(image)

    # Convert the ITK array to the appropriate buffer type expected by VTK
    # vtk_buffer = np.array(itk_array, dtype=np.uint8)

    # Convert the ITK array to a VTK array
    vtk_array = vtk.vtkUnsignedCharArray()
    vtk_array.SetNumberOfComponents(1)
    vtk_array.SetArray(itk_array.ravel(), itk_array.size, 1)

    # Set the VTK array as the scalar data for the VTK image
    vtk_image.GetPointData().SetScalars(vtk_array)

    return vtk_image

# Convert ITK to VTK image slice without vtk.util and ImageToVTKImageFilter
def itk_to_vtk_slice(itk_slice):
    # Get the image properties
    itk_spacing = itk_slice.GetSpacing()
    itk_origin = itk_slice.GetOrigin()

    # Get the ITK image data array
    itk_array = itk.GetArrayViewFromImage(itk_slice)
    itk_size = itk_array.shape

    # Create a VTK image data object
    vtk_slice = vtk.vtkImageData()
    vtk_slice.SetDimensions(itk_size[::-1])
    vtk_slice.SetSpacing(itk_spacing[::-1])
    vtk_slice.SetOrigin(itk_origin[::-1])

    # Convert the ITK array to the appropriate buffer type expected by VTK
    # vtk_buffer = np.array(itk_array, dtype=np.uint8)

    # Convert the ITK array to a VTK array
    vtk_array = vtk.vtkUnsignedCharArray()
    vtk_array.SetNumberOfComponents(1)
    vtk_array.SetArray(itk_array.ravel(), itk_array.size, 1)

    # Set the VTK array as the scalar data for the VTK image
    vtk_slice.GetPointData().SetScalars(vtk_array)

    return vtk_slice

def visualize_vtk_3d_image(vtk_image):
    # Create a VTK renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create a VTK volume mapper and set the input data
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_image)

    # Create a VTK volume and set the mapper
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)

    # Add the volume to the renderer
    renderer.AddVolume(volume)

    # Set the background color to a lighter shade
    renderer.SetBackground(1.0, 1.0, 1.0) # White

    # Create a VTK interactor and set the render window
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Start the interactor
    interactor.Start()

def visualize_vtk_slice(vtk_slice):
    # Create a VTK renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create a VTK volume actor and set the input vtk slice
    image_actor = vtk.vtkImageActor()
    image_actor.SetInputData(vtk_slice)

    # Add the actor to the renderer
    renderer.AddActor(image_actor)

    # Set the background color to a lighter shade
    renderer.SetBackground(1.0, 1.0, 1.0) # White

    # Create a VTK interactor and set the render window
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Start the interactor
    interactor.Start()

def visualize_vtk(nifti_df, index):
    # Load MRI data using ITK
    itk_nifti_img1 = itk_load_nifti(nifti_df.raw.iloc[index])

    # Convert ITK to VTK image
    vtk_nifti_img1 = itk_to_vtk_3d_image(itk_nifti_img1)

    # Create a VTK renderer and render window

    # Create a VTK image viewer

    # Set the titles for the MRI voxels

    # Render and display the MRI voxel

    # Start the interactor



def validate_itk_to_vtk_pixels():
    # Generate ITK image
    itk_image = generate_itk_3d_image()

    # Convert ITK image to VTK image
    vtk_image = itk_to_vtk_3d_image(itk_image)

    # vtk_array_np = vtk_to_numpy(vtk_array)

    # Verify equality
    itk_array = itk.GetArrayViewFromImage(itk_image)

    vtk_array = vtk_image.GetPointData().GetScalars()

    # Get the dimensions of the images
    size = itk_image.GetLargestPossibleRegion().GetSize()

    # Compare the arrays pixel by pixel
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                itk_pixel = itk_array[i, j, k]
                vtk_pixel = vtk_array.GetTuple1(vtk_image.ComputePointId([i, j, k]))
                if itk_pixel != vtk_pixel:
                    print("Pixel values are not equal at index:", (i, j, k))
                    break

    print("Pixel-wise comparison completed")

def test_itk_load_nifti():
    nfbs_path = os.path.join(os.path.expanduser("~"), "Desktop", "NFBS_Dataset")
    nfbs_brain_path = "{}/A00028185/sub-A00028185_ses-NFB3_T1w.nii.gz".format(nfbs_path)
    itk_image, itk_array = itk_load_nifti(nfbs_brain_path)

    # NOTE: Visualize the image using itkwidgets; Must be used in Jupyter
    itkwidgets.view(itk_image)

def test_visualize_vtk_image():
    # itk_image = generate_itk_3d_image()
    nfbs_path = os.path.join(os.path.expanduser("~"), "Desktop", "NFBS_Dataset")
    nfbs_brain_path = "{}/A00028185/sub-A00028185_ses-NFB3_T1w.nii.gz".format(nfbs_path)
    itk_image, itk_array = itk_load_nifti(nfbs_brain_path)
    vtk_image = itk_to_vtk_3d_image(itk_image)  
    visualize_vtk_3d_image(vtk_image)  

def test_visualize_vtk_slice():
    nfbs_path = os.path.join(os.path.expanduser("~"), "Desktop", "NFBS_Dataset")
    nfbs_brain_path = "{}/A00028185/sub-A00028185_ses-NFB3_T1w.nii.gz".format(nfbs_path)
    itk_image, itk_array = itk_load_nifti(nfbs_brain_path)
    itk_slice = itk_extract_nifti_slice(itk_image, slice_index=80)
    vtk_slice = itk_to_vtk_slice(itk_slice)
    visualize_vtk_slice(vtk_slice)

# test_itk_load_nifti()

# test_visualize_vtk_3d_image()

# test_visualize_vtk_slice()
