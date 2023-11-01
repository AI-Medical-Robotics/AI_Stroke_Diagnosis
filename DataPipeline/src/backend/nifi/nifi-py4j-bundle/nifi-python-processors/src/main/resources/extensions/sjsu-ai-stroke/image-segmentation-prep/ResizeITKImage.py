# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import json
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from py4j.java_collections import ListConverter

# TODO (JG): Make this work for training and testing sets
# Also referenced perplexity.ai (GPT-3) at url: 
class ResizeITKImage(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['SimpleITK==2.2.1', 'pandas==1.3.5', 'tqdm==4.66.1']
        description = 'Gets the NIfTI filepaths from the pandas csv dataframe in the flow file, loads each NIfTI file as an ITK voxel and performs SimpleITK resizing on each 3D NIfTI voxel'
        tags = ['sjsu_ms_ai', 'csv', 'itk', 'nifti']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.res_dir = PropertyDescriptor(
            name = 'ITK Resized Folder Path',
            description = 'The folder to stored the ITK Preprocessed Resized Images.',
            default_value="{}/src/datasets/NFBS_Dataset_NiFi/{}".format(os.path.expanduser("~"), "resized"),
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Resized',
            description = 'If ITK Resized Already Performed, then just get filepaths',
            default_value=False,
            required = True,
        )
        self.nifti_data_type = PropertyDescriptor(
            name = 'NifTI Dataset Name',
            description = 'The name of the NifTI Dataset, currently supported: {nfbs, atlas, icpsr_stroke}.',
            default_value = "nfbs",
            required=True
        )
        self.spacing_resolution = PropertyDescriptor(
            name = 'Resize Spacing Resolution (mm)',
            description = 'Control the spacing resolution for NIfTI voxels to resize to be large (< 2) or small (> 2)',
            default_value = "1.0",
            required=True
        )
        self.dimensions = PropertyDescriptor(
            name = 'Resize Target Dimensions',
            description = 'Set the resize target dimensions {width, height, depth} for NIfTI voxels',
            default_value = """{"width": 96, "height": 112, "depth": 48}""",
            required=False
        )
        self.expected_dim_keys = ['width', 'height', 'depth']
        self.descriptors = [self.res_dir, self.already_prepped, self.nifti_data_type, self.spacing_resolution, self.dimensions]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for res_dirpath, etc")
        self.raw_index = list()
        self.skull_mask_index = list()
        # read pre-trained model and config file
        self.res_dirpath = context.getProperty(self.res_dir.name).getValue()
        self.already_res = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.nifti_data_name = context.getProperty(self.nifti_data_type.name).getValue()
        self.resample_spacing_resolution = context.getProperty(self.spacing_resolution.name).asFloat()
        self.target_dims_json_str = context.getProperty(self.dimensions.name).getValue()

        if self.nifti_data_name == "icpsr_stroke":
            self.stroke_mask_index = list()

    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    def sitk_resample(self, input_image, new_resolution):
        self.logger.info("sitk_resample:")
        # Calculate new image size based on new voxel size (replaces target_shape)
            # multiply voxel size by voxel spacing divided by new resolution spacing size 
        new_size = [int(sz*spc/ns) for sz,spc,ns in zip(input_image.GetSize(), input_image.GetSpacing(), new_resolution)]
        self.logger.info("new_size len = {}".format(len(new_size)))
        self.logger.info("new_size list = {}".format(new_size))

        # Define the resampling filter
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_resolution)
        resampler.SetOutputOrigin(input_image.GetOrigin())
        resampler.SetOutputDirection(input_image.GetDirection())

        # resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        # resampler.SetTransform(new_affine)

        # Resample the image uisng ITK
        resampled_image = resampler.Execute(input_image)
        return resampled_image

    def sitk_save(self, output_image, voxel_name, idx):
        output_image_path = os.path.join(self.res_dirpath, self.nifti_data_name + "_" + voxel_name + str(idx) + ".nii.gz")

        # Write the resampled image using ITK
        sitk.WriteImage(output_image, output_image_path)      
        return output_image_path

    def itk_resize(self, nifti_csv_data):
        self.logger.info("Performing SimpleITK Resizing")
        self.mkdir_prep_dir(self.res_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.already_res type = {}".format(type(self.already_res)))
        if self.already_res:
            self.logger.info("Adding Prepped SimpleITK Resized filepaths to data df in raw_index & skull_mask_index")

            for i in tqdm(range(len(nifti_csv_data))):
                self.raw_index.append(self.res_dirpath + os.sep + self.nifti_data_name + "_" + "raw" + str(i) + ".nii.gz")
                
                self.skull_mask_index.append(self.res_dirpath + os.sep + self.nifti_data_name + "_skull_mask" + str(i) + ".nii.gz")

                if self.nifti_data_name == "icpsr_stroke":
                    output_stroke_mask_path = os.path.join(self.res_dirpath, self.nifti_data_name + "_lesion_mask" + str(i) + ".nii.gz")
                    self.stroke_mask_index.append(output_stroke_mask_path)

            self.logger.info("Retrieved Prepped Resized voxel filepaths stored at : {}/".format(self.res_dirpath))
        else:
            # TODO (JG): Could Add NiFi Properties to set these parameters
            target_dims_dict = json.loads(self.target_dims_json_str)
            for dim_key in self.expected_dim_keys:
                if dim_key not in target_dims_dict:
                    self.logger.error("The key '{}' is missing or incorrect in the JSON data.".format(dim_key))
            
            # target_shape = [target_dims_dict['width'], target_dims_dict['height'], target_dims_dict['depth']]
            # new_resolution = [2.0, 2.0, 2.0]
            # 2 is larger than 4

            # Defines new voxel size
            new_resolution = [self.resample_spacing_resolution,]*3
            self.logger.info("new_resolution len = {}".format(len(new_resolution)))
            self.logger.info("new_resolution list = {}".format(new_resolution))

            for i in tqdm(range(len(nifti_csv_data))):
                if self.nifti_data_name == "nfbs":
                    input_brain_image = sitk.ReadImage(nifti_csv_data.raw.iloc[i], sitk.sitkFloat32)
                    input_skull_mask = sitk.ReadImage(nifti_csv_data.brain_mask.iloc[i])
                elif self.nifti_data_name == "atlas":
                    # TODO (JG): For atlas 2.0, change input_skull_mask to input_stroke_mask
                    input_brain_image = sitk.ReadImage(nifti_csv_data.train_t1w_raw.iloc[i], sitk.sitkFloat32)
                    input_skull_mask = sitk.ReadImage(nifti_csv_data.train_lesion_mask.iloc[i])
                elif self.nifti_data_name == "icpsr_stroke":
                    input_brain_img = sitk.ReadImage(nifti_csv_data.brain_dwi_orig.iloc[i], sitk.sitkFloat32)
                    input_skull_mask = sitk.ReadImage(nifti_csv_data.brain_dwi_mask.iloc[i])
                    input_stroke_mask = sitk.ReadImage(nifti_csv_data.stroke_dwi_mask.iloc[i])

                    # Resample the input stroke mask 3D image using SimpleITK
                    resampled_stroke_mask = self.sitk_resample(input_stroke_mask, new_resolution)
                    output_stroke_mask_path = self.sitk_save(resampled_stroke_mask, "_lesion_mask", i)
                    self.stroke_mask_index.append(output_stroke_mask_path)

                # Resample the input brain 3D image using SimpleITK
                resampled_image = self.sitk_resample(input_brain_img, new_resolution)
                output_image_path = self.sitk_save(resampled_image, "raw", i)
                self.raw_index.append(output_image_path)

                # Resample the input skull mask 3D image using SimpleITK
                resampled_skull_mask = self.sitk_resample(input_skull_mask, new_resolution)
                output_skull_mask_path = self.sitk_save(resampled_skull_mask, "_skull_mask", i)
                self.skull_mask_index.append(output_skull_mask_path)
            self.logger.info("ITK Resized voxels stored at: {}/".format(self.res_dirpath))

        nifti_csv_data["raw_index"] = self.raw_index
        nifti_csv_data["skull_mask_index"] = self.skull_mask_index

        if self.nifti_data_name == "icpsr_stroke":
            nifti_csv_data["stroke_mask_index"] = self.stroke_mask_index
            return nifti_csv_data

        return nifti_csv_data

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        # NIfTI: — Neuroimaging Informatics Technology Initiative
        nifti_csv = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: nifti_csv = {}".format(nifti_csv.head()))
        nifti_csv_prepped = self.itk_resize(nifti_csv)
        self.logger.info("output: nifti_csv_prepped = {}".format(nifti_csv_prepped.head()))

        # Create a StringIO object
        nifti_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        nifti_csv_prepped.to_csv(nifti_string_io, index=False)

        # Get the string value and encode it
        nifti_csv_prepped_string = nifti_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = nifti_csv_prepped_string)
