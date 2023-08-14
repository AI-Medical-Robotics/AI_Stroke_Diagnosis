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
# import itk
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from py4j.java_collections import ListConverter

# TODO (JG): Update to use SimpleITK like we did for Bias Field Correction

# TODO (JG): Make this work for training and testing sets
class ResizeCropITKImage(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['SimpleITK', 'pandas', 'tqdm']
        description = 'Gets the corrected bias field NIfTI filepaths from the pandas csv dataframe in the flow file, loads each NIfTI file as an ITK voxel and performs SimpleITK resizing and cropping on each 3D NIfTI voxel'
        tags = ['sjsu_ms_ai', 'csv', 'itk', 'nifti']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.rescrop_dir = PropertyDescriptor(
            name = 'ITK Resized Cropped Folder Path',
            description = 'The folder to stored the ITK Preprocessed Resized & Cropped Images.',
            default_value="{}/src/datasets/NFBS_Dataset_NiFi/{}".format(os.path.expanduser("~"), "resized_cropped"),
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Resized Cropped',
            description = 'If ITK Resized & Cropped Already Performed, then just get filepaths',
            default_value=False,
            required = True,
        )
        self.nifti_data_type = PropertyDescriptor(
            name = 'NifTI Dataset Name',
            description = 'The name of the NifTI Dataset, currently supported: {nfbs, atlas, icpsr_stroke}.',
            default_value = "nfbs",
            required=True
        )
        self.descriptors = [self.rescrop_dir, self.already_prepped, self.nifti_data_type]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for bias_corrected_dirpath, etc")
        self.raw_index = list()
        self.mask_index = list()
        # read pre-trained model and config file
        self.rescrop_dirpath = context.getProperty(self.rescrop_dir.name).getValue()
        self.already_rescropped = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.nifti_data_name = context.getProperty(self.nifti_data_type.name).getValue()

    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    def sitk_resample_crop(self, input_image, new_affine, target_shape, new_resolution, is_binary_mask=False):
        # Resample and crop the image uisng ITK
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(input_image)
        resampler.SetOutputSpacing(new_resolution)
        resampler.SetSize(target_shape)
        resampler.SetOutputOrigin(sitk.VectorDouble(new_affine.GetTranslation()))
        resampler.SetOutputDirection(new_affine.GetMatrix())
        if is_binary_mask:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetTransform(new_affine)
        resampled_image = resampler.Execute(input_image)
        return resampled_image

    def itk_resize_crop(self, nifti_csv_data):
        self.logger.info("Performing SimpleITK Resizing & Cropping")
        resized_cropped_dir = self.mkdir_prep_dir(self.rescrop_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.already_rescropped type = {}".format(type(self.already_rescropped)))
        if self.already_rescropped:
            self.logger.info("Adding Prepped SimpleITK Resized & Cropped filepaths to data df in raw_index & mask_index")

            # TODO (JG): This code is specific to nfbs, we'll need to adjust it for our 3 possibilities

            for i in tqdm(range(len(nifti_csv_data))):
                self.raw_index.append(self.rescrop_dirpath + os.sep + self.nifti_data_name + "_" + "raw" + str(i) + ".nii.gz")
                
                self.mask_index.append(self.rescrop_dirpath + os.sep + self.nifti_data_name + "_" + "mask" + str(i) + ".nii.gz")

            self.logger.info("Retrieved Prepped Resized & Cropped voxel filepaths stored at : {}/".format(resized_cropped_dir))
        else:
            # TODO (JG): Could Add NiFi Properties to set these parameters
            target_shape = [96, 128, 160]
            new_resolution = [2.0, 2.0, 2.0]

            # Create a new affine matrix for resizing and cropping
            new_affine = sitk.AffineTransform(3)
            new_affine.SetMatrix([new_resolution[0], 0, 0,
                                  0, new_resolution[1], 0,
                                  0, 0, new_resolution[2]])
            new_affine.SetTranslation([target_shape[0] * new_resolution[0] / 2*-1,
                                target_shape[1] * new_resolution[1] / 2*-1,
                                target_shape[2] * new_resolution[2] / 2*-1])

            for i in tqdm(range(len(nifti_csv_data))):
                # Load the image using ITK
                input_bias_corr_img = sitk.ReadImage(nifti_csv_data.bias_corr.iloc[i])
                output_image_path = os.path.join(self.rescrop_dirpath, self.nifti_data_name + "_" + "raw" + str(i) + ".nii.gz")

                # Resample and crop the image using SimpleITK
                resampled_image = self.sitk_resample_crop(input_bias_corr_img, new_affine, target_shape, new_resolution)

                # Write the resampled and cropped image using ITK
                sitk.WriteImage(resampled_image, output_image_path)
                self.raw_index.append(output_image_path)

                if self.nifti_data_name == "nfbs":
                    input_mask = sitk.ReadImage(nifti_csv_data.brain_mask.iloc[i])
                elif self.nifti_data_name == "atlas":
                    # TODO (JG): For atlas2.0, we have training and testing, address it
                    input_mask = sitk.ReadImage(nifti_csv_data.train_lesion_mask.iloc[i])
                elif self.nifti_data_name == "icpsr_stroke":
                    input_mask = sitk.ReadImage(nifti_csv_data.brain_dwi_mask.iloc[i])

                output_mask_path = os.path.join(self.rescrop_dirpath, self.nifti_data_name + "_" + "mask" + str(i) + ".nii.gz")

                # Resample and crop the image using SimpleITK
                resampled_mask = self.sitk_resample_crop(input_mask, new_affine, target_shape, new_resolution, is_binary_mask=True)

                # Write the resampled and cropped mask using ITK
                sitk.WriteImage(resampled_mask, output_mask_path)
                self.mask_index.append(output_mask_path)

        nifti_csv_data["raw_index"] = self.raw_index
        nifti_csv_data["mask_index"] = self.mask_index

        self.logger.info("ITK Resized & Cropped voxels stored at: {}/".format(resized_cropped_dir))

        return nifti_csv_data

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        # NIfTI: — Neuroimaging Informatics Technology Initiative
        nifti_csv = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: nifti_csv = {}".format(nifti_csv.head()))
        nifti_csv_prepped = self.itk_resize_crop(nifti_csv)
        self.logger.info("output: nifti_csv_prepped = {}".format(nifti_csv_prepped.head()))

        # Create a StringIO object
        nifti_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        nifti_csv_prepped.to_csv(nifti_string_io, index=False)

        # Get the string value and encode it
        nifti_csv_prepped_string = nifti_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = nifti_csv_prepped_string)
