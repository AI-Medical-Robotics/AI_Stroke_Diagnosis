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
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from py4j.java_collections import ListConverter

# TODO (JG): Make this work for training and testing sets
class NormalizeIntensityInITKImage(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['SimpleITK', 'pandas', 'tqdm']
        description = 'Gets the resized and cropped NIfTI filepaths from the pandas csv dataframe in the flow file, loads each NIfTI file as an ITK voxel and performs SimpleITK intensity normalization on each 3D NIfTI voxel'
        tags = ['sjsu_ms_ai', 'csv', 'itk', 'nifti']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.intensity_norm_dir = PropertyDescriptor(
            name = 'ITK Intensity Normalized Folder Path',
            description = 'The folder to stored the ITK Preprocessed Intensity Normalized Images.',
            default_value="{}/src/datasets/NFBS_Dataset_NiFi/{}".format(os.path.expanduser("~"), "intensity_norm"),
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Intensity Normalized',
            description = 'If ITK Intensity Normalization Already Performed, then just get filepaths',
            default_value=False,
            required = True,
        )
        self.nifti_data_type = PropertyDescriptor(
            name = 'NifTI Dataset Name',
            description = 'The name of the NifTI Dataset, currently supported: {nfbs, atlas, icpsr_stroke}.',
            default_value = "nfbs",
            required=True
        )
        self.descriptors = [self.intensity_norm_dir, self.already_prepped, self.nifti_data_type]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for intensity_norm_dirpath, etc")
        self.intensity_norm = list()
        # read pre-trained model and config file
        self.intensity_norm_dirpath = context.getProperty(self.intensity_norm_dir.name).getValue()
        self.already_intens_normed = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.nifti_data_name = context.getProperty(self.nifti_data_type.name).getValue()

    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    def sitk_intensity_normalized(self, input_image, max=255, min=0):
        # Resample and crop the image uisng ITK
        resacle_filter = sitk.RescaleIntensityImageFilter()
        resacle_filter.SetOutputMaximum(max)
        resacle_filter.SetOutputMinimum(min)
        normalized_image = resacle_filter.Execute(input_image)
        return normalized_image

    def itk_intensity_normalized(self, nifti_csv_data):
        self.logger.info("Performing SimpleITK Intensity Normalization")
        intensity_norm_dir = self.mkdir_prep_dir(self.intensity_norm_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.already_intens_normed type = {}".format(type(self.already_intens_normed)))
        if self.already_intens_normed:
            self.logger.info("Adding Prepped SimpleITK Intensity Normalized filepaths to data df intensity_norm")

            # TODO (JG): This code is specific to nfbs, we'll need to adjust it for our 3 possibilities

            for i in tqdm(range(len(nifti_csv_data))):
                self.intensity_norm.append(self.intensity_norm_dirpath + os.sep + self.nifti_data_name + "_" + "intensity_norm_" + str(i) + ".nii.gz")

            self.logger.info("Retrieved Prepped Intensity Normalized voxel filepaths stored at : {}/".format(self.intensity_norm_dirpath))
        else:
            for i in tqdm(range(len(nifti_csv_data))):
                # Load the Raw Resized & Cropped image using ITK
                input_resize_cropped_img = sitk.ReadImage(nifti_csv_data.raw_index.iloc[i])
                output_image_path = os.path.join(self.intensity_norm_dirpath + os.sep + self.nifti_data_name + "_" + "intensity_norm_" + str(i) + ".nii.gz")

                # Resample and crop the image using SimpleITK
                intensity_normalized_image = self.sitk_intensity_normalized(input_resize_cropped_img, max=255, min=0)

                # Write the resampled and cropped image using ITK
                sitk.WriteImage(intensity_normalized_image, output_image_path)
                self.intensity_norm.append(output_image_path)
            self.logger.info("ITK Intensity Normalized voxels stored at: {}/".format(self.intensity_norm_dirpath))

        nifti_csv_data["intensity_norm"] = self.intensity_norm

        return nifti_csv_data

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        # NIfTI: — Neuroimaging Informatics Technology Initiative
        nifti_csv = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: nifti_csv = {}".format(nifti_csv.head()))
        nifti_csv_prepped = self.itk_intensity_normalized(nifti_csv)
        self.logger.info("output: nifti_csv_prepped = {}".format(nifti_csv_prepped.head()))

        # Create a StringIO object
        nifti_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        nifti_csv_prepped.to_csv(nifti_string_io, index=False)

        # Get the string value and encode it
        nifti_csv_prepped_string = nifti_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = nifti_csv_prepped_string)
