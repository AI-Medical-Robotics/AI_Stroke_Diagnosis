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
import pandas as pd

import pickle5 as pickle

import SimpleITK as sitk

from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

# TODO (JG): Limitation in flow is flow file not passed to next processor until processor finishes work. This is with each processor like this

# TODO (JG): Update after saving pickle bytes file, then further compress to NIfTI file (smaller MB instead of GB)
    # PreprocessVoxelsForTorch previous processor name

class MapVoxelNamesToPrepVoxels(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['pandas==1.3.5', 'SimpleITK==2.2.1', 'pickle5==0.0.11']
        description = 'Gets SimpleITK Preprocessed Voxel NIfTI filepaths from the pandas csv dataframe in the flow file, loads each NIfTI file as a ITK Voxel, gets the voxel filename and creates a dictionary mapping voxel names to preprocessed voxels.'
        tags = ['sjsu_ms_ai', 'csv', 'nifti', 'itk']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.name_to_prep_voxel_dir = PropertyDescriptor(
            name = 'Voxel Name Mappings to Prepped Voxels Destination Path',
            description = 'The folder to store voxel names mapped to the preprocessed voxels.',
            default_value="/media/ubuntu/ai_projects/data/ICPSR_38464_Stroke_Data_NiFi/{}".format("map_voxnames_to_prepvoxs"),
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Mapped Voxel Names to Preprocessed Voxels',
            description = 'If Image Names Mapped to Preprocessed Voxels Already Performed, then just get filepaths',
            default_value=False,
            required = True,
        )
        self.data_type = PropertyDescriptor(
            name = 'Dataset Name',
            description = 'The name of the Dataset, currently supported: { icpsr_stroke }.',
            default_value = "icpsr_stroke",
            required=True
        )
        self.descriptors = [self.name_to_prep_voxel_dir, self.already_prepped, self.data_type]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for name_to_prep_voxel_dirpath, etc")
        self.img_name_to_prep_img_map = list()
        # read pre-trained model and config file
        self.name_to_prep_voxel_dirpath = context.getProperty(self.name_to_prep_voxel_dir.name).getValue()
        self.img_prep_already_done = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.data_name = context.getProperty(self.data_type.name).getValue()

    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    # TODO (JG): Finish
    def torch_preprocess_voxels(self, img_cap_csv_data):
        self.logger.info("Mapping Voxel Names to Preprocessed Voxels")
        prep_torch_dir = self.mkdir_prep_dir(self.name_to_prep_voxel_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.img_prep_already_done type = {}".format(type(self.img_prep_already_done)))
        if self.img_prep_already_done:
            self.logger.info("Adding Mapped Names to Preprocessed Voxels filepaths to data df in name_to_prep_voxel_dir")

            self.img_name_to_prep_img_map = [prep_torch_dir + os.sep + self.data_name + "_" + str(i) + ".pk1" for i in range(len(img_cap_csv_data))]
            img_cap_csv_data["img_name_to_prep_img"] = self.img_name_to_prep_img_map
            self.logger.info("Retrieved Mapped Names to Prepped Voxels filepaths stored at : {}/".format(prep_torch_dir))
        else:
            self.logger.info("Doing the Mapped Names to Prepped Voxels From Scratch")
            for i in range(len(img_cap_csv_data)):
                torch_imgname_to_prepimg = {}
                if self.data_name == "icpsr_stroke":
                    input_voxel_filepath = img_cap_csv_data.intensity_norm.iloc[i]
                    # input_voxel = sitk.ReadImage(img_cap_csv_data.intensity_norm.iloc[i], sitk.sitkFloat32)
                    # input_voxel_array = sitk.GetArrayFromImage(input_voxel)
                    img_name = os.path.basename(img_cap_csv_data.brain_dwi_orig.iloc[i])

                torch_imgname_to_prepimg[img_name] = input_voxel_filepath

                voxname_to_prepvoxs_bytes = pickle.dumps(torch_imgname_to_prepimg)

                # Save the voxel name mapped torch preprocessed voxel
                output_path = os.path.join(prep_torch_dir, self.data_name + "_" + str(i) + ".pk1")
                self.logger.info("Torch Mapped Names to Prepped Voxels pickle output_path = {}".format(output_path))
                try:
                    with open(output_path, "wb") as file:
                        file.write(voxname_to_prepvoxs_bytes)
                    self.logger.info("Torch Mapped Names to Prepped Voxels pickle bytes saved successfully!")
                except Exception as e:
                    self.logger.error("An error occurred while saving Mapped Names to Prepped Voxels!!!: {}".format(e))
                self.img_name_to_prep_img_map.append(output_path)

        img_cap_csv_data["img_name_to_prep_img"] = self.img_name_to_prep_img_map
        self.logger.info("Torch Mapped Names to Prepped Voxels pickle bytes stored at: {}/".format(prep_torch_dir))
        return img_cap_csv_data

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        img_cap_csv_pd = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: img_cap_csv_pd = {}".format(img_cap_csv_pd.head()))
        img_cap_csv_pd_up = self.torch_preprocess_voxels(img_cap_csv_pd)
        self.logger.info("output: img_cap_csv_pd_up = {}".format(img_cap_csv_pd_up.head()))

        # Create a StringIO object
        img_cap_csv_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        img_cap_csv_pd_up.to_csv(img_cap_csv_string_io, index=False)

        # Get the string value and encode it
        img_cap_csv_pd_up_string = img_cap_csv_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = img_cap_csv_pd_up_string)
