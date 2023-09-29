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
import pandas as pd

import torch
# import torchvision
import pickle5 as pickle

import monai
# from monai.networks.nets import BasicUNet
# from monai.transforms import Compose, ToTensor

# from tqdm import tqdm
from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

# TODO (JG): Limitation in flow is flow file not passed to next processor until processor finishes work. This is with each processor like this

# TODO (JG): Update after saving pickle bytes file, then further compress to NIfTI file (smaller MB instead of GB)
class MapVoxelIDsToExtFeatures(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['pandas==1.3.5', 'pillow==10.0.0', 'pickle5==0.0.11', "torch", "torchvision", "torchaudio", "monai"]
        description = 'Gets MONAI PyTorch Preprocessed 3D voxel pickle bytes filepaths from the pandas csv dataframe in the flow file, loads each pickle bytes as a dictionary mapping 3D voxel names to preprocessed 3D vxels and runs pretrained MONAI pytorch 3D cnn model (BasicUNet or UNet) to extract features and then map to 3D voxel IDs'
        tags = ['sjsu_ms_ai', 'csv', 'nifti', 'pytorch', 'monai']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.imgid_to_ext_feature_dir = PropertyDescriptor(
            name = 'Voxel ID Mappings to Extracted Features Destination Path',
            description = 'The folder to store the voxel IDs mapped to extracted features',
            default_value="/media/ubuntu/ai_projects/data/ICPSR_38464_Stroke_Data_NiFi/{}".format("map_voxids_to_features"),
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Mapped Voxel IDs to Extracted Features',
            description = 'If Voxel IDs Mapped to Extracted Features Already Performed, then just get filepaths',
            default_value=False,
            required = True,
        )
        self.data_type = PropertyDescriptor(
            name = 'Dataset Name',
            description = 'The name of the Dataset, currently supported: { icpsr_stroke }.',
            default_value = "icpsr_stroke",
            required=True
        )
        self.torch_model_type = PropertyDescriptor(
            name = 'PyTorch 3D CNN Model',
            description = 'The pretrained pytorch model or monai pytorch model to use for voxel feature extraction, currently supported: { monai_basic_unet }.',
            default_value = "monai_basic_unet",
            required=True
        )
        self.descriptors = [self.imgid_to_ext_feature_dir, self.already_prepped, self.data_type, self.torch_model_type]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for bias_corrected_dirpath, etc")
        self.imgid_to_feature_map = list()
        # read pre-trained model and config file
        self.imgid_to_ext_feature_dirpath = context.getProperty(self.imgid_to_ext_feature_dir.name).getValue()
        self.img_map_already_done = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.data_name = context.getProperty(self.data_type.name).getValue()
        self.torch_model_name = context.getProperty(self.torch_model_type.name).getValue()

    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    # TODO (JG): Finish
    def map_imgids_to_extfeatures(self, img_cap_csv_data):
        self.logger.info("Mapping Voxel IDs to PyTorch CNN Extracted Features")
        imgid_to_extfeature_dir = self.mkdir_prep_dir(self.imgid_to_ext_feature_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.img_map_already_done type = {}".format(type(self.img_map_already_done)))
        if self.img_map_already_done:
            self.logger.info("Adding Mapped IDs to Extracted Features filepaths to data df in imgid_to_ext_feature_dir")

            self.imgid_to_feature_map = [imgid_to_extfeature_dir + os.sep + self.data_name + "_" + str(i) + ".pk1" for i in range(len(img_cap_csv_data))]
            img_cap_csv_data["imgid_to_feature"] = self.imgid_to_feature_map
            self.logger.info("Retrieved Mapped IDs to Extracted Features filepaths stored at : {}/".format(imgid_to_extfeature_dir))
        else:
            self.logger.info("Doing the Torch Mapped IDs to Extracted Features From Scratch")
            for i in range(len(img_cap_csv_data)):
                torch_imgid_to_extfeatures = {}
                # elif self.data_name == "atlas":
                #     input_voxel = sitk.ReadImage(img_cap_csv_data.train_t1w_raw.iloc[i], sitk.sitkFloat32)
                if self.data_name == "icpsr_stroke":
                    with open(img_cap_csv_data.img_name_to_prep_img.iloc[i], "rb") as file:
                        torch_prep_name_to_img = pickle.load(file)
                    # input_voxel = sitk.ReadImage(img_cap_csv_data.brain_dwi_orig.iloc[i], sitk.sitkFloat32)

                # Perform Torch Feature Extraction with (VGG16 or Resnet50)
                voxel_filename = list(torch_prep_name_to_img.keys())[0]
                prep_voxel_tensor_batch = list(torch_prep_name_to_img.values())[0]

                # TODO (JG): Add support for 3D U-Net, V-Net, Dense VoxelNet or VoxResNet
                # Use preprocessed voxel input to VGG16 model
                if self.torch_model_name == "monai_basic_unet":
                    # TODO (JG): Need to check if its pre-trained
                    unet_model = monai.networks.nets.BasicUNet()

                with torch.no_grad():
                    features_extracted = unet_model(prep_voxel_tensor_batch)

                voxel_id = voxel_filename.split(".")[0]

                torch_imgid_to_extfeatures[voxel_id] = features_extracted

                imgid_to_extfeature_bytes = pickle.dumps(torch_imgid_to_extfeatures)

                # Save the voxel name mapped torch preprocessed voxel
                output_path = os.path.join(imgid_to_extfeature_dir, self.data_name + "_" + str(i) + ".pk1")
                self.logger.info("Torch Mapped IDs to Extracted Features pickle output_path = {}".format(output_path))
                try:
                    with open(output_path, "wb") as file:
                        file.write(imgid_to_extfeature_bytes)
                    self.logger.info("Torch Mapped IDs to Extracted Features pickle bytes saved successfully!")
                except Exception as e:
                    self.logger.error("An error occurred while saving Mapped IDs to Extracted Features!!!: {}".format(e))
                self.imgid_to_feature_map.append(output_path)

        img_cap_csv_data["imgid_to_feature"] = self.imgid_to_feature_map
        self.logger.info("Torch Mapped IDs to Extracted Features pickle bytes stored at: {}/".format(imgid_to_extfeature_dir))
        return img_cap_csv_data

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        img_cap_csv_pd = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: img_cap_csv_pd = {}".format(img_cap_csv_pd.head()))
        img_cap_csv_pd_up = self.map_imgids_to_extfeatures(img_cap_csv_pd)
        self.logger.info("output: img_cap_csv_pd_up = {}".format(img_cap_csv_pd_up.head()))

        # Create a StringIO object
        img_cap_csv_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        img_cap_csv_pd_up.to_csv(img_cap_csv_string_io, index=False)

        # Get the string value and encode it
        img_cap_csv_pd_up_string = img_cap_csv_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = img_cap_csv_pd_up_string)
