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

import torchvision.transforms as transforms
import pickle5 as pickle

import PIL

# from tqdm import tqdm
from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

# Verified we can run SimpleITK N4 Bias Field Correction and produces expected results faster than nipype's version

# TODO (JG): Limitation in flow is flow file not passed to next processor until processor finishes work. This is with each processor like this

# TODO (JG): Make this work for training and testing sets
class PreprocessImagesForTorch(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['pandas==1.3.5', 'pillow==10.0.0', 'pickle5==0.0.11', "torch", "torchvision", "torchaudio"]
        description = 'Gets Flickr JPEG filepaths from the pandas csv dataframe in the flow file, loads each JPEG file as a PIL image and performs resizing, pixel to tensor conversion and normalization on each PIL image using pytorch'
        tags = ['sjsu_ms_ai', 'csv', 'PIL', 'jpeg', 'pytorch']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.prep_torch_imgs_dir = PropertyDescriptor(
            name = 'Preprocessed Torch Images Destination Path',
            description = 'The folder to stored the Preprocessed Torch Images.',
            default_value="{}/src/datasets/flickr8k_NiFi/{}".format(os.path.expanduser("~"), "prep_torch_images"),
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Preprocessed Torch Images',
            description = 'If Preprocessed Torch Images Already Performed, then just get filepaths',
            default_value=False,
            required = True,
        )
        self.jpeg_data_type = PropertyDescriptor(
            name = 'JPEG Dataset Name',
            description = 'The name of the JPEG Dataset, currently supported: { flickr }.',
            default_value = "flickr",
            required=True
        )
        self.resize = PropertyDescriptor(
            name = 'Resize',
            description = 'Resize the Image. Ex: if 2D image, resize by 256 for each dimension',
            default_value = "256",
            required=True
        )
        self.center_crop = PropertyDescriptor(
            name = 'Center Crop',
            description = 'Leaves the center region of the image, removing the outer parts',
            default_value = "224",
            required=True
        )
        self.normalization = PropertyDescriptor(
            name = 'Mean & Std Dev Normalization',
            description = 'Bring pixels in the image to a common scale by doing mean and std dev channel wise normalization',
            default_value = """{"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}""",
            required=True
        )
        self.expected_norm_keys = ['mean', 'std']
        self.descriptors = [self.prep_torch_imgs_dir, self.already_prepped, self.jpeg_data_type, self.resize, self.center_crop, self.normalization]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for prep_torch_imgs_dirpath, etc")
        self.img_name_to_prep_img_map = list()
        # read pre-trained model and config file
        self.prep_torch_imgs_dirpath = context.getProperty(self.prep_torch_imgs_dir.name).getValue()
        self.img_prep_already_done = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.jpeg_data_name = context.getProperty(self.jpeg_data_type.name).getValue()
        self.resize_img = context.getProperty(self.resize.name).asInteger()
        self.center_crop_img = context.getProperty(self.center_crop.name).asInteger()
        self.mean_std_norm_json_str = context.getProperty(self.normalization.name).getValue()

    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    # TODO (JG): Finish
    def torch_preprocess_images(self, img_cap_csv_data):
        self.logger.info("Performing Torch Image Preprocessing")
        prep_torch_dir = self.mkdir_prep_dir(self.prep_torch_imgs_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.img_prep_already_done type = {}".format(type(self.img_prep_already_done)))
        if self.img_prep_already_done:
            self.logger.info("Adding Prepped Torch Image filepaths to data df in prep_torch")

            self.img_name_to_prep_img_map = [prep_torch_dir + os.sep + self.jpeg_data_name + "_" + str(i) + ".pk1" for i in range(len(img_cap_csv_data))]
            img_cap_csv_data["img_name_to_prep_img"] = self.img_name_to_prep_img_map
            self.logger.info("Retrieved Prepped Torch Image filepaths stored at : {}/".format(prep_torch_dir))
        else:
            self.logger.info("Doing the Torch Image Preprocessing From Scratch")
            for i in range(len(img_cap_csv_data)):
                torch_imgname_to_prepimg = {}
                # Load the image using PIL
                if self.jpeg_data_name == "flickr":
                    input_image = PIL.Image(img_cap_csv_data.natural_image.iloc[i])
                    img_name = os.path.basename(img_cap_csv_data.natural_image.iloc[i])
                # elif self.jpeg_data_name == "atlas":
                #     input_image = sitk.ReadImage(img_cap_csv_data.train_t1w_raw.iloc[i], sitk.sitkFloat32)
                # elif self.jpeg_data_name == "icpsr_stroke":
                #     input_image = sitk.ReadImage(img_cap_csv_data.brain_dwi_orig.iloc[i], sitk.sitkFloat32)

                # Extract mean & std dev from json
                norm_dict = json.loads(self.mean_std_norm_json_str)
                for norm_key in self.expected_norm_keys:
                    if norm_key not in norm_dict:
                        self.logger.error("The key '{}' is missing or incorrect in the JSON data.".format(norm_key))

                # Perform Torch Preprocessing: Resize, CenterCrop, ToTensor, Normalize
                preprocess = transforms.Compose([
                    transforms.Resize(self.resize_img),
                    transforms.CenterCrop(self.center_crop_img),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_dict["mean"], std=norm_dict["std"])
                ])

                input_img_tensor = preprocess(input_image)
                input_tensor_batch = input_img_tensor.unsqueeze(0)

                torch_imgname_to_prepimg[img_name] = input_tensor_batch

                preprocessed_torch_bytes = pickle.dumps(torch_imgname_to_prepimg)

                # Save the image name mapped torch preprocessed image
                output_path = os.path.join(prep_torch_dir, self.jpeg_data_name + "_" + str(i) + ".pk1")
                self.logger.info("Torch Preprocessed Image pickle output_path = {}".format(output_path))
                try:
                    with open(output_path, "wb") as file:
                        file.write(preprocessed_torch_bytes)
                    self.logger.info("Torch Preprocessed Image Mapped to Img Name pickle bytes saved successfully!")
                except Exception as e:
                    self.logger.error("An error occurred while saving Torch Preprocessed Image!!!: {}".format(e))
                self.img_name_to_prep_img_map.append(output_path)

        img_cap_csv_data["img_name_to_prep_img"] = self.img_name_to_prep_img_map
        self.logger.info("Torch Mapped Img Names to Preprocessed Images pickle bytes stored at: {}/".format(prep_torch_dir))
        return img_cap_csv_data

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        img_cap_csv_pd = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: img_cap_csv_pd = {}".format(img_cap_csv_pd.head()))
        img_cap_csv_pd_up = self.torch_preprocess_images(img_cap_csv_pd)
        self.logger.info("output: img_cap_csv_pd_up = {}".format(img_cap_csv_pd_up.head()))

        # Create a StringIO object
        img_cap_csv_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        img_cap_csv_pd_up.to_csv(img_cap_csv_string_io, index=False)

        # Get the string value and encode it
        img_cap_csv_pd_up_string = img_cap_csv_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = img_cap_csv_pd_up_string)
