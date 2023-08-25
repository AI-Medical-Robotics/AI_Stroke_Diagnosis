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
import torchvision
import pickle5 as pickle

# from tqdm import tqdm
from nifiapi.properties import PropertyDescriptor, StandardValidators
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

# Verified we can run SimpleITK N4 Bias Field Correction and produces expected results faster than nipype's version

# TODO (JG): Limitation in flow is flow file not passed to next processor until processor finishes work. This is with each processor like this

# TODO (JG): Make this work for training and testing sets
class MapImageIDsToCaptions(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['pandas==1.3.5', 'pillow==10.0.0', 'pickle5==0.0.11', "torch", "torchvision", "torchaudio"]
        description = 'Gets Flickr image captions txt metadata, performs dictionary mapping image ID to image captions, saves each dictionary element as pickle bytes and stores filepath to pandas dataframe'
        tags = ['sjsu_ms_ai', 'csv', 'jpeg', 'pytorch']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.captions_source_path = PropertyDescriptor(name="Flickr Captions Source Path",
            description="Flickr Captions Source Path where image captions txt metadata is located",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
            default_value="{}/src/datasets/flickr8k/captions.txt".format(os.path.expanduser("~")),
            required=True)
        self.imgid_to_caption_dir = PropertyDescriptor(
            name = 'Image ID Mappings to Captions Destination Path',
            description = 'The folder to store the image IDs mapped to captions',
            default_value="{}/src/datasets/flickr8k_NiFi/{}".format(os.path.expanduser("~"), "map_imgids_to_captions"),
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Mapped Image IDs to Captions',
            description = 'If Image IDs Mapped to Captions Already Performed, then just get filepaths',
            default_value=False,
            required = True,
        )
        self.jpeg_data_type = PropertyDescriptor(
            name = 'JPEG Dataset Name',
            description = 'The name of the JPEG Dataset, currently supported: { flickr }.',
            default_value = "flickr",
            required=True
        )
        self.descriptors = [self.captions_source_path, self.imgid_to_caption_dir, self.already_prepped, self.jpeg_data_type]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for imgid_to_caption_dirpath, etc")
        self.imgid_to_captions_files = list()
        # read pre-trained model and config file
        self.captions_source_filepath = context.getProperty(self.captions_source_path.name).getValue()
        self.imgid_to_caption_dirpath = context.getProperty(self.imgid_to_caption_dir.name).getValue()
        self.img_map_already_done = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.jpeg_data_name = context.getProperty(self.jpeg_data_type.name).getValue()

    def load_caption_data(self):
        with open(self.captions_source_filepath, "r") as f:
            next(f)
            captions_doc = f.read()
        return captions_doc

    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    # TODO (JG): Finish
    def map_imgids_to_captions(self, img_cap_csv_data):
        self.logger.info("Mapping Image IDs to Captions")
        imgid_to_caption_dir = self.mkdir_prep_dir(self.imgid_to_caption_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.img_map_already_done type = {}".format(type(self.img_map_already_done)))
        if self.img_map_already_done:
            self.logger.info("Adding Mapped IDs to Captions filepaths to data df in imgid_to_caption_dir")

            self.imgid_to_captions_files = [imgid_to_caption_dir + os.sep + self.jpeg_data_name + "_" + str(i) + ".pk1" for i in range(len(img_cap_csv_data))]
            img_cap_csv_data["imgid_to_captions"] = self.imgid_to_captions_files
            self.logger.info("Retrieved Mapped IDs to Captions filepaths stored at : {}/".format(imgid_to_caption_dir))
        else:
            self.logger.info("Doing the Mapped Image IDs to Captions From Scratch")

            if self.jpeg_data_name == "flickr":
                caption_txt_labels = self.load_caption_data()
            # elif self.jpeg_data_name == "atlas":
            #     input_image = sitk.ReadImage(img_cap_csv_data.train_t1w_raw.iloc[i], sitk.sitkFloat32)
            # elif self.jpeg_data_name == "icpsr_stroke":
            #     input_image = sitk.ReadImage(img_cap_csv_data.brain_dwi_orig.iloc[i], sitk.sitkFloat32)

            imgid_to_captions_map = {}

            for caption_label in caption_txt_labels.split("\n"):

                tokens = caption_label.split(",")
                if len(caption_label) < 2:
                    continue
                
                image_id, caption = tokens[0], tokens[1:]

                image_id = image_id.split(".")[0]

                caption_str = " ".join(caption)

                if image_id not in imgid_to_captions_map:
                    imgid_to_captions_map[image_id] = list()

                imgid_to_captions_map[image_id].append(caption_str)

                imgid_to_caption_bytes = pickle.dumps(imgid_to_captions_map)

                # Save the image name mapped torch preprocessed image
                output_path = os.path.join(imgid_to_caption_dir, self.jpeg_data_name + "_" + str(i) + ".pk1")
                self.logger.info("Mapped Image IDs to Captions pickle output_path = {}".format(output_path))
                try:
                    with open(output_path, "wb") as file:
                        file.write(imgid_to_caption_bytes)
                    self.logger.info("Mapped Image IDs to Captions pickle bytes saved successfully!")
                except Exception as e:
                    self.logger.error("An error occurred while saving Mapped Image IDs to Captions!!!: {}".format(e))
                self.imgid_to_captions_files.append(output_path)

        img_cap_csv_data["imgid_to_captions"] = self.imgid_to_captions_files
        self.logger.info("Mapped Image IDs to Captions pickle bytes stored at: {}/".format(imgid_to_caption_dir))
        return img_cap_csv_data

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        img_cap_csv_pd = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: img_cap_csv_pd = {}".format(img_cap_csv_pd.head()))
        img_cap_csv_pd_up = self.map_imgids_to_captions(img_cap_csv_pd)
        self.logger.info("output: img_cap_csv_pd_up = {}".format(img_cap_csv_pd_up.head()))

        # Create a StringIO object
        img_cap_csv_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        img_cap_csv_pd_up.to_csv(img_cap_csv_string_io, index=False)

        # Get the string value and encode it
        img_cap_csv_pd_up_string = img_cap_csv_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = img_cap_csv_pd_up_string)
