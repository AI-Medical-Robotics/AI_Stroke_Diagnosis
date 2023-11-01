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

import SimpleITK as sitk
import pickle5 as pickle

# from tqdm import tqdm
from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

# TODO (JG): Limitation in flow is flow file not passed to next processor until processor finishes work. This is with each processor like this

# TODO (JG): Make this work for training and testing sets
class PreprocessCaptionsForTorch(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['pandas==1.3.5', 'pillow==10.0.0', 'pickle5==0.0.11', 'SimpleITK==2.2.1']
        description = 'Gets Mapped 2D or 3D Image IDs to Captions pickle bytes filepaths and 3D CNN Extracted Features bytes filepaths from the pandas csv dataframe in the flow file, loads these mappings from pickle bytes files, performs multiple preprocessing operations (lowercasing; removing digits, special chars, white space) on each image ID\'s captions string, returning back an updated map of image ID to prepped captions, storing these mappings to pickle bytes files and has the filepaths saved in pd df'
        tags = ['sjsu_ms_ai', 'csv', 'jpeg', 'nifti']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.prep_torch_captions_dir = PropertyDescriptor(
            name = 'Preprocessed Captions Destination Path',
            description = 'The folder to store the Preprocessed Captions.',
            default_value="{}/src/datasets/flickr8k_NiFi/{}".format(os.path.expanduser("~"), "map_imgids_to_prep_captions"),
            required = True
        )
        self.torch_extfets_dir = PropertyDescriptor(
            name = 'Extracted Features Destination Path',
            description = 'The folder to store the 3D CNN Extracted Features associated with image id in filename.',
            default_value="{}/src/datasets/flickr8k_NiFi/{}".format(os.path.expanduser("~"), "voxid_extfets_rel_to_caps"),
            required = True
        )
        self.source_colname1_df = PropertyDescriptor(
            name = 'Source DataFrame Colname1',
            description = 'Specify the source column name for holding the incoming paths of image id to captions mappings.',
            default_value="imgid_to_captions",
            required = True
        )
        self.source_colname2_df = PropertyDescriptor(
            name = 'Source DataFrame Colname2',
            description = 'Specify the source column name for holding the incoming paths of image id to extracted feature mappings.',
            default_value="imgid_to_feature",
            required = True
        )
        self.target_colname1_df = PropertyDescriptor(
            name = 'Target DataFrame Colname1',
            description = 'Specify the target column name for holding the destination paths of image id to preprocessed captions mappings.',
            default_value="imgid_to_prep_captions",
            required = True
        )
        self.target_colname2_df = PropertyDescriptor(
            name = 'Target DataFrame Colname2',
            description = 'Specify the target column name for holding the destination paths of image id extracted feature files to prep caption mappings.',
            default_value="imgid_extfet_filepaths",
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Preprocessed Captions',
            description = 'If Preprocessed Captions Already Performed, then just get filepaths',
            default_value=False,
            required = False,
        )
        self.data_type = PropertyDescriptor(
            name = 'Dataset Name',
            description = 'The name of the Dataset, currently supported: { flickr , icpsr_stroke}.',
            default_value = "flickr",
            required=True
        )
        self.cap_del_pattern = PropertyDescriptor(
            name = 'Delete Caption Characters Pattern',
            description = 'Provide a Regex Pattern to Delete Caption Characters. By default, deletes digits, special characters & other chars not in the alphabet',
            default_value = "[^A-Za-z]",
            required=True
        )
        self.descriptors = [self.prep_torch_captions_dir, self.torch_extfets_dir, self.source_colname1_df, self.source_colname2_df, self.target_colname1_df, self.target_colname2_df, self.already_prepped, self.data_type, self.cap_del_pattern]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for prep_torch_captions_dirpath, etc")
        self.imgid_to_prep_captions_map = list()
        self.prep_torch_captions_dirpath = context.getProperty(self.prep_torch_captions_dir.name).getValue()
        self.torch_extfets_dirpath = context.getProperty(self.torch_extfets_dir.name).getValue()
        self.source_colname1 = context.getProperty(self.source_colname1_df.name).getValue()
        self.source_colname2 = context.getProperty(self.source_colname2_df.name).getValue()

        self.target_colname1 = context.getProperty(self.target_colname1_df.name).getValue()
        self.target_colname2 = context.getProperty(self.target_colname2_df.name).getValue()
        self.cap_prep_already_done = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.data_name = context.getProperty(self.data_type.name).getValue()
        self.cap_del_regex_pattern = context.getProperty(self.cap_del_pattern.name).getValue()


    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

# map: nifti_csv_data["imgid_to_feature"].tolist(), nifti_csv_data["voxid_to_prep_captions"].tolist()

    # TODO (JG): Finish
    def preprocess_captions(self, img_cap_csv_data):
        self.logger.info("Performing Captions Preprocessing")
        prep_captions_dir = self.mkdir_prep_dir(self.prep_torch_captions_dirpath)
        self.mkdir_prep_dir(self.torch_extfets_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.cap_prep_already_done type = {}".format(type(self.cap_prep_already_done)))
        if self.cap_prep_already_done:
            self.logger.info("Adding Prepped Captions filepaths to data df in prep_torch")

            self.imgid_to_prep_captions_map = [prep_captions_dir + os.sep + self.data_name + "_" + str(i) + ".pk1" for i in range(len(img_cap_csv_data))]
            img_cap_csv_data[self.target_colname1] = self.imgid_to_prep_captions_map
            self.logger.info("Retrieved Prepped Captions filepaths stored at : {}/".format(prep_captions_dir))
        else:
            self.logger.info("Doing the Captions Preprocessing From Scratch")
            # 1st col is extfets filepaths, 2nd col is imgid to prep captions filepaths
            new_voxid_extfets_caps_df = pd.DataFrame(columns = [self.target_colname2, self.target_colname1])

            for i in range(len(img_cap_csv_data)):
                imgid_to_caps = None
                imgid_to_prep_captions_set = ()

                with open(img_cap_csv_data[self.source_colname2].iloc[i], "rb") as file:
                    imgid_to_extfets = pickle.load(file)

                imgid_from_fets = list(imgid_to_extfets.keys())[0]
                imgid_unet3d_extfeatures = list(imgid_to_extfets.values())[0]
                self.logger.info(f"imgid_from_fets = {imgid_from_fets}")
                self.logger.info(f"imgid_unet3d_extfeatures.shape = {imgid_unet3d_extfeatures.shape}")


                self.logger.info("img_cap_csv_data[\"{}\"].iloc[i] = {}".format(self.source_colname1, img_cap_csv_data[self.source_colname1].iloc[i]))
                with open(img_cap_csv_data[self.source_colname1].iloc[i], "rb") as file:
                    imgid_to_caps = pickle.load(file)
                
                    self.logger.info("check1: imgid_to_caps len = {}".format(len(imgid_to_caps)))
                    self.logger.info("imgid_to_caps[0] imgid_from_caps = {}".format(imgid_to_caps[0]))
                    self.logger.info("imgid_to_caps[1] captions_str = {}".format(imgid_to_caps[1]))

                self.logger.info("check2: imgid_to_caps len = {}".format(len(imgid_to_caps)))
                # imgid_from_caps = list(imgid_to_caps.keys())[0]
                # captions_str = list(imgid_to_caps.values())[0]

                imgid_from_caps = imgid_to_caps[0]
                captions_str = imgid_to_caps[1]

                if imgid_from_caps in imgid_from_fets:
                    self.logger.info("Voxel ID from UNet3D ext features matches Voxel ID from captions")
                    # output_imgid_extfets_path = os.path.join(self.torch_extfets_dirpath, imgid_from_caps + "_" + str(i) + ".nii.gz")
                    # sitk.WriteImage(imgid_unet3d_extfeatures, output_imgid_extfets_path)
                    # self.logger.info("UNet3D ext features saved with voxid_from_caps in filename {}".format(output_imgid_extfets_path))


                    imgid_to_extfeatures_set = (imgid_from_caps, imgid_unet3d_extfeatures)

                    imgid_to_extfeatures_bytes = pickle.dumps(imgid_to_extfeatures_set)

                    # Save the image name mapped torch preprocessed image
                    output_imgid_extfets_path = os.path.join(self.torch_extfets_dirpath, imgid_from_caps + "_" + str(i) + ".pk1")
                    self.logger.info("UNet3D ext features being saved with voxid_from_caps in filename {}".format(output_imgid_extfets_path))
                    try:
                        with open(output_imgid_extfets_path, "wb") as file:
                            file.write(imgid_to_extfeatures_bytes)
                        self.logger.info("UNet3D ext features with voxid_from_caps in filename pickle bytes saved successfully!")
                    except Exception as e:
                        self.logger.error("An error occurred while saving UNet3D ext features with voxid_from_caps in filename!!!: {}".format(e))


                    for cap_i in range(len(captions_str)):
                        caption = captions_str[cap_i]
                        # Convert to lowercase
                        caption = caption.lower()

                        # delete digits, special cahrs, etc
                        caption = caption.replace(self.cap_del_regex_pattern, '')

                        # delete additional spaces
                        caption = caption.replace('\s+', ' ')

                        # add start and end tags to caption
                        # caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'

                        # captions_str[cap_i] = caption

                        imgid_to_prep_captions_set = (imgid_from_caps, caption)

                        imgid_to_prep_caption_bytes = pickle.dumps(imgid_to_prep_captions_set)

                        # Save the image name mapped torch preprocessed image
                        output_imgid_prepcaps_path = os.path.join(prep_captions_dir, self.data_name + "_" + str(i) + "_cap_" + str(cap_i) + ".pk1")
                        self.logger.info("Mapped Image IDs to Preprocessed Captions pickle output_path = {}".format(output_imgid_prepcaps_path))
                        try:
                            with open(output_imgid_prepcaps_path, "wb") as file:
                                file.write(imgid_to_prep_caption_bytes)
                            self.logger.info("Mapped Image IDs to Preprocessed Captions pickle bytes saved successfully!")
                        except Exception as e:
                            self.logger.error("An error occurred while saving Mapped Image IDs to Preprocessed Captions!!!: {}".format(e))
                        # TODO (JG): Write to pd dataframe two columns
                        # self.imgid_to_prep_captions_map.append(output_imgid_prepcaps_path)

                        new_voxid_extfets_caps_df.loc[len(new_voxid_extfets_caps_df)] = [output_imgid_extfets_path, output_imgid_prepcaps_path]
                        self.logger.info("Wrote new row of voxid extracted feature filepath and voxid prep caption filepath to pd")
                else:
                    self.logger.info("WARNING: Voxel ID from UNet3D ext features unequal to Voxel ID from prep captions")

        # img_cap_csv_data[self.target_colname1] = self.imgid_to_prep_captions_map
        self.logger.info("Mapped Image IDs to Preprocessed Captions pickle bytes stored at: {}/".format(prep_captions_dir))
        return new_voxid_extfets_caps_df

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        img_cap_csv_pd = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: img_cap_csv_pd = {}".format(img_cap_csv_pd.head()))
        img_cap_csv_pd_up = self.preprocess_captions(img_cap_csv_pd)
        self.logger.info("output: img_cap_csv_pd_up = {}".format(img_cap_csv_pd_up.head()))

        # Create a StringIO object
        img_cap_csv_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        img_cap_csv_pd_up.to_csv(img_cap_csv_string_io, index=False)

        # Get the string value and encode it
        img_cap_csv_pd_up_string = img_cap_csv_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = img_cap_csv_pd_up_string)
