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
import re
import json
import pandas as pd

import SimpleITK as sitk
import pickle5 as pickle

from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

class MapPrepVoxelToPrepCaptions(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['pandas==1.3.5', 'pillow==10.0.0', 'pickle5==0.0.11', 'SimpleITK==2.2.1']
        description = 'Gets Mapped 3D Voxel IDs to Clinical Captions pickle bytes filepaths and Mapped 3D Voxel IDs to Prepped Voxels pickle bytes filepaths from the incoming pandas csv dataframe in the flow file, loads these mappings from pickle bytes files, performs multiple preprocessing operations (lowercasing; removing digits, special chars, white space) on each voxel ID\'s captions string, returning back an updated map of voxel ID to prepped captions, and then maps prepped voxels to prepped captions by voxel ID, storing these mappings to pickle bytes files and has the filepaths saved in a new pd df'
        tags = ['sjsu_ms_ai', 'csv', 'jpeg', 'nifti']

    def __init__(self, **kwargs):
        self.prep_captions_dir = PropertyDescriptor(
            name = 'Preprocessed Captions Destination Path',
            description = 'The folder to store the preprocessed captions.',
            default_value="{}/src/datasets/flickr8k_NiFi/{}".format(os.path.expanduser("~"), "map_voxids_to_prep_captions"),
            required = True
        )
        self.prep_voxels_dir = PropertyDescriptor(
            name = 'Preprocessed 3D Voxels Destination Path',
            description = 'The folder to store the preprocessed 3D voxels associated with voxel id in filename.',
            default_value="{}/src/datasets/flickr8k_NiFi/{}".format(os.path.expanduser("~"), "voxid_prepvoxs_rel_to_caps"),
            required = True
        )
        self.source_colname1_df = PropertyDescriptor(
            name = 'Source DataFrame Colname1',
            description = 'Specify the source column name for holding the incoming paths of 3D voxel id to captions mappings. For ex: one voxel id to one or more captions',
            default_value="voxid_to_captions",
            required = True
        )
        self.source_colname2_df = PropertyDescriptor(
            name = 'Source DataFrame Colname2',
            description = 'Specify the source column name for holding the incoming paths of 3D voxel id to prepped voxel file mappings.',
            default_value="voxid_to_prep_vox",
            required = True
        )
        self.target_colname1_df = PropertyDescriptor(
            name = 'Target DataFrame Colname1',
            description = 'Specify the target column name for holding the destination paths of 3D voxel id to preprocessed caption mappings. For ex: one voxel ID to one prepped caption.',
            default_value="voxid_to_prep_caption",
            required = True
        )
        self.target_colname2_df = PropertyDescriptor(
            name = 'Target DataFrame Colname2',
            description = 'Specify the target column name for holding the destination paths of 3D voxel id to prepped voxel file mappings.',
            default_value="voxid_to_prep_vox",
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Mapped Prep Voxels to Prep Captions',
            description = 'If Mapped Prepped Voxels to Prepped Captions Already Performed, then just get filepaths',
            default_value=False,
            required = False,
        )
        self.data_type = PropertyDescriptor(
            name = 'Dataset Name',
            description = 'The name of the Dataset, currently supported: { icpsr_stroke}.',
            default_value = "icpsr_stroke",
            required=True
        )
        self.cap_del_pattern = PropertyDescriptor(
            name = 'Delete Caption Characters Pattern',
            description = 'Provide a Regex Pattern to Delete Caption Characters. By default, deletes digits, special characters & other chars not in the alphabet',
            default_value = "[^A-Za-z]",
            required=True
        )
        self.descriptors = [self.prep_captions_dir, self.prep_voxels_dir, self.source_colname1_df, self.source_colname2_df, self.target_colname1_df, self.target_colname2_df, self.already_prepped, self.data_type, self.cap_del_pattern]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for prep_captions_dirpath, etc")
        self.voxid_to_prep_captions_map = list()
        self.prep_captions_dirpath = context.getProperty(self.prep_captions_dir.name).getValue()
        self.prep_voxels_dirpath = context.getProperty(self.prep_voxels_dir.name).getValue()
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

    def map_prep_voxel_to_prep_caption(self, vox_cap_csv_data):
        self.logger.info("Doing the Captions Preprocessing and Mapping Prepped Voxel to Prepped Captions")
        self.mkdir_prep_dir(self.prep_captions_dirpath)
        self.mkdir_prep_dir(self.prep_voxels_dirpath)

        self.logger.info("self.cap_prep_already_done type = {}".format(type(self.cap_prep_already_done)))
        if self.cap_prep_already_done:
            self.logger.info("Adding Prepped Captions filepaths to data df in prep_torch")

            self.voxid_to_prep_captions_map = [self.prep_captions_dirpath + os.sep + self.data_name + "_" + str(i) + ".pk1" for i in range(len(vox_cap_csv_data))]
            vox_cap_csv_data[self.target_colname1] = self.voxid_to_prep_captions_map
            self.logger.info("Retrieved Prepped Captions filepaths stored at : {}/".format(self.prep_captions_dirpath))
        else:
            self.logger.info("Doing the Captions Preprocessing and Mapping Prepped Voxel to Prepped Captions From Scratch")
            # 1st col is prep voxel filepaths, 2nd col is voxid to prep captions filepaths
            new_voxid_prep_voxel_caps_df = pd.DataFrame(columns = [self.target_colname2, self.target_colname1])

            for i in range(len(vox_cap_csv_data)):
                voxid_to_caps = None
                voxid_to_prep_caption_set = ()

                with open(vox_cap_csv_data[self.source_colname2].iloc[i], "rb") as file:
                    voxid_to_prepvoxs = pickle.load(file)

                voxid_from_prepvox = list(voxid_to_prepvoxs.keys())[0]
                voxid_prepvoxel_filepath = list(voxid_to_prepvoxs.values())[0]
                self.logger.info(f"voxid_from_prepvox = {voxid_from_prepvox}")
                self.logger.info(f"voxid_prepvoxel_filepath = {voxid_prepvoxel_filepath}")


                self.logger.info("vox_cap_csv_data[\"{}\"].iloc[i] = {}".format(self.source_colname1, vox_cap_csv_data[self.source_colname1].iloc[i]))
                with open(vox_cap_csv_data[self.source_colname1].iloc[i], "rb") as file:
                    voxid_to_caps = pickle.load(file)
                
                    self.logger.info("check1: voxid_to_caps len = {}".format(len(voxid_to_caps)))
                    self.logger.info("voxid_to_caps[0] voxid_from_caps = {}".format(voxid_to_caps[0]))
                    self.logger.info("voxid_to_caps[1] captions_str = {}".format(voxid_to_caps[1]))

                self.logger.info("check2: voxid_to_caps len = {}".format(len(voxid_to_caps)))

                # voxid_from_caps = list(voxid_to_caps.keys())[0]
                # clinical_caption_list = list(voxid_to_caps.values())[0]
                # clinical_label = clinical_caption_tuple[0]
                # captions_str = clinical_caption_tuple[1]

                voxid_from_caps = voxid_to_caps[0]
                clinical_caption_list = voxid_to_caps[1]

                if voxid_from_caps in voxid_from_prepvox:
                    self.logger.info("Voxel ID from Prepped voxel matches Voxel ID from captions")

                    voxid_to_prepvoxel_set = (voxid_from_caps, voxid_prepvoxel_filepath)

                    voxid_to_prepvoxel_bytes = pickle.dumps(voxid_to_prepvoxel_set)

                    output_voxid_prepvoxel_path = os.path.join(self.prep_voxels_dirpath, voxid_from_caps + "_" + str(i) + ".pk1")
                    self.logger.info("Prepped voxel being saved with voxid_from_caps in filename {}".format(output_voxid_prepvoxel_path))
                    try:
                        with open(output_voxid_prepvoxel_path, "wb") as file:
                            file.write(voxid_to_prepvoxel_bytes)
                        self.logger.info("Prepped voxel with voxid_from_caps in filename pickle bytes saved successfully!")
                    except Exception as e:
                        self.logger.error("An error occurred while saving Prepped voxel with voxid_from_caps in filename!!!: {}".format(e))


                    for idx in range(len(clinical_caption_list)):
                        clinical_caption_tuple = clinical_caption_list[idx]
                        clinical_label = clinical_caption_tuple[0]
                        caption = clinical_caption_tuple[1]
                        # Convert to lowercase
                        caption = caption.lower()

                        # delete digits, special cahrs, etc
                        caption = caption.replace(self.cap_del_regex_pattern, '')

                        # delete additional spaces
                        # caption = caption.replace('\s+', ' ')
                        caption = re.sub('\s+', ' ', caption)

                        voxid_to_prep_caption_dict = {}
                        voxid_to_prep_caption_dict[voxid_from_caps] = (clinical_label, caption)

                        voxid_to_prep_caption_bytes = pickle.dumps(voxid_to_prep_caption_dict)

                        output_voxid_prepcap_path = os.path.join(self.prep_captions_dirpath, self.data_name + "_" + str(i) + "clinical_cap_" + str(idx) + ".pk1")
                        self.logger.info("Mapped 3D Voxel IDs to Preprocessed Captions pickle output_path = {}".format(output_voxid_prepcap_path))
                        try:
                            with open(output_voxid_prepcap_path, "wb") as file:
                                file.write(voxid_to_prep_caption_bytes)
                            self.logger.info("Mapped 3D Voxel IDs to Preprocessed Captions pickle bytes saved successfully!")
                        except Exception as e:
                            self.logger.error("An error occurred while saving Mapped 3D Voxel IDs to Preprocessed Captions!!!: {}".format(e))

                        new_voxid_prep_voxel_caps_df.loc[len(new_voxid_prep_voxel_caps_df)] = [output_voxid_prepvoxel_path, output_voxid_prepcap_path]
                        self.logger.info("Wrote new row of voxid prep voxel filepath and voxid prep caption filepath to pd")
                else:
                    self.logger.info("WARNING: Voxel ID from Prepped voxel unequal to Voxel ID from prep captions")

        self.logger.info("Preprocessed Captions, then Mapped Prepped Voxel to Prepped Captions")
        self.logger.info(f"Stored Voxel ID to Prepped Voxels in: {self.prep_voxels_dirpath}")
        self.logger.info(f"Stored Voxel ID to Prepped Captions in: {self.prep_captions_dirpath}")

        return new_voxid_prep_voxel_caps_df

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        vox_cap_csv_pd = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: vox_cap_csv_pd = {}".format(vox_cap_csv_pd.head()))
        vox_cap_csv_pd_up = self.map_prep_voxel_to_prep_caption(vox_cap_csv_pd)
        self.logger.info("output: vox_cap_csv_pd_up = {}".format(vox_cap_csv_pd_up.head()))

        # Create a StringIO object
        vox_cap_csv_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        vox_cap_csv_pd_up.to_csv(vox_cap_csv_string_io, index=False)

        # Get the string value and encode it
        vox_cap_csv_pd_up_string = vox_cap_csv_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = vox_cap_csv_pd_up_string)
