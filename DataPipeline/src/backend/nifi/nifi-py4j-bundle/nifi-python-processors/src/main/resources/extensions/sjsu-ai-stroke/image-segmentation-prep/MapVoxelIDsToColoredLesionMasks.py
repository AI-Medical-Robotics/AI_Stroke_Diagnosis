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
import numpy as np
import pandas as pd

import SimpleITK as sitk
import pickle5 as pickle

from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

# voxid_to_clinical_lesions, voxid_to_prep_vox, vox_id_to_lesion_mask

class MapVoxelIDsToColoredLesionMasks(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['pandas==1.3.5', 'pillow==10.0.0', 'pickle5==0.0.11', 'SimpleITK==2.2.1']
        description = 'Gets Mapped 3D Voxel IDs to Clinical Captions pickle bytes filepaths and Mapped 3D Voxel IDs to Prepped Voxels pickle bytes filepaths from the incoming pandas csv dataframe in the flow file, loads these mappings from pickle bytes files, performs multiple preprocessing operations (lowercasing; removing digits, special chars, white space) on each voxel ID\'s captions string, returning back an updated map of voxel ID to prepped captions, and then maps prepped voxels to prepped captions by voxel ID, storing these mappings to pickle bytes files and has the filepaths saved in a new pd df'
        tags = ['sjsu_ms_ai', 'csv', 'jpeg', 'nifti']

    def __init__(self, **kwargs):
        self.icpsr_path = os.path.join('/media', 'bizon', 'projects_1', 'data', 'ICPSR_38464_Stroke_Data_NiFi')
        self.prep_captions_dir = PropertyDescriptor(
            name = 'Clinical Lesion Labels Destination Path',
            description = 'The folder to store the Voxel IDs to Clinical Lesions.',
            default_value=f"{self.icpsr_path}/map_voxids_to_clinical_lesions",
            required = True
        )
        self.prep_voxels_dir = PropertyDescriptor(
            name = 'Preprocessed 3D Voxels Destination Path',
            description = 'The folder to store the preprocessed 3D voxels associated with voxel id in filename.',
            default_value=f"{self.icpsr_path}/voxid_prepvoxs_rel_color_lesions",
            required = True
        )
        self.color_masks_dir = PropertyDescriptor(
            name = 'Colored Lesion Masks Destination Path',
            description = 'The folder to store the preprocessed 3D voxels associated with voxel id in filename.',
            default_value=f"{self.icpsr_path}/voxid_color_lesion_masks",
            required = True
        )
        self.source_colname1_df = PropertyDescriptor(
            name = 'Source DataFrame Colname1',
            description = 'Specify the source column name for holding the incoming paths of 3D voxel id to captions mappings. For ex: one voxel id to one or more captions',
            default_value="voxid_to_clinical_lesions",
            required = True
        )
        self.source_colname2_df = PropertyDescriptor(
            name = 'Source DataFrame Colname2',
            description = 'Specify the source column name for holding the incoming paths of 3D voxel id to prepped voxel file mappings.',
            default_value="vox_id_to_lesion_mask",
            required = True
        )
        self.source_colname3_df = PropertyDescriptor(
            name = 'Source DataFrame Colname3',
            description = 'Specify the source column name for holding the incoming paths of 3D voxel id to prepped voxel file mappings.',
            default_value="voxid_to_prep_vox",
            required = True
        )
        self.target_colname1_df = PropertyDescriptor(
            name = 'Target DataFrame Colname1',
            description = 'Specify the target column name for holding the destination paths of 3D voxel id to preprocessed caption mappings. For ex: one voxel ID to one prepped caption.',
            default_value="voxid_to_clinical_lesions",
            required = True
        )
        self.target_colname2_df = PropertyDescriptor(
            name = 'Target DataFrame Colname2',
            description = 'Specify the target column name for holding the destination paths of 3D voxel id to prepped voxel file mappings.',
            default_value="voxid_to_colored_lesion_mask",
            required = True
        )
        self.target_colname3_df = PropertyDescriptor(
            name = 'Target DataFrame Colname3',
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
        self.stroke_lesion_color = PropertyDescriptor(
            name = 'Lesion Mask Color for Stroke Type',
            description = 'Choose the color for the lesion segmentation mask per stroke type',
            default_value = """{"ischemic": "green", "hemorrhagic": "blue"}""",
            required=True
        )
        self.expected_stroke_keys = ["ischemic", "hemorrhagic"]
        self.expected_stroke_colors = ["red", "green", "blue"]
        self.descriptors = [self.prep_captions_dir, self.prep_voxels_dir, self.color_masks_dir, self.source_colname1_df, self.source_colname2_df, self.source_colname3_df, self.target_colname1_df, self.target_colname2_df, self.target_colname3_df, self.already_prepped, self.data_type, self.stroke_lesion_color]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for prep_captions_dirpath, etc")
        self.voxid_to_prep_captions_map = list()
        self.prep_captions_dirpath = context.getProperty(self.prep_captions_dir.name).getValue()
        self.prep_voxels_dirpath = context.getProperty(self.prep_voxels_dir.name).getValue()
        self.color_masks_dirpath = context.getProperty(self.color_masks_dir.name).getValue()
        self.source_colname1 = context.getProperty(self.source_colname1_df.name).getValue()
        self.source_colname2 = context.getProperty(self.source_colname2_df.name).getValue()
        self.source_colname3 = context.getProperty(self.source_colname3_df.name).getValue()

        self.target_colname1 = context.getProperty(self.target_colname1_df.name).getValue()
        self.target_colname2 = context.getProperty(self.target_colname2_df.name).getValue()
        self.target_colname3 = context.getProperty(self.target_colname3_df.name).getValue()
        self.cap_prep_already_done = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.data_name = context.getProperty(self.data_type.name).getValue()
        self.stroke_lesion_color_json = context.getProperty(self.stroke_lesion_color.name).getValue()


    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    def map_prep_voxel_to_prep_caption(self, vox_cap_csv_data):
        self.logger.info("Going to Map Voxel IDs to Colored Lesion Masks")
        self.mkdir_prep_dir(self.prep_captions_dirpath)
        self.mkdir_prep_dir(self.prep_voxels_dirpath)
        self.mkdir_prep_dir(os.path.join(self.color_masks_dirpath, "nifti"))
        self.mkdir_prep_dir(os.path.join(self.color_masks_dirpath, "pickle_bytes"))

        self.logger.info("self.cap_prep_already_done type = {}".format(type(self.cap_prep_already_done)))
        if self.cap_prep_already_done:
            self.logger.info("Adding Map Voxel IDs to Colored Lesion Masks filepaths to data df in prep_torch")

            self.voxid_to_prep_captions_map = [self.prep_captions_dirpath + os.sep + self.data_name + "_" + str(i) + ".pk1" for i in range(len(vox_cap_csv_data))]
            vox_cap_csv_data[self.target_colname1] = self.voxid_to_prep_captions_map
            self.logger.info("Retrieved Prepped Captions filepaths stored at : {}/".format(self.prep_captions_dirpath))
        else:
            # 1st voxid_to_clinical_lesions
            # 2nd vox_id_to_colored_lesion_mask
            # 3rd voxid_to_prep_vox
            new_voxid_prep_voxel_caps_df = pd.DataFrame(columns = [self.target_colname1, self.target_colname2, self.target_colname3])

            # Extract lesion segmentation mask color for each stroke type from json
            stroke_color_dict = json.loads(self.stroke_lesion_color_json)
            for stroke_key in self.expected_stroke_keys:
                if stroke_key not in stroke_color_dict:
                    self.logger.error("The key '{}' is missing or incorrect in the JSON data.".format(stroke_key))


            # Sources:
            # voxid_to_clinical_lesions
            # vox_id_to_lesion_mask
            # voxid_to_prep_vox

            for i in range(len(vox_cap_csv_data)):
                voxid_to_caps = None
                voxid_to_prep_caption_set = ()

                with open(vox_cap_csv_data[self.source_colname2].iloc[i], "rb") as file:
                    voxid_to_lesion_mask = pickle.load(file)

                voxid_from_lesion_mask = list(voxid_to_lesion_mask.keys())[0]
                voxid_lesion_mask_filepath = list(voxid_to_lesion_mask.values())[0]
                self.logger.info(f"voxid_from_lesion_mask = {voxid_from_lesion_mask}")
                self.logger.info(f"voxid_lesion_mask_filepath = {voxid_lesion_mask_filepath}")

                # load simple itk for lesion mask
                input_lesion_mask_voxel = sitk.ReadImage(voxid_lesion_mask_filepath)
                # convert masks to numpy
                input_lesion_mask_array = sitk.GetArrayFromImage(input_lesion_mask_voxel)


                with open(vox_cap_csv_data[self.source_colname3].iloc[i], "rb") as file:
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

                if voxid_from_caps in voxid_from_prepvox and voxid_from_caps in voxid_from_lesion_mask:
                    self.logger.info("Voxel ID from Clinical Lesion Labels matches Voxel ID from Prepped Voxels and Lesion Masks")

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

                        if clinical_label == self.expected_stroke_keys[0]:
                            self.logger.info(f"clinical label = {clinical_label}, so applying GREEN to each 2D slice of lesion mask")
                            # ischemic, so apply green to each 2D slice of 3D lesion mask
                            if stroke_color_dict[clinical_label] == self.expected_stroke_colors[1]:
                                ischemic_color = (0, 255, 0) # Green
                                for idx in range(input_lesion_mask_array.shape[0]):
                                    self.logger.info("Get 2D slice as numpy")
                                    ischemic_slice = input_lesion_mask_array[idx, :, :]
                                    self.logger.info("Apply GREEN mask to 2D slice of 3D ITK lesion mask")
                                    ischemic_mask = np.stack([ischemic_slice]*3, axis=-1)
                                    ischemic_mask_3d = np.zeros_like(ischemic_mask)
                                    ischemic_mask_3d[ischemic_mask[:, :, 0] > 0] = ischemic_color
                                    ischemic_mask_2d = ischemic_mask_3d[:, :, 0]
                                    self.logger.info("Convert 2D slice back to a SimpleITK image")
                                    ischemic_color_slice = sitk.GetImageFromArray(ischemic_mask_2d)
                                    self.logger.info("Set origin and spacing of 2D slice ITK image to match original")
                                    ischemic_color_slice.SetOrigin(input_lesion_mask_voxel.GetOrigin())
                                    ischemic_color_slice.SetSpacing(input_lesion_mask_voxel.GetSpacing())
                                    self.logger.info("Insert 2D slice ITK image back into 3D ITK lesion mask")
                                    ischemic_color_slice = sitk.Paste(input_lesion_mask_voxel, ischemic_color_slice, ischemic_color_slice.GetSize(), (0, 0, i))
                            else:
                                self.logger.info("ischemic stroke color for lesion mask not supported: only green")
                            
                        elif clinical_label == self.expected_stroke_keys[1]:
                            self.logger.info(f"clinical label = {clinical_label}, so applying BLUE to each 2D slice of lesion mask")
                            # hemorrhagic, so apply blue to each 2D slice of 3D lesion mask
                            if stroke_color_dict[clinical_label] == self.expected_stroke_colors[2]:
                                hemorrhagic_color = (0, 0, 255) # Blue
                                for idx in range(input_lesion_mask_array.shape[0]):
                                    self.logger.info("Get 2D slice as numpy")
                                    hemorrhagic_slice = input_lesion_mask_array[idx, :, :]
                                    self.logger.info("Apply BLUE mask to 2D slice of 3D ITK lesion mask")
                                    hemorrhagic_mask = np.stack([hemorrhagic_slice]*3, axis=-1)
                                    hemorrhagic_mask_3d = np.zeros_like(hemorrhagic_mask)
                                    hemorrhagic_mask_3d[hemorrhagic_mask[:, :, 0] > 0] = hemorrhagic_color
                                    hemorrhagic_mask_2d = hemorrhagic_mask_3d[:, :, 0]
                                    self.logger.info("Convert 2D slice back to a SimpleITK image")
                                    hemorrhagic_color_slice = sitk.GetImageFromArray(hemorrhagic_mask_2d)
                                    self.logger.info("Set origin and spacing of 2D slice ITK image to match original")
                                    hemorrhagic_color_slice.SetOrigin(input_lesion_mask_voxel.GetOrigin())
                                    hemorrhagic_color_slice.SetSpacing(input_lesion_mask_voxel.GetSpacing())
                                    self.logger.info("Insert 2D slice ITK image back into 3D ITK lesion mask")
                                    hemorrhagic_color_slice = sitk.Paste(input_lesion_mask_voxel, hemorrhagic_color_slice, hemorrhagic_color_slice.GetSize(), (0, 0, i))
                            else:
                                self.logger.info("hemorrhagic stroke color for lesion mask not supported: only blue")


                        output_color_lesion_mask_path = os.path.join(self.color_masks_dirpath, "nifti", self.data_name + "_" + str(i) + "_color_lesion_mask_" + str(idx) + ".nii.gz")
                        self.logger.info(f"Saving Colored ITK Lesion Mask as NIfTI 3D voxel at: {output_color_lesion_mask_path}")
                        sitk.WriteImage(input_lesion_mask_voxel, output_color_lesion_mask_path)

                        self.logger.info(f"Creating 1 key value pair with Voxel ID to Colored Lesion Mask filepath")
                        voxid_to_color_lesion_mask = {}
                        voxid_to_color_lesion_mask[voxid_from_caps] = output_color_lesion_mask_path
                        voxid_to_color_lesion_mask_bytes = pickle.dumps(voxid_to_color_lesion_mask)

                        output_voxid_color_lesion_path = os.path.join(self.color_masks_dirpath, "pickle_bytes", self.data_name + "_" + str(i) + "voxid_color_lesion_mask_" + str(idx) + ".pk1")
                        self.logger.info("Mapped 3D Voxel IDs to Colored Lesion Masks pickle output_path = {}".format(output_voxid_color_lesion_path))
                        try:
                            with open(output_voxid_color_lesion_path, "wb") as file:
                                file.write(voxid_to_color_lesion_mask_bytes)
                            self.logger.info("Mapped 3D Voxel IDs to Colored Lesion Masks pickle bytes saved successfully!")
                        except Exception as e:
                            self.logger.error("An error occurred while saving Mapped 3D Voxel IDs to Colored Lesion Masks!!!: {}".format(e))


                        voxid_to_prep_caption_dict = {}
                        voxid_to_prep_caption_dict[voxid_from_caps] = (clinical_label, caption)

                        voxid_to_prep_caption_bytes = pickle.dumps(voxid_to_prep_caption_dict)

                        output_voxid_clinical_lesion_path = os.path.join(self.prep_captions_dirpath, self.data_name + "_" + str(i) + "clinical_lesion_" + str(idx) + ".pk1")
                        self.logger.info("Mapped 3D Voxel IDs to Preprocessed Captions pickle output_path = {}".format(output_voxid_clinical_lesion_path))
                        try:
                            with open(output_voxid_clinical_lesion_path, "wb") as file:
                                file.write(voxid_to_prep_caption_bytes)
                            self.logger.info("Mapped 3D Voxel IDs to Preprocessed Captions pickle bytes saved successfully!")
                        except Exception as e:
                            self.logger.error("An error occurred while saving Mapped 3D Voxel IDs to Preprocessed Captions!!!: {}".format(e))

                        # 1st voxid_to_clinical_lesions
                        # 2nd vox_id_to_colored_lesion_mask
                        # 3rd voxid_to_prep_vox
                        new_voxid_prep_voxel_caps_df.loc[len(new_voxid_prep_voxel_caps_df)] = [output_voxid_clinical_lesion_path, output_voxid_color_lesion_path, output_voxid_prepvoxel_path]
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
