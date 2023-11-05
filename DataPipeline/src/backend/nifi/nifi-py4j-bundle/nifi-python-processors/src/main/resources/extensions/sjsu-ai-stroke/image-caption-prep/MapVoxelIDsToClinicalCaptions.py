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

class MapVoxelIDsToClinicalCaptions(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['pandas==1.3.5', 'pillow==10.0.0', 'pickle5==0.0.11', "torch", "torchvision", "torchaudio"]
        description = 'Gets 3D voxel captions txt metadata, performs dictionary mapping voxel ID to list of tuples clinical labels and captions, saves each dictionary element as pickle bytes and stores filepath to pandas dataframe'
        tags = ['sjsu_ms_ai', 'csv', 'nifti', 'pytorch']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.captions_source_path = PropertyDescriptor(name="Clinical Captions Source Path",
            description="Clinical Captions Source Path where 3D voxel IDs to captions txt metadata is located",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
            default_value="/media/bizon/projects_1/data/ICPSR_38464_Stroke_Data_NiFi/annotations/voxel_id_clinical_captions.csv",
            required=True
        )
        self.voxid_to_caption_dir = PropertyDescriptor(
            name = '3D Voxel ID Mappings to Captions Destination Path',
            description = 'The folder to store the 3D voxel IDs mapped to tuple of clinical label and captions',
            default_value="/media/bizon/projects_1/data/ICPSR_38464_Stroke_Data_NiFi/{}".format("map_voxids_to_clinical_captions"),
            required = True
        )
        self.target_colname_df = PropertyDescriptor(
            name = 'Target DataFrame Colname',
            description = 'Specify the target column name for holding the destination paths of voxel id to clinical caption mappings',
            default_value="voxid_to_clinical_captions",
            required = True
        )
        self.source_colname_df = PropertyDescriptor(
            name = 'Source Colname from Input Prepped DataFrame',
            description = 'Specify the source column name for holding the incoming voxel IDs from voxel clinical caption prepped df, so we filter only those voxel IDs in the voxel ID to clinical captions mappings',
            default_value="brain_dwi_orig",
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Mapped 3D Voxel IDs to Clinical Captions',
            description = 'If 3D Voxel IDs Mapped to Clinical Captions Already Performed, then just get filepaths',
            default_value=False,
            required = False,
        )
        self.data_type = PropertyDescriptor(
            name = 'Dataset Name',
            description = 'The name of the Dataset, currently supported: { icpsr_stroke }.',
            default_value = "icpsr_stroke",
            required=True
        )
        self.descriptors = [self.captions_source_path, self.voxid_to_caption_dir, self.target_colname_df, self.source_colname_df, self.already_prepped, self.data_type]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for voxid_to_caption_dirpath, etc")
        self.voxid_to_captions_files = list()
        # read pre-trained model and config file
        self.captions_source_filepath = context.getProperty(self.captions_source_path.name).getValue()
        self.voxid_to_caption_dirpath = context.getProperty(self.voxid_to_caption_dir.name).getValue()
        self.prep_df_source_colname = context.getProperty(self.source_colname_df.name).getValue()
        self.target_colname = context.getProperty(self.target_colname_df.name).getValue()
        self.img_map_already_done = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.data_name = context.getProperty(self.data_type.name).getValue()

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
    def map_imgids_to_captions(self, vox_cap_csv_data):
        self.logger.info("Mapping 3D Voxel IDs to Captions")
        voxid_to_caption_dir = self.mkdir_prep_dir(self.voxid_to_caption_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.img_map_already_done type = {}".format(type(self.img_map_already_done)))
        # if self.img_map_already_done:
            # self.logger.info("Adding Mapped IDs to Captions filepaths to data df in voxid_to_caption_dir")

            # self.voxid_to_captions_files = [voxid_to_caption_dir + os.sep + self.data_name + "_" + str(i) + ".pk1" for i in range(len(vox_cap_csv_data))]
            # vox_cap_csv_data[self.target_colname] = self.voxid_to_captions_files
            # self.logger.info("Retrieved Mapped IDs to Captions filepaths stored at : {}/".format(voxid_to_caption_dir))
        # else:
        self.logger.info("Doing the Mapped 3D Voxel IDs to Captions From Scratch")

        # if self.data_name == "flickr":
        #     cap_features_imgid_labels = self.load_caption_data()
            # elif self.data_name == "atlas":
            #     input_image = sitk.ReadImage(vox_cap_csv_data.train_t1w_raw.iloc[i], sitk.sitkFloat32)
        if self.data_name == "icpsr_stroke":
            # imgid_cap_label_df = pd.read_csv(self.captions_source_filepath)
            cap_features_imgid_labels = self.load_caption_data()

        # TODO (JG): 3 tokens: voxel_id, clinical_label, caption
        # key value pair: voxel_id: [(clinical_label, caption)]
        voxid_to_captions_map = {}
        for caption_label in cap_features_imgid_labels.split("\n"):

            tokens = caption_label.split(",")
            if len(caption_label) < 2:
                continue
                
            voxel_id, clinical_label, captions = tokens[0], tokens[1:2], tokens[2:]
            self.logger.info("voxel_id = {}".format(voxel_id))
            self.logger.info("clinical_label = {}".format(clinical_label))
            self.logger.info("captions = {}".format(captions))
            # voxel_id = voxel_id.split(".")[0]

            label_str = " ".join(clinical_label)

            captions_str = " ".join(captions)
            if voxel_id not in voxid_to_captions_map:
                voxid_to_captions_map[voxel_id] = list()
            voxid_to_captions_map[voxel_id].append( (label_str, captions_str) )

        # col_names = list(vox_cap_csv_data.columns).append(self.target_colname)
        col_names = list(vox_cap_csv_data.columns)
        self.logger.info("new_voxid_caps_df col_names = {}".format(col_names))
        new_voxid_caps_df = pd.DataFrame(columns = col_names)

        cap_pair_i = 0
        for idx, vox_cap_row in vox_cap_csv_data.iterrows():
            # get each long voxel ID in right order, check participant ID from captions map
            # brain_dwi_orig
            voxel_filename = os.path.basename(vox_cap_row[self.prep_df_source_colname])
            long_voxel_id = voxel_filename.split(".")[0]

            for voxid_to_captions_pair in voxid_to_captions_map.items():
                short_voxel_id = voxid_to_captions_pair[0]
                if short_voxel_id in long_voxel_id:
                    self.logger.info("In long_voxel_id, short_voxel_id substr found = {}".format(short_voxel_id))
                    # loop through dictionary, save each key value pair as bytes to filesystem
                    voxid_to_caption_bytes = pickle.dumps(voxid_to_captions_pair)

                    # Save the image name mapped torch preprocessed image
                    output_path = os.path.join(voxid_to_caption_dir, self.data_name + "_" + str(cap_pair_i) + ".pk1")
                    self.logger.info("Mapped 3D Voxel IDs to Captions pickle output_path = {}".format(output_path))
                    try:
                        with open(output_path, "wb") as file:
                            file.write(voxid_to_caption_bytes)
                        self.logger.info("Mapped 3D Voxel IDs to Captions pickle bytes saved successfully!")
                    except Exception as e:
                        self.logger.error("An error occurred while saving Mapped 3D Voxel IDs to Captions!!!: {}".format(e))

                    self.voxid_to_captions_files.append(output_path)
                    # brain_dwi_orig,brain_adc_orig,brain_bo_orig,brain_dwi_mask,stroke_dwi_mask,raw_index,skull_mask_index,stroke_mask_index,bias_corr,intensity_norm
                    new_voxid_caps_df.loc[len(new_voxid_caps_df)] = vox_cap_row

                    # new_voxid_caps_df.loc[len(new_voxid_caps_df)] = [vox_cap_row["brain_dwi_orig"],vox_cap_row["brain_adc_orig"],vox_cap_row["brain_bo_orig"],
                    #     vox_cap_row["brain_dwi_mask"], vox_cap_row["stroke_dwi_mask"],vox_cap_row["raw_index"],vox_cap_row["skull_mask_index"],vox_cap_row["stroke_mask_index"],
                    #     vox_cap_row["bias_corr"],vox_cap_row["intensity_norm"], output_path]
                    cap_pair_i += 1
                else:
                    self.logger.info("short_voxel_id substr not found in long voxel id, skipping, so we remain in order")

        # vox_cap_csv_data[self.target_colname] = self.voxid_to_captions_files
        new_voxid_caps_df[self.target_colname] = self.voxid_to_captions_files
        self.logger.info("Mapped 3D Voxel IDs to Captions pickle bytes stored at: {}/".format(voxid_to_caption_dir))
        return new_voxid_caps_df

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        vox_cap_csv_pd = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: vox_cap_csv_pd = {}".format(vox_cap_csv_pd.head()))
        vox_cap_csv_pd_up = self.map_imgids_to_captions(vox_cap_csv_pd)
        self.logger.info("output: vox_cap_csv_pd_up = {}".format(vox_cap_csv_pd_up.head()))

        # Create a StringIO object
        vox_cap_csv_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        vox_cap_csv_pd_up.to_csv(vox_cap_csv_string_io, index=False)

        # Get the string value and encode it
        vox_cap_csv_pd_up_string = vox_cap_csv_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = vox_cap_csv_pd_up_string)
