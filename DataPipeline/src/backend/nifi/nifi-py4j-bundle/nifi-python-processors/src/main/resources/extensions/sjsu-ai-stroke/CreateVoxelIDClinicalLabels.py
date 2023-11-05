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
import csv
import pandas as pd

import torch
import torchvision
import pickle5 as pickle

# from tqdm import tqdm
from nifiapi.properties import PropertyDescriptor, StandardValidators
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult


# TODO (JG): Limitation in flow is flow file not passed to next processor until processor finishes work. This is with each processor like this

# TODO (JG): Make this work for training and testing sets
class CreateVoxelIDClinicalLabels(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['pandas==1.3.5', 'pillow==10.0.0', 'pickle5==0.0.11', "torch", "torchvision", "torchaudio"]
        description = 'Gets ICPSR38464 participants tsv metadata or other clinical metadata, maps the voxel ID to one or more clinical labels (ex: medical-history, lesion-type, etc), creates a new participant clinical labels csv file, saves it to a new folder'
        tags = ['sjsu_ms_ai', 'csv', 'jpeg', 'nifti', 'pytorch']

    # TODO (JG): Add Property option for short caption and long caption
    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.icpsr_path = os.path.join('/media', 'bizon', 'ai_projects', 'data', 'ICPSR_38464_Stroke_Data')
        self.clinical_meta_source_path = PropertyDescriptor(
            name="Clinical Metadata Source Path",
            description="Clinical Metadata Source Path where ID, Age, Diagnosis, Medical-history, Lesion-type and other metadata can be parsed",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
            default_value="{}/MRI-Doc-Files/pkg38464-0001_documentation_REST/documentation/participants.tsv".format(self.icpsr_path),
            required=True)
        self.voxid_clinical_label_file = PropertyDescriptor(
            name = 'Voxel ID Clinical Labels File Destination Path',
            description = 'The folder to store the new Voxel ID Clinical Labels file, which contains Voxel ID mapped to one or more Clinical Labels (ex: Medical-history, Lesion-type, etc)',
            default_value="{}_NiFi/annotations/{}".format(self.icpsr_path, "partipant_id_clinical_labels.csv"),
            required = True
        )
        # Label
        self.voxel_id_colname = PropertyDescriptor(
            name = 'Voxel ID Column Name in Clinical Metadata',
            description = 'From the Clinical Metadata Source file, choose the Voxel ID column name. Keep in mind, we\'ll later be associating this Voxel ID with target clinical labels. For example, the voxel ID column name will have filenames of the 3D voxels, etc',
            default_value="participant_id",
            required = True
        )
        # Input Feature(s)
        self.target_clinical_label_colname = PropertyDescriptor(
            name = 'Target Clinical Column Name for Voxel ID',
            description = 'From the Clinical Metadata Source file, choose the target clinical label by column name you want to be associated with Voxel IDs. For example, one Voxel ID to one or more clinical label values that come from the target column',
            default_value="Medical-history",
            required = True
        )
        self.target_clinical_label_demarcator = PropertyDescriptor(
            name = 'Target Clinical Label Demarcator',
            description = 'From the Clinical Metadata Source file, choose the target clinical label demarcator. For example, one clinical label string may have forward slash "/" separators for multiple labels within string just one label per string',
            default_value="NA",
            required = True
        )
        self.source_colname_df = PropertyDescriptor(
            name = 'Source Colname from Input Prepped DataFrame',
            description = 'Specify the source column name for holding the incoming voxel IDs from voxel clinical label prepped df, so we filter only those voxel IDs in the voxel ID to clinical label mappings',
            default_value="brain_dwi_orig",
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Created Voxel ID Clinical Labels File',
            description = 'If Voxel ID Clinical Labels File Already Created, then no creation will happen since its already done, run the processor',
            default_value=False,
            required = True,
        )
        self.data_type = PropertyDescriptor(
            name = 'Dataset Name',
            description = 'The name of the Dataset, currently supported: { icpsr_stroke }.',
            default_value = "icpsr_stroke",
            required=True
        )
        self.descriptors = [self.clinical_meta_source_path, self.voxid_clinical_label_file, self.voxel_id_colname, self.target_clinical_label_colname, self.target_clinical_label_demarcator, self.source_colname_df, self.already_prepped, self.data_type]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for clinical_meta_source_filepath, etc")
        self.clinical_meta_source_filepath = context.getProperty(self.clinical_meta_source_path.name).getValue()
        self.voxid_clinical_label_filepath = context.getProperty(self.voxid_clinical_label_file.name).getValue()
        self.vox_id_colname = context.getProperty(self.voxel_id_colname.name).getValue() # Label
        self.target_clinical_colname = context.getProperty(self.target_clinical_label_colname.name).getValue() # Feature
        self.target_clinical_demarcator = context.getProperty(self.target_clinical_label_demarcator.name).getValue()
        self.prep_df_source_colname = context.getProperty(self.source_colname_df.name).getValue()
        self.labelfile_already_done = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.data_name = context.getProperty(self.data_type.name).getValue()
        self.voxid_seg_label_dirpath, self.voxid_clinical_label_filename = os.path.split(self.voxid_clinical_label_filepath)

    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    # TODO (JG): Finish
    def create_voxid_clinical_label_file(self, vox_seg_csv_pd):
        self.logger.info("Creating Voxel ID Clinical Label File")
        self.mkdir_prep_dir(self.voxid_seg_label_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.labelfile_already_done type = {}".format(type(self.labelfile_already_done)))
        if self.labelfile_already_done:
            self.logger.info("Already created Voxel ID Clinical Label file in voxid_seg_label_dirpath")
            imgid_cap_label_df = pd.read_csv(self.voxid_clinical_label_filepath)
        else:
            self.logger.info("Doing the Parsing from participants tsv to extract Voxel ID to Clinical Labels From Scratch")

            if self.data_name == "icpsr_stroke":
                input_df = pd.read_csv(self.clinical_meta_source_filepath, delimiter='\t')

            # elif self.data_name == "atlas":
            #     input_image = sitk.ReadImage(img_cap_csv_data.train_t1w_raw.iloc[i], sitk.sitkFloat32)

            # TODO (JG): When creating voxel ID list based on prepped DF, then filtering participant IDs
            # out of clinical table, so we only keep participant IDs that match with our voxel IDs, I could
            # make sure that each participant ID is organized in a way that matches the order of voxel ID
            voxid_prep_df_list = []
            for i in range(len(vox_seg_csv_pd)):
                # brain_dwi_orig
                voxel_filename = os.path.basename(vox_seg_csv_pd[self.prep_df_source_colname].iloc[i])
                long_voxel_id = voxel_filename.split(".")[0]
                voxid_prep_df_list.append(long_voxel_id)


            with open(self.voxid_clinical_label_filepath, "w", newline="") as file:
                writer = csv.writer(file)

                # just for icpsr_stroke data
                for i in range(len(input_df)):
                    # if medical_history is not NaN, write rows mapping participant_id to medical_history
                    if not pd.isnull(input_df[self.target_clinical_colname].iloc[i]):
                        # TODO (JG): Check if short caption, then do following, else long caption, map short medical history to long one
                        participant_id = input_df[self.vox_id_colname].iloc[i]
                        if any(participant_id in long_voxel_id for long_voxel_id in voxid_prep_df_list):
                            self.logger.info("participant_id is in prepped df = {}".format(participant_id))
                            clinical_label_str = input_df[self.target_clinical_colname].iloc[i]
                            self.logger.info("type = {}; clinical_label_str = {}".format(type(clinical_label_str), clinical_label_str))

                            if self.target_clinical_demarcator == "NA":
                                writer.writerow([participant_id, clinical_label])
                            elif self.target_clinical_demarcator in clinical_label_str:
                                clinical_label_list = clinical_label_str.split(self.target_clinical_demarcator)

                                for clinical_label in clinical_label_list:
                                    writer.writerow([participant_id, clinical_label])
                        else:
                            self.logger.info("WARN: participant_id isnt in prepped df = {}".format(participant_id))

            # imgid_cap_label_df = pd.read_csv(self.voxid_clinical_label_filepath)

        self.logger.info("Voxel ID Clinical Label CSV file stored at: {}/".format(self.voxid_seg_label_dirpath))
        # return imgid_cap_label_df

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        # Leaving in the retrieval of incoming flow file, expecting some table, we may use later, but we dont use it now
        vox_seg_csv_pd = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: vox_seg_csv_pd = {}".format(vox_seg_csv_pd.head()))

        # Create the voxel captions csv file, pass on same prepped df though. In next processor, we'll pass source captions filepath
        self.create_voxid_clinical_label_file(vox_seg_csv_pd)
        # self.logger.info("output: imgid_cap_label_pd = {}".format(imgid_cap_label_pd.head()))

        # Create a StringIO object
        vox_seg_csv_pd_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        vox_seg_csv_pd.to_csv(vox_seg_csv_pd_string_io, index=False)

        # Get the string value and encode it
        vox_seg_csv_pd_string = vox_seg_csv_pd_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = vox_seg_csv_pd_string)
