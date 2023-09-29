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
class CreateImageIDCaptionLabels(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['pandas==1.3.5', 'pillow==10.0.0', 'pickle5==0.0.11', "torch", "torchvision", "torchaudio"]
        description = 'Gets ICPSR38464 participants tsv metadata, maps the image ID to one or more medical-history captions, creates a new participant captions csv file, saves it to a new folder'
        tags = ['sjsu_ms_ai', 'csv', 'jpeg', 'nifti', 'pytorch']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.icpsr_path = os.path.join('/media', 'ubuntu', 'ai_projects', 'data', 'ICPSR_38464_Stroke_Data')
        self.participants_source_path = PropertyDescriptor(
            name="Participants Metadata Source Path",
            description="ICPSR Participants Metadata Source Path where ID, Age, Diagnosis, Medical-history and other metadata can be parsed",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
            default_value="{}/MRI-Doc-Files/pkg38464-0001_documentation_REST/documentation/participants.tsv".format(self.icpsr_path),
            required=True)
        self.imgid_caption_label_file = PropertyDescriptor(
            name = 'Image ID Caption Labels File Destination Path',
            description = 'The folder to store the new Image ID Caption Labels file, which contains Image ID mapped to one or more Medical-history captions',
            default_value="{}_NiFi/annotations/{}".format(self.icpsr_path, "partipant_id_captions.csv"),
            required = True
        )
        # Label
        self.image_id_colname = PropertyDescriptor(
            name = 'Image ID Column Name',
            description = 'From the Participants Metadata Source file, choose the Image ID column name. Keep in mind, we\'ll later be associating this Image ID with target captions. For example, the image ID column name will have filenames of the 2D images or 3D voxels, etc',
            default_value="participant_id",
            required = True
        )
        # Input Feature(s)
        self.target_cap_label_colname = PropertyDescriptor(
            name = 'Target Caption Column Name for Image ID',
            description = 'From the Participants Metadata Source file, choose the target caption by column name you want to be associated with Image IDs. For example, one Image ID to one or more caption values that come from the target column',
            default_value="Medical-history",
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Created Image ID Caption Labels File',
            description = 'If Image ID Caption Labels File Already Created, then no creation will happen since its already done, run the processor',
            default_value=False,
            required = True,
        )
        self.data_type = PropertyDescriptor(
            name = 'Dataset Name',
            description = 'The name of the Dataset, currently supported: { icpsr_stroke }.',
            default_value = "icpsr_stroke",
            required=True
        )
        self.descriptors = [self.participants_source_path, self.imgid_caption_label_file, self.image_id_colname, self.target_cap_label_colname, self.already_prepped, self.data_type]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for participants_source_filepath, etc")
        # self.imgid_to_captions_files = list()
        # read pre-trained model and config file
        self.participants_source_filepath = context.getProperty(self.participants_source_path.name).getValue()
        self.imgid_caption_label_filepath = context.getProperty(self.imgid_caption_label_file.name).getValue()
        self.img_id_colname = context.getProperty(self.image_id_colname.name).getValue() # Label
        self.target_caption_colname = context.getProperty(self.target_cap_label_colname.name).getValue() # Feature
        self.labelfile_already_done = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.data_name = context.getProperty(self.data_type.name).getValue()
        self.imgid_cap_label_dirpath, self.imgid_caption_label_filename = os.path.split(self.imgid_caption_label_filepath)

    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    # TODO (JG): Finish
    def create_imgid_caption_label_file(self):
        self.logger.info("Creating Image ID Caption Label File")
        self.mkdir_prep_dir(self.imgid_cap_label_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.labelfile_already_done type = {}".format(type(self.labelfile_already_done)))
        if self.labelfile_already_done:
            self.logger.info("Already created ImageID Caption Label file in imgid_to_caption_dir")
            imgid_cap_label_df = pd.read_csv(self.imgid_caption_label_filepath)
        else:
            self.logger.info("Doing the Parsing from participants tsv to extract ImageID to Caption Labels From Scratch")

            if self.data_name == "icpsr_stroke":
                input_df = pd.read_csv(self.participants_source_filepath, delimiter='\t')

            # elif self.data_name == "atlas":
            #     input_image = sitk.ReadImage(img_cap_csv_data.train_t1w_raw.iloc[i], sitk.sitkFloat32)

            with open(self.imgid_caption_label_filepath, "w", newline="") as file:
                writer = csv.writer(file)

                # just for icpsr_stroke data
                for i in range(len(input_df)):
                    participant_id = input_df[self.img_id_colname].iloc[i]
                    medical_history_str = input_df[self.target_caption_colname].iloc[i]
                    medical_history_list = medical_history_str.split("/")

                    for med_hist_caption in medical_history_list:
                        writer.writerow([participant_id, med_hist_caption])

            imgid_cap_label_df = pd.read_csv(self.imgid_caption_label_filepath)

        self.logger.info("Image ID Caption Label CSV file stored at: {}/".format(self.imgid_cap_label_dirpath))
        return imgid_cap_label_df

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        # Leaving in the retrieval of incoming flow file, expecting some table, we may use later, but we dont use it now
        img_cap_csv_pd = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: img_cap_csv_pd = {}".format(img_cap_csv_pd.head()))

        imgid_cap_label_pd = self.create_imgid_caption_label_file()
        self.logger.info("output: imgid_cap_label_pd = {}".format(imgid_cap_label_pd.head()))

        # Create a StringIO object
        imgid_cap_label_pd_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        imgid_cap_label_pd.to_csv(imgid_cap_label_pd_string_io, index=False)

        # Get the string value and encode it
        imgid_cap_label_pd_string = imgid_cap_label_pd_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = imgid_cap_label_pd_string)
