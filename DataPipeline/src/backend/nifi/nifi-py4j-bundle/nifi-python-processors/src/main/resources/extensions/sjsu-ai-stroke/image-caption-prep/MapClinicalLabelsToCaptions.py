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

import json
from jsonpath_ng import parse

import pickle5 as pickle

# from tqdm import tqdm
from nifiapi.properties import PropertyDescriptor, StandardValidators
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult


# TODO (JG): Limitation in flow is flow file not passed to next processor until processor finishes work. This is with each processor like this

# TODO (JG): Make this work for training and testing sets
class MapClinicalLabelsToCaptions(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['pandas==1.3.5', 'pillow==10.0.0', 'pickle5==0.0.11', 'jsonpath_ng']
        description = 'Gets the clinical labels file of voxel IDs and clinical labels, then uses a medical dictionary to map clinical labels to short or long captions (ex: medical-history, lesion-type, etc), creates a new voxel ID to clinical labels to captions csv file, saves it to a new folder'
        tags = ['sjsu_ms_ai', 'csv', 'jpeg', 'nifti', 'jsonpath']

    # TODO (JG): Add Property option for short caption and long caption
    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.icpsr_path = os.path.join('/media', 'bizon', 'ai_projects', 'data', 'ICPSR_38464_Stroke_Data')
        self.dict_path = "{}/MRI-Doc-Files/pkg38464-0001_documentation_REST/documentation/participants.tsv".format(self.icpsr_path)
        self.clinical_labels_source_path = PropertyDescriptor(
            name="Clinical Labels File Source Path",
            description="Clinical Labels Source Path where Voxel ID and Clinical Label can be parsed",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
            default_value="{}_NiFi/annotations/{}".format(self.icpsr_path, "partipant_id_clinical_labels.csv"),
            required=True)
        self.clinical_captions_source_path = PropertyDescriptor(
            name="Clinical Captions File Source Path",
            description="Clinical Captions Source Path where our source Clinical Label can be mapped to short or long captions in a JSON dictionary",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
            default_value=f"{self.icpsr_path}/MRI-Doc-Files/pkg38464-0001_documentation_REST/documentation/dictionary.json",
            required=True)
        self.voxid_clinical_captions_file = PropertyDescriptor(
            name = 'Clinical Labels to Captions File Destination Path',
            description = 'The folder to store the new Voxel ID Clinical Labels to short or long captions file where captions can be for ex: Medical-history, Lesion-type, etc',
            default_value="{}_NiFi/annotations/{}".format(self.icpsr_path, "voxel_id_clinical_captions.csv"),
            required = True
        )
        self.clinical_label_colname = PropertyDescriptor(
            name = 'Column Name in Clinical Labels Source File',
            description = 'From the Clinical Labels Source file, choose the clinical label column name. Keep in mind, we\'ll later be mapping these clinical labels with target clinical captions from the dictionary json source file',
            default_value="Medical-history",
            required = True
        )
        self.target_captions_json_path = PropertyDescriptor(
            name = 'Target Captions Base JSON Dict Indexing',
            description = 'From the Clinical Captions JSON Source file, choose the target clinical caption base JSON Path you want to be associated with Clinical Labels. For ex, with this JSON Path, we will use the keys from our Clinical Labels to then extract each caption',
            default_value="Medical-history, Levels",
            required = True
        )
        # self.source_colname_df = PropertyDescriptor(
        #     name = 'Source Colname from Input Prepped DataFrame',
        #     description = 'Specify the source column name for holding the incoming voxel IDs from voxel clinical label prepped df, so we filter only those voxel IDs in the voxel ID to clinical label mappings',
        #     default_value="brain_dwi_orig",
        #     required = True
        # )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Mapped Clinical Labels To Captions',
            description = 'If Clinical Labels Mapped to Captions File Already Created, then no creation will happen since its already done, run the processor',
            default_value=False,
            required = True,
        )
        self.data_type = PropertyDescriptor(
            name = 'Dataset Name',
            description = 'The name of the Dataset, currently supported: { icpsr_stroke }.',
            default_value = "icpsr_stroke",
            required=True
        )
        self.descriptors = [self.clinical_labels_source_path, self.clinical_captions_source_path, self.voxid_clinical_captions_file, self.clinical_label_colname, self.target_captions_json_path, self.already_prepped, self.data_type]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for clinical_labels_source_filepath, etc")
        self.clinical_labels_source_filepath = context.getProperty(self.clinical_labels_source_path.name).getValue()
        self.clinical_captions_source_filepath = context.getProperty(self.clinical_captions_source_path.name).getValue()
        self.voxid_clinical_captions_filepath = context.getProperty(self.voxid_clinical_captions_file.name).getValue()
        self.clinical_label_column_name = context.getProperty(self.clinical_label_colname.name).getValue() # Label
        self.target_captions_base_json_path = context.getProperty(self.target_captions_json_path.name).getValue() # Feature
        # self.target_clinical_demarcator = context.getProperty(self.target_clinical_label_demarcator.name).getValue()
        # self.prep_df_source_colname = context.getProperty(self.source_colname_df.name).getValue()
        self.labelfile_already_done = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.data_name = context.getProperty(self.data_type.name).getValue()
        self.voxid_seg_label_dirpath, self.voxid_clinical_captions_filename = os.path.split(self.voxid_clinical_captions_filepath)
        self.csv_header = ["voxel_id", "clinical_label", "caption"]

    def load_labels_data(self):
        with open(self.clinical_labels_source_filepath, "r") as f:
            next(f)
            clinical_labels_doc = f.read()
        return clinical_labels_doc

    def load_json_data(self):
        with open(self.clinical_captions_source_filepath) as json_file:
            clinical_captions_dict = json.load(json_file)
        return clinical_captions_dict

    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    # TODO (JG): Finish
    def create_voxid_clinical_captions_file(self, vox_seg_csv_pd):
        self.logger.info("Mapping Clinical Label To Captions File")
        self.mkdir_prep_dir(self.voxid_seg_label_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.labelfile_already_done type = {}".format(type(self.labelfile_already_done)))
        if self.labelfile_already_done:
            self.logger.info("Already Mapped Clinical Label To Captions file in voxid_seg_label_dirpath")
            imgid_cap_label_df = pd.read_csv(self.voxid_clinical_captions_filepath)
        else:
            self.logger.info("Mapping Clinical Labels To Captions From Dictionary JSON")

            if self.data_name == "icpsr_stroke":
                input_df = pd.read_csv(self.clinical_labels_source_filepath, delimiter='\t')

                clinical_labels_data = self.load_labels_data()

                clinical_captions_dict = self.load_json_data()

                # remove any whitespaces
                json_dict_indexing = self.target_captions_base_json_path.replace(" ", "")
                self.logger.info(f"cleaned json_dict_indexing = {json_dict_indexing}")
                json_index_tokens = json_dict_indexing.split(",")

            with open(self.voxid_clinical_captions_filepath, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(self.csv_header)

                for clinical_label in clinical_labels_data.split("\n"):
                    tokens = clinical_label.split(",")

                    voxel_id, label = tokens[0], tokens[1:]
                    label_str = " ".join(label)
                    self.logger.info(f"Clinical Label: {label}")
                    self.logger.info(f"Clinical Label Str: {label_str}")


                    # self.logger.info(f"JSONPath Expression: {self.target_captions_base_json_path}.{label_str}")
                    # jsonpath_expression = parse(f"{self.target_captions_base_json_path}.{label_str}")

                    # label_caption_matches = jsonpath_expression.find(clinical_captions_dict)

                    self.logger.info(f"JSON Index Tokens: {json_index_tokens}")
                    captions_dict = clinical_captions_dict[json_index_tokens[0]][json_index_tokens[1]]
                    if label_str in captions_dict:
                        self.logger.info(f"{label_str} is in captions_dict")
                        target_caption = captions_dict[label_str]
                        self.logger.info(f"Label: {label_str} => {target_caption}")

                        writer.writerow([voxel_id, label_str, target_caption])
                    else:
                        self.logger.info(f"WARNING: key {label_str} didnt matched in captions_dict, ignoring")
                    # NOTE (JG): Trying to see if I can take substring of key to use to extract Dict key value, but for now ignore
                    # elif any(label_str in caption_key for caption_key in captions_dict):
                    #     self.logger.info(f"{label_str} substring is in captions_dict, trying JSONPath expression to get key's value from captions_dict")
                    #     self.logger.info(f'Using JSONPath: $..*[?(@.key =~ /.*{label_str}.*/i)]')
                    #     jsonpath_expr = parse(f'$..*[?(@.key =~ /.*{label_str}.*/i)]')
                    #     caption_matches = jsonpath_expr.find(captions_dict)
                    #     for target_caption in caption_matches:
                    #         self.logger.info(f"Label: {label_str} => {target_caption}")
                    #         writer.writerow([voxel_id, label, target_caption])




        self.logger.info("Voxel ID Clinical Label CSV file stored at: {}/".format(self.voxid_seg_label_dirpath))

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        # Leaving in the retrieval of incoming flow file, expecting some table, we may use later, but we dont use it now
        vox_seg_csv_pd = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: vox_seg_csv_pd = {}".format(vox_seg_csv_pd.head()))

        # Create the voxel captions csv file, pass on same prepped df though. In next processor, we'll pass source captions filepath
        self.create_voxid_clinical_captions_file(vox_seg_csv_pd)
        # self.logger.info("output: imgid_cap_label_pd = {}".format(imgid_cap_label_pd.head()))

        # Create a StringIO object
        vox_seg_csv_pd_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        vox_seg_csv_pd.to_csv(vox_seg_csv_pd_string_io, index=False)

        # Get the string value and encode it
        vox_seg_csv_pd_string = vox_seg_csv_pd_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = vox_seg_csv_pd_string)
