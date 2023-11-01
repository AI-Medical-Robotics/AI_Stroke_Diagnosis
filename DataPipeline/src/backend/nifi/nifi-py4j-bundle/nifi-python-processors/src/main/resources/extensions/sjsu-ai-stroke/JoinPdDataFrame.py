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

# import torch
# import torchvision
# import pickle5 as pickle

# from tqdm import tqdm
from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

# Verified we can run SimpleITK N4 Bias Field Correction and produces expected results faster than nipype's version

# TODO (JG): Limitation in flow is flow file not passed to next processor until processor finishes work. This is with each processor like this

# TODO (JG): Make this work for training and testing sets
class JoinPdDataFrame(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['pandas==1.3.5']
        description = 'Gets 2 pandas csv dataframes from 2 flow files, joins them on a common column and passes joined dataframe onward'
        tags = ['sjsu_ms_ai', 'csv', 'jpeg', 'pytorch']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.col_join_df_by = PropertyDescriptor(
            name = 'Join DataFrames On Column',
            description = 'Specify the column name to join the pandas dataframes by',
            default_value="natural_image",
            required = True
        )
        self.data_type = PropertyDescriptor(
            name = 'Dataset Name',
            description = 'The name of the Dataset, currently supported: { flickr }.',
            default_value = "flickr",
            required=True
        )
        self.pandas_df_title1 = PropertyDescriptor(
            name = 'Pandas DataFrame Name1',
            description = 'The name of the Pandas DataFrame that will be used in the join.',
            default_value = "map_imgid_to_caps",
            required=True
        )
        self.pandas_df_title2 = PropertyDescriptor(
            name = 'Pandas DataFrame Name2',
            description = 'The name of the Pandas DataFrame that will be used in the join.',
            default_value = "map_imgid_to_fets",
            required=True
        )
        self.descriptors = [self.col_join_df_by, self.data_type, self.pandas_df_title1, self.pandas_df_title2]
        self.counter = 0
        self.img_cap_csv_pd = None
        self.img_fet_csv_pd = None

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for col_join_df_on, etc")
        # read pre-trained model and config file
        self.col_join_df_on = context.getProperty(self.col_join_df_by.name).getValue()
        self.data_name = context.getProperty(self.data_type.name).getValue()
        self.pandas_df_name1 = context.getProperty(self.pandas_df_title1.name).getValue()
        self.pandas_df_name2 = context.getProperty(self.pandas_df_title2.name).getValue()


    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    # TODO (JG): Finish
    def join_dataframes_on_col(self, img_cap_csv_data, img_fet_csv_data):
        self.logger.info("Joining Image Captions DF & Image Features DF")

        joined_df = pd.merge(img_cap_csv_data, img_fet_csv_data, on=self.col_join_df_on)
        
        # Drop any rows not in both source data frames
        joined_df.dropna(inplace=True)

        return joined_df

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        pd_df_name = flowFile.getAttribute('pandas_df_name')
        # TODO (JG): Add 2 pandas dataframe names to check
        if pd_df_name == self.pandas_df_name1:
            if self.img_cap_csv_pd is None:
                self.img_cap_csv_pd = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
                self.logger.info("input: img_cap_csv_pd = {}".format(self.img_cap_csv_pd.head()))
                self.counter += 1

        elif pd_df_name == self.pandas_df_name2:
            if self.img_fet_csv_pd is None:
                self.img_fet_csv_pd = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
                self.logger.info("input: img_fet_csv_pd = {}".format(self.img_fet_csv_pd.head()))
                self.counter += 1

        if self.counter == 2:
            if self.img_cap_csv_pd is not None and self.img_fet_csv_pd is not None:
                joined_imgids_cap_fet_df = self.join_dataframes_on_col(self.img_cap_csv_pd, self.img_fet_csv_pd)
                self.logger.info("output: joined_imgids_cap_fet_df = {}".format(joined_imgids_cap_fet_df.head()))

                # Create a StringIO object
                joined_imgids_cap_fet_string_io = io.StringIO()

                # Use the to_csv() method to write to CSV
                joined_imgids_cap_fet_df.to_csv(joined_imgids_cap_fet_string_io, index=False)

                # Get the string value and encode it
                joined_imgids_cap_fet_df_string = joined_imgids_cap_fet_string_io.getvalue().encode("utf-8")

                self.img_cap_csv_pd = None
                self.img_fet_csv_pd = None
                self.counter = 0

                return FlowFileTransformResult(relationship = "success", contents = joined_imgids_cap_fet_df_string)
            
        return FlowFileTransformResult(relationship = "failure", contents = "waiting on 2 pandas dataframes to be joined")
