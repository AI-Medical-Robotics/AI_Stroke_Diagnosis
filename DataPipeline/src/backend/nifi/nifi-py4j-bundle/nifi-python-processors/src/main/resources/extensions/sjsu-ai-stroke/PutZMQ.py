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
import zmq
import time
import pickle
import pandas as pd
from tqdm import tqdm
from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

# TODO (JG): Make this work for training and testing sets
class PutZMQ(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['pandas', 'tqdm', 'pyzmq', 'pickle']
        description = 'Gets the prepped NIfTI filepaths from the pandas csv dataframe in the flow file, sends the pandas dataframe over ZMQ'
        tags = ['sjsu_ms_ai', 'csv', 'zmq', 'nifti']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.zmq_socket_address = PropertyDescriptor(
            name = 'ZMQ Socket Address',
            description = 'The ZMQ Socket Address on where to Publish the data.',
            default_value="tcp://127.0.0.1:5555",
            required = True
        )
        self.nifti_data_type = PropertyDescriptor(
            name = 'NifTI Dataset Name',
            description = 'The name of the NifTI Dataset, currently supported: {nfbs, atlas, icpsr_stroke}.',
            default_value = "nfbs",
            required=True
        )
        self.descriptors = [self.zmq_socket_address, self.nifti_data_type]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for nifti_data_name, etc")
        self.nifti_data_name = context.getProperty(self.nifti_data_type.name).getValue()
        self.zmq_socket_addr = context.getProperty(self.zmq_socket_address.name).getValue()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(self.zmq_socket_addr)

    def onStopped(self, context):
        self.logger.info("Cleaning up PutZMQ")
        self.socket.close()

    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    def zmq_publish_data(self, nifti_csv_df):
        try:
            self.logger.info("Performing ZMQ Publishing")

            # Serialize the DataFrame using pickle
            nifti_csv_df_bytes = pickle.dumps(nifti_csv_df)

            # Publish the DataFrame bytes
            self.socket.send(nifti_csv_df_bytes)

            self.logger.info("DataFrame ZMQ published successfully!")
        except Exception as e:
            self.logger.error("Publishing DataFrame: {}".format(e))

        return nifti_csv_df_bytes

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        # NIfTI: — Neuroimaging Informatics Technology Initiative
        nifti_csv = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: nifti_csv = {}".format(nifti_csv.head()))

        nifti_csv_df_bytes = self.zmq_publish_data(nifti_csv)
        self.logger.info("output: nifti_csv_df_bytes len = {}".format(len(nifti_csv_df_bytes)))

        encoded_nifti_csv_df_bytes = nifti_csv_df_bytes.encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = encoded_nifti_csv_df_bytes)
