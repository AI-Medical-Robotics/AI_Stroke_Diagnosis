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
import pickle5 as pickle
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
        dependencies = ['pandas==1.3.5', 'tqdm==4.66.1', 'pyzmq==25.1.1', 'pickle5==0.0.11']
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
        # this might be a good optional property to show to the user, so they are aware
        self.socket.setsockopt(zmq.LINGER, 0) # After socket closes, discard pending messages in queue immediately
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

    # Reference perplexity.ai for logic to keep publishing until subscriber confirmation is received:
    # https://www.perplexity.ai/search/cf3ca7f0-a7f2-4395-8952-67198b09aef7?s=u
        # Alternative to while True loop, does NiFi Python have onTrigger like NiFi Java?
    def zmq_publish_data(self, nifti_csv_df):
        # keep publishing pandas dataframe until subscriber's confirmation message is received
        while True:
            try:
                self.logger.info("Performing ZMQ Publishing; Wait 10 seconds for ZMQ Subscriber Confirmation, else publishing")

                # Serialize the DataFrame using pickle
                nifti_csv_df_bytes = pickle.dumps(nifti_csv_df)

                # Publish the DataFrame bytes
                self.socket.send(nifti_csv_df_bytes)

                # Wait for 10 seconds to receive confirmation
                poller = zmq.Poller()
                poller.register(self.socket, zmq.POLLIN)
                if poller.poll(10000):
                    sub_confirmation = self.socket.recv_string(zmq.NOBLOCK)
                    if sub_confirmation == "Received":
                        self.logger.info("PutZMQ Publisher received Subscriber's confirmation, ending publish task")
                        self.logger.info("DataFrame ZMQ published successfully!")
                        break
                else:
                    print("No confirmation received, publishing again")
                    # time.sleep(0.1) # sleep short duration before checking again

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

        return FlowFileTransformResult(relationship = "success", contents = nifti_csv_df_bytes)
