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
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.properties import PropertyDescriptor, StandardValidators

class GetNFBSNIfTIFiles(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']

    class ProcessorDetails:
        dependencies = ['pandas==1.3.5']
        version = '0.0.1-SNAPSHOT'
        description = 'Gets NIfTI NFBS filepaths and loads them into a pandas csv dataframe to be accessible from a flow file'
        tags = ['sjsu_ms_ai', 'csv', 'nifti']

    def __init__(self, **kwargs):
        # property for nfbs base path
        self.dataset_base_path = PropertyDescriptor(name="NFBS Dataset Base Path",
            description="NFBS Dataset Base Path where origin is located",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
            default_value="{}/src/datasets/NFBS_Dataset".format(os.path.expanduser("~")),
            required=True)
        self.properties = [self.dataset_base_path]

    def onScheduled(self, context):
        # self.logger.info("onScheduled: Initializing Get NFBS NiFTiFiles Processor")
        self.brain_mask = list()
        self.brain = list()
        self.raw = list()

        self.nfbs_path = context.getProperty(self.dataset_base_path.name).getValue()
        # self.logger.info("nfbs_path = {}".format(self.nfbs_path))
        # self.logger.info("Getting NFBS Dataset Filepaths")
        for subdir, dirs, files in os.walk(self.nfbs_path):
            # self.logger.info("subdir = {}".format(subdir))
            for file in files:
                # self.logger.info("file = {}".format(os.path.join(subdir, file)))
                filepath = subdir + os.sep + file
                
                if filepath.endswith(".gz"):
                    if "_brainmask." in filepath:
                        self.brain_mask.append(filepath)
                    elif "_brain." in filepath:
                        self.brain.append(filepath)
                    else:
                        self.raw.append(filepath)

        # self.logger.info("Retrieved NFBS Dataset Filepaths")

        # self.logger.info("onScheduled: Creating pandas with NFBS Dataset Filepaths")
        self.nfbs_df = pd.DataFrame(
            {"brain_mask": self.brain_mask,
            "brain": self.brain,
            "raw": self.raw
            }
        )
    
        # self.logger.info("onScheduled: nfbs_df = {}".format(self.nfbs_df.head()))


    def transform(self, context, flowFile):
        if flowFile.getSize() >= 0:
            self.logger.info("flowFile size >= 0: Getting NFBS Base Path from Generate FlowFile")
            self.logger.info("nfbs_base_path = {}".format(io.BytesIO(flowFile.getContentsAsBytes())))
        self.logger.info("transform: nfbs_df = {}".format(self.nfbs_df.head()))

        # Create a StringIO object
        nfbs_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        self.nfbs_df.to_csv(nfbs_string_io, index=False)

        # Get the string value and encode it
        nfbs_csv_string = nfbs_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = nfbs_csv_string)


    def getPropertyDescriptors(self):
        return self.properties
