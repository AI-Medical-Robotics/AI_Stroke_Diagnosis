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

class GetFlickrJpegFiles(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']

    class ProcessorDetails:
        dependencies = ['pandas==1.3.5']
        version = '0.0.1-SNAPSHOT'
        description = 'Gets Flikr JPEG filepaths and loads them into a pandas csv dataframe to be accessible from a flow file'
        tags = ['sjsu_ms_ai', 'csv', 'jpeg']

    def __init__(self, **kwargs):
        # property for flickr base path
        self.dataset_source_path = PropertyDescriptor(name="Flickr Dataset Source Path",
            description="Flickr Dataset Source Path where images are located",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
            default_value="{}/src/datasets/flickr8k/images".format(os.path.expanduser("~")),
            required=True)
        self.properties = [self.dataset_source_path]

    def onScheduled(self, context):
        # self.logger.info("onScheduled: Initializing Get Flickr JPEG Files Processor")
        self.natural_image = list()

        self.flickr_img_path = context.getProperty(self.dataset_source_path.name).getValue()
        # self.logger.info("flickr_img_path = {}".format(self.flickr_img_path))
        # self.logger.info("Getting Flickr Dataset Filepaths")
        for subdir, dirs, files in os.walk(self.flickr_img_path):
            # self.logger.info("subdir = {}".format(subdir))
            for file in files:
                # self.logger.info("file = {}".format(os.path.join(subdir, file)))
                filepath = os.path.join(subdir, file)
                
                if filepath.endswith(".jpg"):
                    self.natural_image.append(filepath)

        # self.logger.info("Retrieved Flickr Dataset Filepaths")

        # self.logger.info("onScheduled: Creating pandas with Flickr Dataset Filepaths")
        self.flickr_df = pd.DataFrame(
            {"natural_image": self.natural_image}
        )
    
        # self.logger.info("onScheduled: flickr_df = {}".format(self.flickr_df.head()))


    def transform(self, context, flowFile):
        if flowFile.getSize() >= 0:
            self.logger.info("flowFile size >= 0: Getting Flickr Source Path from Generate FlowFile")
            self.logger.info("flickr_source_path = {}".format(io.BytesIO(flowFile.getContentsAsBytes())))
        self.logger.info("transform: flickr_df = {}".format(self.flickr_df.head()))

        # Create a StringIO object
        flickr_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        self.flickr_df.to_csv(flickr_string_io, index=False)

        # Get the string value and encode it
        flickr_csv_string = flickr_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = flickr_csv_string)


    def getPropertyDescriptors(self):
        return self.properties
