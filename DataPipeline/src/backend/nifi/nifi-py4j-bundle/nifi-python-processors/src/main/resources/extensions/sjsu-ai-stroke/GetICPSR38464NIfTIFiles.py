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

class GetICPSR38464NIfTIFiles(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']

    class ProcessorDetails:
        dependencies = ['pandas==1.3.5']
        version = '0.0.1-SNAPSHOT'
        description = 'Gets NIfTI ICPSR 38464 filepaths and loads them into a pandas csv dataframe to be accessible from a flow file'
        tags = ['sjsu_ms_ai', 'csv', 'nifti']

    def __init__(self, **kwargs):
        # property for icpsr base path
        self.icpsr_path = os.path.join('/media', 'james', 'ai_projects', 'data', 'ICPSR_38464_Stroke_Data', 'MRI-DS-1-48')
        self.dataset_base_path = PropertyDescriptor(name="ICPSR38464 Dataset Base Path",
            description="ICPSR38464 Dataset Base Path where stroke MRI data is located",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
            default_value="{}".format(self.icpsr_path),
            required=True)
        self.properties = [self.dataset_base_path]

    def onScheduled(self, context):
        self.logger.info("Initilializing ICPSR38464 Dataset Filepath Lists")
        self.brain_dwi_orig = list()
        self.brain_dwi_mask = list()
 
        # Stroke Lesion Segmentation Files
        self.stroke_dwi_mask = list()

        self.logger.info("Getting ICPSR Base Path from Property")
        self.icpsr_base_path = context.getProperty(self.dataset_base_path.name).getValue()
        self.logger.info("icpsr_base_path = {}".format(self.icpsr_base_path))

    def transform(self, context, flowFile):
        if flowFile.getSize() >= 0:
            self.logger.info("flowFile size >= 0: Getting ICPSR38464 Base Path from Generate FlowFile")
            self.logger.info("icpsr_base_path = {}".format(io.BytesIO(flowFile.getContentsAsBytes())))

        self.logger.info("Getting ICPSR38464 Dataset Filepath Lists")
        for subdir, dirs, files in os.walk(self.icpsr_base_path):
            # self.logger.info("subdir = {}".format(subdir))
            brain_dwi_orig_single = list()
            brain_dwi_mask_single = list()
            stroke_dwi_mask_single = list()

            for file in files:
                filepath = subdir + os.sep + file
                
                if filepath.endswith(".gz"):
                    if "desc-brain_mask.nii.gz" in filepath:
                        brain_dwi_mask_single.append(filepath)
                        # self.brain_dwi_mask.append(filepath)
                        self.logger.info("filepath = {}".format(filepath))
                    elif "DWI_space-orig.nii.gz" in filepath:
                        brain_dwi_orig_single.append(filepath)
                        # self.brain_dwi_orig.append(filepath)
                        self.logger.info("filepath = {}".format(filepath))
                    elif "desc-stroke_mask.nii.gz" in filepath:
                        stroke_dwi_mask_single.append(filepath)
                        # self.stroke_dwi_mask.append(filepath)
                        self.logger.info("filepath = {}".format(filepath))

            if len(brain_dwi_mask_single) == 1 and len(brain_dwi_orig_single) == 1 and len(stroke_dwi_mask_single) == 1:
                self.brain_dwi_orig.extend(brain_dwi_orig_single)
                self.brain_dwi_mask.extend(brain_dwi_mask_single)
                self.stroke_dwi_mask.extend(stroke_dwi_mask_single)
            elif len(brain_dwi_mask_single) != 1:
                self.logger.info("brain_dwi_mask_single didnt equal 1; it equals = {}".format(len(brain_dwi_mask_single)))
            elif len(brain_dwi_orig_single) != 1:
                self.logger.info("brain_dwi_orig_single didnt equal 1; it equals = {}".format(len(brain_dwi_orig_single)))
            elif len(stroke_dwi_mask_single) != 1:
                self.logger.info("stroke_dwi_mask_single didnt equal 1; it equals = {}".format(len(stroke_dwi_mask_single)))
            else:
                self.logger.info("problem occurred when processing paths for brain_dwi_orig, brain_dwi_mask or stroke_dwi_mask")
            


        self.logger.info("Creating icpsr_df: brain_dwi_orig, brain_dwi_mask, stroke_dwi_mask")
        self.icpsr_df = pd.DataFrame(
            {"brain_dwi_orig": self.brain_dwi_orig,
            "brain_dwi_mask": self.brain_dwi_mask,
            "stroke_dwi_mask": self.stroke_dwi_mask
            }
        )

        self.logger.info("transform: icpsr_df = {}".format(self.icpsr_df.head()))

        # Create a StringIO object
        icpsr_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        self.icpsr_df.to_csv(icpsr_string_io, index=False)

        # Get the string value and encode it
        icpsr_csv_string = icpsr_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = icpsr_csv_string)


    def getPropertyDescriptors(self):
        return self.properties
