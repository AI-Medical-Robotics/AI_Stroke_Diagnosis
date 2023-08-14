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

import os
import pandas as pd
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

class GetATLASNIfTIFiles(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']

    class ProcessorDetails:
        dependencies = ['pandas']
        version = '0.0.1-SNAPSHOT'
        description = 'Gets NIfTI ATLAS 2.0 filepaths and loads them into a pandas csv dataframe to be accessible from a flow file'
        tags = ['sjsu_ms_ai', 'csv', 'nifti']

    def __init__(self, **kwargs):
        # Planning to have 2 properties for atlas_path and training or testing set

        # property for atlas base path
        self.atlas_path = "/media/bizon/ai_projects/ATLAS_2"
        # atlas_path = "/media/james/ai_projects/ATLAS_2"

        # property for training or testing set
        self.train_atlas_path = atlas_path + "/Training"
        self.test_atlas_path = atlas_path + "/Testing"

        self.train_lesion_mask = list()
        self.train_t1w_raw = list()
        self.test_t1w_raw = list()

        self.train_atlas_df = pd.DataFrame()
        self.test_atlas_df = pd.DataFrame()

    def load_training_paths():
        # Load training filepaths into list
        for subdir, dirs, files in os.walk(self.train_atlas_path):
            if len(files) == 2:
                lesion_mask = list()
                t1w_raw = list()
                for file in files:
                    # print(os.path.join(subdir, file))
                    filepath = subdir + os.sep + file

                    if filepath.endswith(".gz"):
                        if "label-L_desc-T1lesion_mask" in filepath:
                            lesion_mask.append(filepath)
                        elif "_T1w" in filepath:
                            t1w_raw.append(filepath)
        #         print("len(lesion_mask) = ", len(lesion_mask))
        #         print("len(t1w_raw) = ", len(t1w_raw))
                            
                if len(lesion_mask) == 1 and len(t1w_raw) == 1:
                    self.train_lesion_mask.extend(lesion_mask)
                    self.train_t1w_raw.extend(t1w_raw)
                    # self.train_t1w_lesion_mask_tuples.append((t1w_raw[0], lesion_mask[0]))
                        
            else:
                pass
        #         print("Skipping subdir {} since it doesnt have 2 files".format(subdir))
    
        print("len(train_lesion_mask) = ", len(self.train_lesion_mask))
        print("len(train_t1w_raw) = ", len(self.train_t1w_raw))

    def load_testing_paths():
        # store the address of 1 type of files
        for subdir, dirs, files in os.walk(self.test_atlas_path):
            for file in files:
                # print(os.path.join(subdir, file))
                filepath = subdir + os.sep + file
                
                if filepath.endswith(".gz"):
                    if "_T1w" in filepath:
                        self.test_t1w_raw.append(filepath)
        
        print("len(test_t1w_raw) = ", len(self.test_t1w_raw))

    def load_training_paths_in_df():
        load_training_paths()
        self.train_atlas_df = pd.DataFrame(
            {"train_lesion_mask": self.train_lesion_mask,
            "train_t1w_raw": self.train_t1w_raw
            }
        )

        print("train_t1w_raw in df = ", self.train_atlas_df.train_t1w_raw.iloc[0])
        print("train_lesion_mask in df = ", self.train_atlas_df.train_lesion_mask.iloc[0])


        print("Saving train atlas df to csv")
        self.train_atlas_df.to_csv("train_atlas_df.csv", index=False)

    def load_testing_paths_in_df():
        load_testing_paths()
        self.test_atlas_df = pd.DataFrame(
            {"test_t1w_raw": self.test_t1w_raw}
        )

        print("Saving test atlas df to csv")
        self.test_atlas_df.to_csv("test_atlas_df.csv", index=False)

    def transform(self, context, flowFile):
        self.train_atlas_df = load_training_paths_in_df()

        # Do I need to convert pandas to a string or can I pass it directly to str.encode()?

        return FlowFileTransformResult(relationship = "success", contents = str.encode(self.train_atlas_df))


    def getPropertyDescriptors(self):
        return []