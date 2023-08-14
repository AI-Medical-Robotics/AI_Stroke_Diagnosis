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
import itk
import pandas as pd
import matplotlib.pyplot as plt
from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

class SaveITKImageSlice(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['itk', 'pandas', 'matplotlib', 'numpy']
        description = 'Gets a sample index for a NIfTI filepath for a particular stage in the AI stroke diagnosis pipeline located in the pandas csv dataframe column in the flow file, loads that NIfTI file as an ITK voxel, slices that voxel into a 2D image slice and save it as a png image'
        tags = ['sjsu_ms_ai', 'csv', 'itk', 'matplotlib', 'nifti']


    def __init__(self, jvm=None, **kwargs):
        # Build Property Descriptors
        self.nifti_index = PropertyDescriptor(
            name = 'NifTI File Index',
            description = 'Choose table index for NifTI filepath',
            default_value="1",
            required = True
        )
        self.nifti_slice_divisor = PropertyDescriptor(
            name = 'NifTI 2D Slice Image Divisor',
            description = 'Choose the 2D Sliced Image Divisor from NifTI 3D voxel',
            default_value="2",
            required = True,
        )
        self.nifti_data_type = PropertyDescriptor(
            name = 'NifTI Dataset Name',
            description = 'The name of the NifTI Dataset, currently supported: {nfbs, atlas, icpsr_stroke}.',
            default_value = "nfbs",
            required=True
        )
        self.nifti_csv_col = PropertyDescriptor(
            name = 'NifTI CSV Column Name',
            description = 'The name of the NifTI Data Prep section you want to see an image slice from. Examples: get_nifti, correct_bias, resize_crop, intens_norm, etc',
            default_value = 'get_nifti',
            required = True
        )
        self.saved_img_dir = PropertyDescriptor(
            name = 'Saved ITK Image Folder Path',
            description = 'The folder to store the ITK 2D Sliced Images.',
            default_value="{}/src/datasets/NFBS_Dataset_NiFi/{}".format(os.path.expanduser("~"), "get_nifti"),
            required = True
        )
        self.descriptors = [self.nifti_index, self.nifti_slice_divisor, self.nifti_data_type, self.nifti_csv_col, self.saved_img_dir]

    def getPropertyDescriptors(self):
        return self.descriptors

    def onScheduled(self, context):
        self.logger.info("Getting properties for nifti_index, etc")

        self.nifti_voxel_index = context.getProperty(self.nifti_index.name).asInteger()
        self.nifti_image_divisor = context.getProperty(self.nifti_slice_divisor.name).asInteger()
        self.nifti_data_name = context.getProperty(self.nifti_data_type.name).getValue()
        self.nifti_csv_col_name = context.getProperty(self.nifti_csv_col.name).getValue()
        self.saved_img_dirpath = context.getProperty(self.saved_img_dir.name).getValue()

    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    def transform(self, context, flowFile):
        # TODO (JG): Add the checks for which section of data pipeline we are in that we want to print image slice for
        # Read FlowFile contents into an image
        nifti_csv = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))

        if self.nifti_csv_col_name == "get_nifti":
            if self.nifti_data_name == "nfbs":
                nifti_voxel = itk.imread(nifti_csv.raw.iloc[self.nifti_voxel_index])
            elif self.nifti_data_name == "atlas":
                nifti_voxel = itk.imread(nifti_csv.train_t1w_raw.iloc[self.nifti_voxel_index])
            elif self.nifti_data_name == "icpsr_stroke":
                nifti_voxel = itk.imread(nifti_csv.brain_dwi_orig.iloc[self.nifti_voxel_index])
        elif self.nifti_csv_col_name == "correct_bias":
            nifti_voxel = itk.imread(nifti_csv.bias_corr.iloc[self.nifti_voxel_index], itk.F)
        elif self.nifti_csv_col_name == "resize_crop":
            nifti_voxel = itk.imread(nifti_csv.raw_index.iloc[self.nifti_voxel_index], itk.F)
            nifti_voxel_mask = itk.imread(nifti_csv.mask_index.iloc[self.nifti_voxel_index], itk.UC)
        elif self.nifti_csv_col_name == "intens_norm":
            nifti_voxel = itk.imread(nifti_csv.intens_norm.iloc[self.nifti_voxel_index], itk.F)

        if self.nifti_csv_col_name == "get_nifti" or self.nifti_csv_col_name == "correct_bias" or self.nifti_csv_col_name == "intens_norm":
            # Convert ITK image to NumPy array for matplotlib visualization
            nifti_voxel_array = itk.GetArrayViewFromImage(nifti_voxel)

            # Create a figure and axis for visualization
            fig, ax = plt.subplots(1, 1)
            ax.set_title("NifTI 2D Image Slice = {}".format(nifti_voxel_array.shape))
            # Would Display the 2D image slice, but in headless mode
            ax.imshow(nifti_voxel_array[nifti_voxel_array.shape[0]//self.nifti_image_divisor])

            # Save the 2D image slice as file
            saved_itk_image_dir = self.mkdir_prep_dir(self.saved_img_dirpath)
            output_filename = "nifti_image_slice_{}_{}.{}".format(nifti_voxel_array.shape[0]//self.nifti_image_divisor, self.nifti_csv_col_name, "png")
            output_filepath = os.path.join(saved_itk_image_dir, output_filename)
            plt.savefig(output_filepath)
        elif self.nifti_csv_col_name == "resize_crop":
            # Convert ITK image to NumPy array for matplotlib visualization
            nifti_voxel_array = itk.GetArrayViewFromImage(nifti_voxel)
            nifti_voxel_mask_array = itk.GetArrayViewFromImage(nifti_voxel_mask)

            # Create a figure and axis for visualization
            fig, ax = plt.subplots(1, 2, figsize=(14, 10))
            ax[0].set_title("NifTI {} 2D Image Slice = {}".format(self.nifti_data_name, nifti_voxel_array.shape))
            # Display the 2D image slice
            ax[0].imshow(nifti_voxel_array[nifti_voxel_array.shape[0]//self.nifti_image_divisor])

            ax[1].set_title("NifTI {} 2D Image Mask Slice = {}".format(self.nifti_data_name, nifti_voxel_array.shape))
            ax[1].imshow(nifti_voxel_mask_array[nifti_voxel_mask_array.shape[0]//self.nifti_image_divisor])

            # Save the 2D image slice as file
            saved_itk_image_dir = self.mkdir_prep_dir(self.saved_img_dirpath)
            output_filename = "nifti_image_slice_{}_{}.{}".format(nifti_voxel_array.shape[0]//self.nifti_image_divisor, self.nifti_csv_col_name, "png")
            output_filepath = os.path.join(saved_itk_image_dir, output_filename)
            plt.savefig(output_filepath)

        return FlowFileTransformResult(relationship = "success", contents = str.encode(output_filepath))
