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
from tqdm import tqdm
import matplotlib.pyplot as plt
from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

class SaveITKImageSlice(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['itk==5.3.0', 'pandas==1.3.5', 'matplotlib', 'tqdm==4.66.1']
        description = 'Gets a sample index for a NIfTI filepath for a particular stage in the AI stroke diagnosis pipeline located in the pandas csv dataframe column in the flow file, loads that NIfTI file as an ITK voxel, slices that voxel into a 2D image slice and save it as a png image'
        tags = ['sjsu_ms_ai', 'csv', 'itk', 'matplotlib', 'nifti']


    def __init__(self, jvm=None, **kwargs):
        # Build Property Descriptors
        self.nifti_slices_to_save = PropertyDescriptor(
            name = 'NifTI Percent Of 2D Image Files To Save',
            description = 'As we process 3D voxels, and we if we verify with a sample image slice per voxel, choose a percentage of 2D image slices to save across all 3D voxels',
            default_value="0.025",
            required = False
        )
        self.nifti_index = PropertyDescriptor(
            name = 'NifTI Filepath Pandas DF Row',
            description = 'Choose pandas dataframe row for NifTI filepath',
            default_value="0",
            required = False
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
            description = 'The name of the NifTI Data Prep section you want to see an image slice from. Examples: get_nifti, correct_bias, resize_crop, intensity_norm, etc',
            default_value = 'get_nifti',
            required = True
        )
        self.saved_img_dir = PropertyDescriptor(
            name = 'Saved ITK Image Folder Path',
            description = 'The folder to store the ITK 2D Sliced Images.',
            default_value="{}/src/datasets/NFBS_Dataset_NiFi/{}".format(os.path.expanduser("~"), "get_nifti"),
            required = True
        )
        self.descriptors = [self.nifti_slices_to_save, self.nifti_index, self.nifti_slice_divisor, self.nifti_data_type, self.nifti_csv_col, self.saved_img_dir]

    def getPropertyDescriptors(self):
        return self.descriptors

    def onScheduled(self, context):
        self.logger.info("Getting properties for nifti_index, etc")

        self.nifti_percent_slices_save = context.getProperty(self.nifti_slices_to_save.name).asFloat()
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

    def load_itk_from_nifti(self, nifti_csv, nifti_index):
        nifti_voxel = None
        nifti_voxel_mask = None
        if self.nifti_csv_col_name == "get_nifti":
            if self.nifti_data_name == "nfbs":
                nifti_voxel = itk.imread(nifti_csv.raw.iloc[nifti_index])
            elif self.nifti_data_name == "atlas":
                nifti_voxel = itk.imread(nifti_csv.train_t1w_raw.iloc[nifti_index])
            elif self.nifti_data_name == "icpsr_stroke":
                nifti_voxel = itk.imread(nifti_csv.brain_dwi_orig.iloc[nifti_index])
        elif self.nifti_csv_col_name == "correct_bias":
            nifti_voxel = itk.imread(nifti_csv.bias_corr.iloc[nifti_index], itk.F)
        elif self.nifti_csv_col_name == "resize_crop":
            nifti_voxel = itk.imread(nifti_csv.raw_index.iloc[nifti_index], itk.F)
            nifti_voxel_mask = itk.imread(nifti_csv.mask_index.iloc[nifti_index], itk.UC)
        elif self.nifti_csv_col_name == "intensity_norm":
            self.logger.info("Reading intensity_norm voxel ID {} taking 1 2D slice example per voxel".format(nifti_index))
            nifti_voxel = itk.imread(nifti_csv.intensity_norm.iloc[nifti_index], itk.F)

        return nifti_voxel, nifti_voxel_mask

    def transform(self, context, flowFile):
        # TODO (JG): Add the checks for which section of data pipeline we are in that we want to print image slice for
        # Read FlowFile contents into an image
        nifti_csv = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))

        if self.nifti_csv_col_name == "get_nifti" or self.nifti_csv_col_name == "correct_bias":
            nifti_voxel, nifti_voxel_mask = self.load_itk_from_nifti(nifti_csv, self.nifti_voxel_index)
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
            self.logger.info("Saving Image to path = {}".format(output_filepath))
            plt.savefig(output_filepath)
        elif self.nifti_csv_col_name == "resize_crop":
            fraction_length = int(len(nifti_csv) * self.nifti_percent_slices_save)
            for nifti_index in tqdm(range(fraction_length)):
                nifti_voxel, nifti_voxel_mask = self.load_itk_from_nifti(nifti_csv, nifti_index)
                # Convert ITK image to NumPy array for matplotlib visualization
                nifti_voxel_array = itk.GetArrayViewFromImage(nifti_voxel)
                nifti_voxel_mask_array = itk.GetArrayViewFromImage(nifti_voxel_mask)

                # Create a figure and axis for visualization
                fig, ax = plt.subplots(1, 2, figsize=(14, 10))
                ax[0].set_title("NifTI {} 2D Image ID {} Slice = {}".format(nifti_index, self.nifti_data_name, nifti_voxel_array.shape))
                # Display the 2D image slice
                ax[0].imshow(nifti_voxel_array[nifti_voxel_array.shape[0]//self.nifti_image_divisor])

                ax[1].set_title("NifTI {} 2D Image ID {} Mask Slice = {}".format(nifti_index, self.nifti_data_name, nifti_voxel_mask_array.shape))
                ax[1].imshow(nifti_voxel_mask_array[nifti_voxel_mask_array.shape[0]//self.nifti_image_divisor])

                # Save the 2D image slice as file
                saved_itk_image_dir = self.mkdir_prep_dir(self.saved_img_dirpath)
                output_filename = "nifti_image_id_{}_slice_{}_{}.{}".format(nifti_index, nifti_voxel_array.shape[0]//self.nifti_image_divisor, self.nifti_csv_col_name, "png")
                output_filepath = os.path.join(saved_itk_image_dir, output_filename)
                self.logger.info("Saving Image to path = {}".format(output_filepath))
                plt.savefig(output_filepath)
        elif self.nifti_csv_col_name == "intensity_norm":
            self.logger.info("Saving all intensity_norm voxels taking 1 2D slice example per voxel")
            fraction_length = int(len(nifti_csv) * self.nifti_percent_slices_save)
            for nifti_index in tqdm(range(fraction_length)):
                nifti_voxel, nifti_voxel_mask = self.load_itk_from_nifti(nifti_csv, nifti_index)
                # Convert ITK image to NumPy array for matplotlib visualization
                nifti_voxel_array = itk.GetArrayViewFromImage(nifti_voxel)

                # Create a figure and axis for visualization
                fig, ax = plt.subplots(1, 1)
                ax.set_title("NifTI {} 2D Image Slice = {}".format(nifti_index, nifti_voxel_array.shape))
                # Would Display the 2D image slice, but in headless mode
                ax.imshow(nifti_voxel_array[nifti_voxel_array.shape[0]//self.nifti_image_divisor])

                # Save the 2D image slice as file
                saved_itk_image_dir = self.mkdir_prep_dir(self.saved_img_dirpath)
                output_filename = "nifti_image_id_{}_slice_{}_{}.{}".format(nifti_index, nifti_voxel_array.shape[0]//self.nifti_image_divisor, self.nifti_csv_col_name, "png")
                output_filepath = os.path.join(saved_itk_image_dir, output_filename)
                self.logger.info("Saving Image to path = {}".format(output_filepath))
                plt.savefig(output_filepath)

        return FlowFileTransformResult(relationship = "success")
        # return FlowFileTransformResult(relationship = "success", contents = str.encode(output_filepath))
