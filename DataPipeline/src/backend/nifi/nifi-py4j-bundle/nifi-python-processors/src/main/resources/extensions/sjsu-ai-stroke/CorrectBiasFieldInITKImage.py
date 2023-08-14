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
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

# Verified we can run SimpleITK N4 Bias Field Correction and produces expected results faster than nipype's version

# TODO (JG): Make this work for training and testing sets
class CorrectBiasFieldInITKImage(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['SimpleITK', 'pandas', 'tqdm']
        description = 'Gets NIfTI filepaths from the pandas csv dataframe in the flow file, loads each NIfTI file as an ITK voxel and performs SimpleITK N4 Bias Field Correction on each 3D NIfTI voxel'
        tags = ['sjsu_ms_ai', 'csv', 'itk', 'nifti']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.bias_field_dir = PropertyDescriptor(
            name = 'ITK Bias Correction Folder Path',
            description = 'The folder to stored the ITK Preprocessed Bias Field Corrected Images.',
            default_value="{}/src/datasets/NFBS_Dataset_NiFi/{}".format(os.path.expanduser("~"), "bias_correction"),
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Corrected Bias Field',
            description = 'If ITK Bias Field Correction Already Performed, then just get filepaths',
            default_value=False,
            required = True,
        )
        self.nifti_data_type = PropertyDescriptor(
            name = 'NifTI Dataset Name',
            description = 'The name of the NifTI Dataset, currently supported: {nfbs, atlas, icpsr_stroke}.',
            default_value = "nfbs",
            required=True
        )
        self.descriptors = [self.bias_field_dir, self.already_prepped, self.nifti_data_type]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for bias_corrected_dirpath, etc")
        self.index_corr = list()
        # read pre-trained model and config file
        self.bias_corrected_dirpath = context.getProperty(self.bias_field_dir.name).getValue()
        self.bias_already_corrected = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.nifti_data_name = context.getProperty(self.nifti_data_type.name).getValue()

    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    def itk_bias_correction(self, nifti_csv_data):
        self.logger.info("Performing SimpleITK Bias Field Correction")
        bias_dir = self.mkdir_prep_dir(self.bias_corrected_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.bias_already_corrected type = {}".format(type(self.bias_already_corrected)))
        if self.bias_already_corrected:
            self.logger.info("Adding Prepped SimpleITK N4BiasFieldCorrection filepaths to data df in bias_corr")

            self.index_corr = [bias_dir + os.sep + self.nifti_data_name + "_" + str(i) + ".nii.gz" for i in range(len(nifti_csv_data))]
            nifti_csv_data["bias_corr"] = self.index_corr
            self.logger.info("Retrieved Prepped bias corrected voxel filepaths stored at : {}/".format(bias_dir))
        else:
            self.logger.info("Doing the SimpleITK Bias Field Correction From Scratch")
            for i in tqdm(range(len(nifti_csv_data))):
                # Load the image using ITK
                if self.nifti_data_name == "nfbs":
                    input_image = sitk.ReadImage(nifti_csv_data.raw.iloc[i], sitk.sitkFloat32)
                elif self.nifti_data_name == "atlas":
                    input_image = sitk.ReadImage(nifti_csv_data.train_t1w_raw.iloc[i], sitk.sitkFloat32)
                elif self.nifti_data_name == "icpsr_stroke":
                    input_image = sitk.ReadImage(nifti_csv_data.brain_dwi_orig.iloc[i], sitk.sitkFloat32)

                # Set shrink factor to 3
                # https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
                shrinkFactor = 3
                if shrinkFactor > 1:
                    image = sitk.Shrink(
                        input_image, [shrinkFactor] * input_image.GetDimension()
                    )

                # Perform N4 bias field correction
                self.logger.info("Create SimpleITK N4 Bias Field Correction Filter")
                corrector = sitk.N4BiasFieldCorrectionImageFilter()

                self.logger.info("Set SimpleITK N4 Filter Parameters")
                corrector.SetMaximumNumberOfIterations([20, 10, 10, 5])

                self.logger.info("Execute SimpleITK N4 Bias Field Correction")
                corrected_image = corrector.Execute(image)

                self.logger.info("Getting corrected_image type = {}".format(type(corrected_image)))

                # Save the corrected image
                output_path = os.path.join(bias_dir, self.nifti_data_name + "_" + str(i) + ".nii.gz")
                self.logger.info("Correct Bias Field prepped output_path = {}".format(output_path))
                try:
                    sitk.WriteImage(corrected_image, output_path)
                    self.logger.info("SimpleITK Correct Field Bias Image saved successfully!")
                except Exception as e:
                    self.logger.error("An error occurred while saving ITK Correct Field Bias Image!!!: {}".format(e))
                self.index_corr.append(output_path)

        nifti_csv_data["bias_corr"] = self.index_corr
        self.logger.info("ITK Bias-corrected voxels stored at: {}/".format(bias_dir))
        return nifti_csv_data

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        # NIfTI: — Neuroimaging Informatics Technology Initiative
        nifti_csv = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: nifti_csv = {}".format(nifti_csv.head()))
        nifti_csv_prepped = self.itk_bias_correction(nifti_csv)
        self.logger.info("output: nifti_csv_prepped = {}".format(nifti_csv_prepped.head()))

        # Create a StringIO object
        nifti_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        nifti_csv_prepped.to_csv(nifti_string_io, index=False)

        # Get the string value and encode it
        nifti_csv_prepped_string = nifti_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = nifti_csv_prepped_string)
