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

import torch
import torch.nn as nn
import torch.nn.functional as F

import SimpleITK as sitk

# from tqdm import tqdm
from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

class BrainMRIDataset(torch.utils.data.Dataset):
    def __init__(self, brain_voxel_list, brain_mask_list, debug=False):
        self.voxel_paths = brain_voxel_list
        self.mask_paths = brain_mask_list
        self.debug = debug

    def __len__(self):
        return len(self.voxel_paths)

    def __getitem__(self, idx):
        if self.debug:
            print("idx = {}".format(idx))

        # sitk to torch tensor dims (channels, depth, height, width)
        if self.debug:
            print("self.voxel_paths[idx] = {}".format(self.voxel_paths[idx]))
        voxel = sitk.ReadImage(self.voxel_paths[idx])
        voxel_array = sitk.GetArrayFromImage(voxel)
        voxel_tensor = torch.tensor(voxel_array).float()
        if self.debug:
            print("voxel_tensor shape = {}".format(voxel_tensor.shape))
            print("self.mask_paths[idx] = {}".format(self.mask_paths[idx]))

        mask_voxel = sitk.ReadImage(self.mask_paths[idx])
        mask_voxel_array = sitk.GetArrayFromImage(mask_voxel)
        mask_voxel_tensor = torch.from_numpy(mask_voxel_array).float()

        if self.debug:
            print("mask_voxel_tensor shape = {}".format(mask_voxel_tensor.shape))
        
        return voxel_tensor, mask_voxel_tensor

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SimpleUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_features=[64, 128, 256, 512], debug=False):
        super(SimpleUNet3D, self).__init__()
        self.debug = debug

        self.upsampling3d = nn.ModuleList()
        self.downsampling3d = nn.ModuleList()

        self.pool3d = nn.MaxPool3d(kernel_size=2, stride=2)

        # Downsampling encoder part of UNET
        for feature in hidden_features:
            self.downsampling3d.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Upsampling decoder part of UNET
        for feature in reversed(hidden_features):
            # we do ConvTranspose2d for up, then DoubleConv for 2 convs
            self.upsampling3d.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.upsampling3d.append(DoubleConv(feature*2, feature))

        self.bottleneck3d = DoubleConv(hidden_features[-1], hidden_features[-1]*2)

        self.final_conv3d = nn.Conv3d(hidden_features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections3d = []

        # we loop for downsampling
        for down3d in self.downsampling3d:
            x = down3d(x)
            # for skip connections, ordering is important here, first is highest resolution, last is smallest resolution
            skip_connections3d.append(x)
            x = self.pool3d(x)

        x = self.bottleneck3d(x)

        # now go backwards to make skip connections easier, reverse list, go with skip connections with highest res to lowest res
        skip_connections3d = skip_connections3d[::-1]

        # we loop taking steps of 2 for upsampling with DoubleConv
        for idx in range(0, len(self.upsampling3d), 2):
            # upsample with ConvTranspose2d
            x = self.upsampling3d[idx](x)
            # Divide idx by 2 to get just idx since we're doing step by 2 above
            skip_connection = skip_connections3d[idx//2]

            # Soln to cases when not divisible by 2 issue
                # NOTE: in the UNET paper, they did cropping, but we'll do resizing for this issue
            if x.shape != skip_connection.shape:
                # we check x from outward part of upsampling, ifnequal resize our x using skip_connection resolutions just by channels, ignore h & w
                if self.debug:
                    print("x.shape = {}".format(x.shape))
                    print("skip_connection.shape = {}".format(skip_connection.shape))
                    print("skip_connection.shape[2:] = {}".format(skip_connection.shape[2:]))
                # x = TF.resize(x, size=skip_connection.shape[2:])
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="trilinear", align_corners=False)
                # monai_resize3d = monai.transforms.Resize(spatial_size=skip_connection.shape[2:])
                # x = monai_resize3d(x)

            if self.debug:
                print("Concatenating skip connections")
            # Concatenate skip connections and add them along channel dimensions (ex: we have batch, channel, h, w)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # Then run through DoubleConv
            x = self.upsampling3d[idx+1](concat_skip)

        return self.final_conv3d(x)



# Verified we can run SimpleITK N4 Bias Field Correction and produces expected results faster than nipype's version

# TODO (JG): Limitation in flow is flow file not passed to next processor until processor finishes work. This is with each processor like this

# TODO (JG): Make this work for training and testing sets
class ExecuteDNNSkullStrippingSegmentation(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['pandas==1.3.5', 'SimpleITK==2.2.1', "torch"]
        description = 'Gets each SimpleITK intensity normalization 3D NIfTI voxel filepath and resampled cropped segmentation mask 3D NIfTI voxel filepath from the pandas csv dataframe in the flow file, creates a PyTorch Brain MRI Dataset, then runs a PyTorch Brain DataLoader, so we can perform Skull Stripping Segmentation on that loaded data'
        tags = ['sjsu_ms_ai', 'csv', 'nifti', 'pytorch', '3D segmentation']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.skull_strip_seg_dir = PropertyDescriptor(
            name = 'Skull Stripping Segmentation Destination Path',
            description = 'The folder to store the 3D skull stripping segmentation NIfTI files',
            default_value="{}/src/datasets/atlas_NiFi/{}".format(os.path.expanduser("~"), "skull_strip_seg"),
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Performed Skull Stripping Segmentation',
            description = 'If Skull Stripping Segmentation Already Performed, then just get filepaths',
            default_value=False,
            required = True,
        )
        self.data_type = PropertyDescriptor(
            name = 'Dataset Name',
            description = 'The name of the Dataset, currently supported: { nfbs, icpsr_stroke, atlas }.',
            default_value = "atlas",
            required=True
        )
        self.torch_learning_rate = PropertyDescriptor(
            name = 'PyTorch Model Learning Rate',
            description = 'The learning rate for 3D skull stripping segmentation model.',
            default_value = "0.1",
            required=True
        )
        self.batch_size = PropertyDescriptor(
            name = 'Model Inference Batch Size',
            description = 'The batch size of data to pass to 3D skull stripping segmentation model for predictions on chunks of data.',
            default_value = "1",
            required=True
        )
        self.torch_model_file = PropertyDescriptor(
            name = 'PyTorch 3D DNN Model',
            description = 'The filepath of the pretrained pytorch 3D skull stripping segmentation model to use for brain tissue extraction, currently supported: { unet3d }.',
            default_value = "best_unet3d_model_loss_100.pt",
            required=True
        )
        self.descriptors = [self.skull_strip_seg_dir, self.already_prepped, self.data_type, self.batch_size, self.torch_model_file]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for bias_corrected_dirpath, etc")
        self.skull_strip_seg_filepaths = list()
        # read pre-trained model and config file
        self.skull_strip_seg_dirpath = context.getProperty(self.skull_strip_seg_dir.name).getValue()
        self.skull_strip_seg_already_done = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.data_name = context.getProperty(self.data_type.name).getValue()
        self.inference_batch_size = context.getProperty(self.batch_size.name).asInteger()
        self.torch_model_filename = context.getProperty(self.torch_model_file.name).getValue()
        self.LEARNING_RATE = context.getProperty(self.torch_learning_rate.name).asFloat()
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    def load_checkpoint(self, checkpoint, model):
        print("=> Loading Checkpoint")
        model.load_state_dict(checkpoint["state_dict"])

    def check_accuracy(self, loader, model, device="cuda"):
        num_correct = 0
        num_pixels = 0
        # TODO: IoU
        dice_score = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device).unsqueeze(1)
                y = y.to(device=device).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)

                dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
                )
        
        print(f"Acc Ratio {num_correct}/{num_pixels} with Acc {num_correct/num_pixels*100:.2f}")
        print(f"Dice Score: {dice_score/len(loader)}")

    def save_predictions_as_segs(self, loader, model, folder, device="cuda"):
        model.eval()

        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device).unsqueeze(1)
            # y = y.unsqueeze(1)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()

            preds_np = preds.squeeze().cpu().numpy()
            preds_sitk = sitk.GetImageFromArray(preds_np)
            # ground_sitk = sitk.GetImageFromArray(y.squeeze().cpu().numpy())

            # mkdir_prep_dir(folder)

            pred_filename = f"pred_{idx}.nii.gz"
            pred_filepath = f"{folder}/{pred_filename}"
            sitk.WriteImage(preds_sitk, pred_filepath)
            # sitk.WriteImage(ground_sitk, f"{folder}/ground_{idx}.nii.gz")

            self.skull_strip_seg_filepaths.append(pred_filepath)

    # TODO (JG): Finish
    def skull_strip_segmentation(self, nifti_csv_data):
        self.logger.info("Executing 3D DNN Skull Stripping Segmentation")
        self.mkdir_prep_dir(self.skull_strip_seg_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.skull_strip_seg_already_done type = {}".format(type(self.skull_strip_seg_already_done)))
        if self.skull_strip_seg_already_done:
            self.logger.info("Adding 3D Skull Stripping Segmentation NIfTI filepaths to data df in skull_strip_seg_dir")

            self.skull_strip_seg_filepaths = [self.skull_strip_seg_dirpath + os.sep + "pred_" + str(i) + ".nii.gz" for i in range(len(nifti_csv_data))]
            nifti_csv_data["skull_strip_seg"] = self.skull_strip_seg_filepaths
            self.logger.info("Retrieved {} 3D Skull Stripping Segmentation NIfTI filepaths stored at : {}/".format(len(self.skull_strip_seg_filepaths), self.skull_strip_seg_dirpath))
        else:
            self.logger.info("Doing the PyTorch 3D Skull Stripping Segmentation NIfTI From Scratch")
            # for i in range(len(nifti_csv_data)):
            brain_voxel_list = nifti_csv_data["intensity_norm"].tolist()
            brain_mask_list = nifti_csv_data["mask_index"].tolist()

            brain_dataset = BrainMRIDataset(brain_voxel_list, brain_mask_list)

            brain_dataloader = DataLoader(brain_train_dataset, batch_size=self.inference_batch_size, shuffle=False)

            # Create the UNet3D model and then load the weights
            unet3d_model = SimpleUNet3D(in_channels=1, out_channels=1).to(device=self.DEVICE)
            optimizer = optim.Adam(unet3d_model.parameters(), lr=self.LEARNING_RATE)
            bce_criterion = nn.BCEWithLogitsLoss().to(device=self.DEVICE)
            self.load_checkpoint(torch.load(self.torch_model_filename), unet3d_model)
            self.check_accuracy(brain_dataloader, unet3d_model, device=self.DEVICE)

            self.save_predictions_as_segs(
                brain_dataloader, unet3d_model, folder=self.skull_strip_seg_dirpath, device=self.DEVICE
            )

        nifti_csv_data["skull_strip_seg"] = self.skull_strip_seg_filepaths
        self.logger.info("Torch 3D Skull Stripping Segmentation, saved {} NIfTI files at: {}/".format(len(self.skull_strip_seg_filepaths), self.skull_strip_seg_dirpath))
        return nifti_csv_data
 
    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        nifti_csv_data = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: nifti_csv_data = {}".format(nifti_csv_data.head()))
        nifti_csv_data_up = self.skull_strip_segmentation(nifti_csv_data)
        self.logger.info("output: nifti_csv_data_up = {}".format(nifti_csv_data_up.head()))

        # Create a StringIO object
        nifti_csv_data_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        nifti_csv_data_up.to_csv(nifti_csv_data_string_io, index=False)

        # Get the string value and encode it
        nifti_csv_data_up_string = nifti_csv_data_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = nifti_csv_data_up_string)
