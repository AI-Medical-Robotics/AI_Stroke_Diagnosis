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
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pickle5 as pickle

import SimpleITK as sitk

# from tqdm import tqdm
from nifiapi.properties import PropertyDescriptor
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

# TODO (JG): Limitation in flow is flow file not passed to next processor until processor finishes work. This is with each processor like this

class BrainMRIDataset(torch.utils.data.Dataset):
    def __init__(self, brain_voxel_list, debug=True):
        self.brain_voxel_paths = brain_voxel_list
        self.debug = debug

    def __len__(self):
        return len(self.brain_voxel_paths)

    def __getitem__(self, idx):
        if self.debug:
            print("idx = {}".format(idx))

        # sitk to torch tensor dims (channels, depth, height, width)
        if self.debug:
            print("self.brain_voxel_paths[idx] = {}".format(self.brain_voxel_paths[idx]))
        voxel = sitk.ReadImage(self.brain_voxel_paths[idx])
        voxel_array = sitk.GetArrayFromImage(voxel)
        voxel_tensor = torch.tensor(voxel_array).float()
        if self.debug:
            print("voxel_tensor shape = {}".format(voxel_tensor.shape))

        return voxel_tensor

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



# TODO (JG): Update after saving pickle bytes file, then further compress to NIfTI file (smaller MB instead of GB)
class MapVoxelIDsToExtFeatures(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'
        dependencies = ['SimpleITK==2.2.1', 'pandas==1.3.5', 'pillow==10.0.0', 'pickle5==0.0.11', "torch", "torchvision", "torchaudio"]
        description = 'Gets SimpleITK Preprocessed 3D voxel pickle bytes filepaths from the pandas csv dataframe in the flow file, loads each pickle bytes as a dictionary mapping 3D voxel names to preprocessed 3D vxels and runs pretrained pytorch 3D UNet model (Skull Strip UNet or Stroke Lesion UNet) to extract features and then map to 3D voxel IDs'
        tags = ['sjsu_ms_ai', 'csv', 'nifti', 'pytorch']

    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.imgid_to_ext_feature_dir = PropertyDescriptor(
            name = 'Voxel ID Mappings to Extracted Features Destination Path',
            description = 'The folder to store the voxel IDs mapped to extracted features',
            default_value="/media/ubuntu/ai_projects/data/ICPSR_38464_Stroke_Data_NiFi/{}".format("map_voxids_to_features"),
            required = True
        )
        self.already_prepped = PropertyDescriptor(
            name = 'Already Mapped Voxel IDs to Extracted Features',
            description = 'If Voxel IDs Mapped to Extracted Features Already Performed, then just get filepaths',
            default_value=False,
            required = True,
        )
        self.data_type = PropertyDescriptor(
            name = 'Dataset Name',
            description = 'The name of the Dataset, currently supported: { icpsr_stroke }.',
            default_value = "icpsr_stroke",
            required=True
        )
        self.batch_size = PropertyDescriptor(
            name = 'Model Inference Batch Size',
            description = 'The batch size of data to pass to 3D stroke lesion segmentation model for predictions on chunks of data.',
            default_value = "1",
            required=True
        )
        self.torch_model_type = PropertyDescriptor(
            name = 'PyTorch Model Type',
            description = 'The pytorch model type to use for voxel feature extraction, currently supported: { 3d_unet }.',
            default_value = "3d_unet",
            required=True
        )
        self.torch_model_filepath = PropertyDescriptor(
            name = 'PyTorch 3D DNN Model',
            description = 'The filepath of the pretrained pytorch 3D stroke lesion segmentation model, currently supported: { unet3d }.',
            default_value = "{}/src/AI_Stroke_Diagnosis/DataPipeline/src/backend/torch/{}/icpsr/models/{}/unet3d_stroke_lesion_seg_500.pth.tar".format(os.path.expanduser("~"), "stroke-lesion-seg", "simple_unet3d_weights"),
            required=True
        )
        self.descriptors = [self.imgid_to_ext_feature_dir, self.already_prepped, self.data_type, self.batch_size, self.torch_model_type, self.torch_model_filepath]

    def getPropertyDescriptors(self):
        return self.descriptors

    def str_to_bool(self, string_input):
        return {"true": True, "false": False}.get(string_input.lower())

    def onScheduled(self, context):
        self.logger.info("Getting properties for bias_corrected_dirpath, etc")
        self.imgid_to_feature_map = list()
        # read pre-trained model and config file
        self.imgid_to_ext_feature_dirpath = context.getProperty(self.imgid_to_ext_feature_dir.name).getValue()
        self.img_map_already_done = self.str_to_bool(context.getProperty(self.already_prepped.name).getValue())
        self.data_name = context.getProperty(self.data_type.name).getValue()
        self.inference_batch_size = context.getProperty(self.batch_size.name).asInteger()
        self.torch_model_name = context.getProperty(self.torch_model_type.name).getValue()
        self.torch_dnn_filepath = context.getProperty(self.torch_model_filepath.name).getValue()
        self.LEARNING_RATE = 1e-1
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def mkdir_prep_dir(self, dirpath):
        """make preprocess directory if doesn't exist"""
        prep_dir = dirpath
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        return prep_dir

    def load_checkpoint(self, checkpoint, model):
        self.logger.info("=> Loading Checkpoint")
        model.load_state_dict(checkpoint["state_dict"])


    """
        Need the ground truth included if I am using this method
    """
    def check_accuracy(self, loader, model, device="cuda"):
        num_correct = 0
        num_pixels = 0
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
        
        self.logger.info(f"Acc Ratio {num_correct}/{num_pixels} with Acc {num_correct/num_pixels*100:.2f}")
        self.logger.info(f"Dice Score: {dice_score/len(loader)}")


    def run_seg_predictions(self, loader, model, device="cuda"):
        model.eval()

        for idx, x in enumerate(loader):
            self.logger.info("3d unet inference idx = {}".format(idx))
            x = x.to(device=device).unsqueeze(1)
            with torch.no_grad():
                ext_features = model(x)
                preds = torch.sigmoid(ext_features)
                preds = (preds > 0.5).float()

            preds_np = preds.squeeze().cpu().numpy()
            preds_sitk = sitk.GetImageFromArray(preds_np)

        return ext_features


    # TODO (JG): Finish
    def map_imgids_to_extfeatures(self, img_cap_csv_data):
        self.logger.info("Mapping Voxel IDs to PyTorch CNN Extracted Features")
        imgid_to_extfeature_dir = self.mkdir_prep_dir(self.imgid_to_ext_feature_dirpath)

        # In NiFi Python Processor, add property for this
        self.logger.info("self.img_map_already_done type = {}".format(type(self.img_map_already_done)))
        if self.img_map_already_done:
            self.logger.info("Adding Mapped IDs to Extracted Features filepaths to data df in imgid_to_ext_feature_dir")

            self.imgid_to_feature_map = [imgid_to_extfeature_dir + os.sep + self.data_name + "_" + str(i) + ".pk1" for i in range(len(img_cap_csv_data))]
            img_cap_csv_data["imgid_to_feature"] = self.imgid_to_feature_map
            self.logger.info("Retrieved Mapped IDs to Extracted Features filepaths stored at : {}/".format(imgid_to_extfeature_dir))
        else:
            self.logger.info("Doing the Torch Mapped IDs to Extracted Features From Scratch")
            for i in range(len(img_cap_csv_data)):
                torch_imgid_to_extfeatures = {}
                # elif self.data_name == "atlas":
                #     input_voxel = sitk.ReadImage(img_cap_csv_data.train_t1w_raw.iloc[i], sitk.sitkFloat32)
                if self.data_name == "icpsr_stroke":
                    with open(img_cap_csv_data.img_name_to_prep_img.iloc[i], "rb") as file:
                        torch_prep_name_to_voxel = pickle.load(file)
                    # input_voxel = sitk.ReadImage(img_cap_csv_data.brain_dwi_orig.iloc[i], sitk.sitkFloat32)

                # Perform Torch Feature Extraction with (VGG16 or Resnet50)
                voxel_filename = list(torch_prep_name_to_voxel.keys())[0]
                intensity_norm_voxel_filepath = list(torch_prep_name_to_voxel.values())
                self.logger.info("intensity_norm_voxel_filepath = {}".format(intensity_norm_voxel_filepath))
                # self.logger.info("list(intensity_norm_voxel_filepath) = {}".format(list(intensity_norm_voxel_filepath)))
                # self.logger.info("list(intensity_norm_voxel_filepath)[0] = {}".format(list(intensity_norm_voxel_filepath)[0]))

                # Dont need skull mask list
                brain_dataset = BrainMRIDataset(intensity_norm_voxel_filepath)

                brain_dataloader = DataLoader(brain_dataset, batch_size=self.inference_batch_size, shuffle=False)

                # TODO (JG): Add support for 3D U-Net, V-Net, Dense VoxelNet or VoxResNet
                # Do we use 3D UNet (Skull Strip or Stroke Lesion?)
                if self.torch_model_name == "3d_unet":
                    unet3d_model = SimpleUNet3D(in_channels=1, out_channels=1).to(device=self.DEVICE)

                    optimizer = optim.Adam(unet3d_model.parameters(), lr=self.LEARNING_RATE)
                    bce_criterion = nn.BCEWithLogitsLoss().to(device=self.DEVICE)
                    self.load_checkpoint(torch.load(self.torch_dnn_filepath), unet3d_model)
                    # self.check_accuracy(brain_dataloader, unet3d_model, device=self.DEVICE)


                with torch.no_grad():
                    # previous prep_voxel_tensor_batch
                    # features_extracted = unet_model(prep_intensity_norm_voxel)
                    features_extracted = self.run_seg_predictions(brain_dataloader, unet3d_model, device=self.DEVICE)

                # TODO (JG): Could add data member here, so its easier to change from top of processor class
                voxel_id = voxel_filename.split(".")[0]

                torch_imgid_to_extfeatures[voxel_id] = features_extracted

                imgid_to_extfeature_bytes = pickle.dumps(torch_imgid_to_extfeatures)

                # Save the voxel name mapped torch preprocessed voxel
                output_path = os.path.join(imgid_to_extfeature_dir, self.data_name + "_" + str(i) + ".pk1")
                self.logger.info("Torch Mapped IDs to Extracted Features pickle output_path = {}".format(output_path))
                try:
                    with open(output_path, "wb") as file:
                        file.write(imgid_to_extfeature_bytes)
                    self.logger.info("Torch Mapped IDs to Extracted Features pickle bytes saved successfully!")
                except Exception as e:
                    self.logger.error("An error occurred while saving Mapped IDs to Extracted Features!!!: {}".format(e))
                self.imgid_to_feature_map.append(output_path)

        img_cap_csv_data["imgid_to_feature"] = self.imgid_to_feature_map
        self.logger.info("Torch Mapped IDs to Extracted Features pickle bytes stored at: {}/".format(imgid_to_extfeature_dir))
        return img_cap_csv_data

    def transform(self, context, flowFile):
        # Read FlowFile contents into a pandas dataframe
        img_cap_csv_pd = pd.read_csv(io.BytesIO(flowFile.getContentsAsBytes()))
        self.logger.info("input: img_cap_csv_pd = {}".format(img_cap_csv_pd.head()))
        img_cap_csv_pd_up = self.map_imgids_to_extfeatures(img_cap_csv_pd)
        self.logger.info("output: img_cap_csv_pd_up = {}".format(img_cap_csv_pd_up.head()))

        # Create a StringIO object
        img_cap_csv_string_io = io.StringIO()

        # Use the to_csv() method to write to CSV
        img_cap_csv_pd_up.to_csv(img_cap_csv_string_io, index=False)

        # Get the string value and encode it
        img_cap_csv_pd_up_string = img_cap_csv_string_io.getvalue().encode("utf-8")

        return FlowFileTransformResult(relationship = "success", contents = img_cap_csv_pd_up_string)
