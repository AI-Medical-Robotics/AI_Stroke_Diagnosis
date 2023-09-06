import os
import pandas as pd
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import torchmetrics
from torchmetrics import Accuracy, IoU
from focal_loss.focal_loss import FocalLoss

from torch_model import UNet3D

# Reference perplexity.ai for pytorch 3D UNet skull strip seg model
# https://www.perplexity.ai/search/0df235a1-27ba-4b67-bf7b-89c2500685c7?s=u

# Also reference perplexity.ai for pytorch train 3D UNet skull strip seg model
# https://www.perplexity.ai/search/cf74b6d5-9888-462b-9063-e90859bbf389
# Refer to section on "MRIDataset(Dataset)"

# Getting two columns from nift csv data pd table
# nifti_csv_data["intensity_norm"] = normalized raw_index
# nifti_csv_data["mask_index"]

# Load Skull Strip Seg dataset
class BrainMRIDataset(torch.utils.data.Dataset):
    def __init__(self, brain_voxel_list, brain_mask_list, batch_size):
        self.batch_size = batch_size
        self.nifti_csv_df = pd.DataFrame({"intensity_norm": brain_voxel_list, "mask_index": brain_mask_list})
        # do we need to shuffle the rows of the pandas df randomly?
        self.nifti_csv_df = self.nifti_csv_df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.nifti_csv_df) // self.batch_size

    def __getitem__(self, idx):
        batch_nifti_df = self.nifti_csv_df.iloc[idx*self.batch_size:(idx+1)*self.batch_size]
        # pytorch image tensor convention: (batch_size, channels, height, width, depth)
        voxel_batch = torch.zeros((self.batch_size, 1, 128, 128, 96)).float()
        mask_batch = torch.zeros((self.batch_size, 1, 128, 128, 96)).float()
        for i, nifti_df in batch_nifti_df.iterrows():
            voxel = sitk.ReadImage(nifti_df["intensity_norm"])
            voxel = sitk.GetArrayFromImage(voxel)
            voxel = torch.from_numpy(voxel).float()
            voxel = voxel.unsqueeze(-1)

            mask_voxel = sitk.ReadImage(nifti_df["mask_index"])
            mask_voxel = sitk.GetArrayFromImage(mask_voxel)
            mask_voxel = torch.from_numpy(mask_voxel).float()
            mask_voxel = mask_voxel.unsqueeze(-1)

            voxel_batch[i] = voxel
            mask_batch[i] = mask_voxel
        
        return voxel_batch, mask_batch

def train_skull_strip_seg_model(nifti_csv_data, epochs=7):
    X_train, X_val, y_train, y_val = train_test_split(nifti_csv_data["intensity_norm"].tolist(), nifti_csv_data["mask_index"].tolist(), test_size=0.1)

    brain_train_dataset = BrainMRIDataset(X_train, y_train, batch_size=128)
    brain_val_dataset = BrainMRIDataset(X_val, y_val, batch_size=128)

    train_loader = DataLoader(brain_train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(brain_val_dataset, batch_size=1, shuffle=False)

    # in_channels=3, out_channels=1 for skull vs non-skull
    unet3d_model = UNet3D(in_channels=3, out_channels=1, init_features=32)

    # compile model with Adam optimizer, focal loss, metrics
    optimizer = optim.Adam(unet3d_model.parameters(), lr=1e-1)
    focal_criterion = FocalLoss(gamma=0.7)
    accuracy = torchmetrics.Accuracy()
    # (num_classes=2 for skull vs non-skull
    iou_score = torchmetrics.IoU(num_classes=2)
    metrics = [iou_score, accuracy]

    # Define desired variables for tracking best validation performance
    best_val_loss = float("inf")
    best_val_metric = float("-inf")

    unet3d_model.train()
    step = 100
    for epoch in range(epochs):
        for batch_idx, (brain_data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = unet3d_model(brain_data)
            loss = focal_criterion(output, target)
            loss.backward()
            optimizer.step()

        # validation
        unet3d_model.eval()
        val_loss = 0.0
        val_metric = 0.0
        with torch.no_grad():
            for val_brain_data, val_target in val_loader:
                val_output = unet3d_model(val_brain_data)
                val_loss += focal_criterion(val_output, val_target)
                val_iou += iou_score(val_output, val_target)
        
        # Calculate average validation loss and IoU score
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        # Update best validation performance and save model if it improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(unet3d_model.state_dict(), "best_unet3d_model_loss_{}.pt".format(step))

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(unet3d_model.state_dict(), "best_unet3d_model_iou_{}.pt".format(step))

        print(f"Epoch {epoch+1}: Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        step += 100

        unet3d_model.train()