import os
import zmq
import time
import pandas as pd
import pickle5 as pickle
import threading
import SimpleITK as sitk

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import torchmetrics
from torchmetrics import Accuracy
from focal_loss.focal_loss import FocalLoss

from torch_model import UNet3D, SimpleUNet3D

# Reference perplexity.ai for pytorch 3D UNet skull strip seg model
# https://www.perplexity.ai/search/0df235a1-27ba-4b67-bf7b-89c2500685c7?s=u

# Also reference perplexity.ai for pytorch train 3D UNet skull strip seg model
# https://www.perplexity.ai/search/cf74b6d5-9888-462b-9063-e90859bbf389
# Refer to section on "MRIDataset(Dataset)"

# Getting two columns from nift csv data pd table
# nifti_csv_data["intensity_norm"] = normalized raw_index
# nifti_csv_data["mask_index"]

def iou_score(pred, target):
    intersection = torch.logical_and(pred, target).sum()
    union = torch.logical_or(pred, target).sum()
    iou = intersection / (union + 1e-7)
    return iou

# Load Skull Strip Seg dataset
# class BrainMRIDataset(torch.utils.data.Dataset):
#     def __init__(self, brain_voxel_list, brain_mask_list, batch_size):
#         self.batch_size = batch_size
#         self.nifti_csv_df = pd.DataFrame({"intensity_norm": brain_voxel_list, "mask_index": brain_mask_list})
#         # do we need to shuffle the rows of the pandas df randomly?
#         self.nifti_csv_df = self.nifti_csv_df.sample(frac=1).reset_index(drop=True)

#     def __len__(self):
#         return len(self.nifti_csv_df) // self.batch_size

#     def __getitem__(self, idx):
#         print("idx = {}".format(idx))
#         print("idx*self.batch_size 1st half = {}".format(idx*self.batch_size))
#         print("(idx+1)*self.batch_size 2nd half = {}".format((idx+1)*self.batch_size))
#         # batch_nifti_df = self.nifti_csv_df.iloc[0*self.batch_size:self.batch_size]
#         batch_nifti_df = self.nifti_csv_df.iloc[idx*self.batch_size:(idx+1)*self.batch_size]
#         # TODO (JG): make sure dims match ResizeCropITK py processor
#         # sitk to torch tensor dims (batch_size, channels, depth, height, width)
#         voxel_batch = torch.zeros((self.batch_size, 1, 96, 128, 160)).float()
#         mask_batch = torch.zeros((self.batch_size, 1, 96, 128, 160)).float()
#         for i, nifti_df in batch_nifti_df.iterrows():
#             print("batch_nifti_df index = {}".format(i))
#             voxel = sitk.ReadImage(nifti_df["intensity_norm"])
#             voxel = sitk.GetArrayFromImage(voxel)
#             voxel = torch.from_numpy(voxel).float()
#             # voxel = voxel.permute(3, 1, 2, 0) # switch the depth at 0 pos and channels at 3 pos
#             voxel = voxel.unsqueeze(0) # add new batch dim at front
#             # print("voxel shape = {}".format(voxel.shape))

#             mask_voxel = sitk.ReadImage(nifti_df["mask_index"])
#             mask_voxel = sitk.GetArrayFromImage(mask_voxel)
#             mask_voxel = torch.from_numpy(mask_voxel).float()
#             mask_voxel = mask_voxel.unsqueeze(0)
#             # print("mask_voxel shape = {}".format(mask_voxel.shape))

#             voxel_batch[i] = voxel
#             mask_batch[i] = mask_voxel
        
#         return voxel_batch, mask_batch

class BrainMRIDataset(torch.utils.data.Dataset):
    def __init__(self, brain_voxel_list, brain_mask_list):
        # self.batch_size = batch_size
        self.nifti_csv_df = pd.DataFrame({"intensity_norm": brain_voxel_list, "mask_index": brain_mask_list})
        # do we need to shuffle the rows of the pandas df randomly?
        self.nifti_csv_df = self.nifti_csv_df.sample(frac=1).reset_index(drop=True)
        # setup transform
        # self.transform = transform

    def __len__(self):
        return len(self.nifti_csv_df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("idx = {}".format(idx))

        # voxel_batch = torch.zeros((self.batch_size, 1, 96, 128, 160)).float()
        # mask_batch = torch.zeros((self.batch_size, 1, 96, 128, 160)).float()

        # sitk to torch tensor dims (channels, depth, height, width)
        voxel = sitk.ReadImage(self.nifti_csv_df["intensity_norm"].iloc[idx])
        voxel = sitk.GetArrayFromImage(voxel)
        voxel = torch.from_numpy(voxel).float()
        voxel = voxel.unsqueeze(0) # add new batch dim at front
        # print("voxel shape = {}".format(voxel.shape))

        mask_voxel = sitk.ReadImage(self.nifti_csv_df["mask_index"].iloc[idx])
        mask_voxel = sitk.GetArrayFromImage(mask_voxel)
        mask_voxel = torch.from_numpy(mask_voxel).float()
        mask_voxel = mask_voxel.unsqueeze(0)
        # print("mask_voxel shape = {}".format(mask_voxel.shape))

            # voxel_batch[i] = voxel
            # mask_batch[i] = mask_voxel
        
        return voxel, mask_voxel

def train_skull_strip_seg_model(nifti_csv_data, epochs=3):
    X_train, X_val, y_train, y_val = train_test_split(nifti_csv_data["intensity_norm"].tolist(), nifti_csv_data["mask_index"].tolist(), test_size=0.1)

    print("X_train len = {}".format(len(X_train)))
    print("X_val len = {}".format(len(X_val)))

    print("Creating brain train & val datasets")
    
    # train_transform = val_transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])

    # brain_train_dataset = BrainMRIDataset(X_train, y_train, transform=train_transform)
    # brain_val_dataset = BrainMRIDataset(X_val, y_val, transform=val_transform)

    brain_train_dataset = BrainMRIDataset(X_train, y_train)
    brain_val_dataset = BrainMRIDataset(X_val, y_val)

    print("brain_train_dataset len = {}".format(len(brain_train_dataset)))
    print("brain_val_dataset len = {}".format(len(brain_val_dataset)))

    print("Creating brain train & val dataloaders")

    train_loader = DataLoader(brain_train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(brain_val_dataset, batch_size=4, shuffle=False)

    brain_voxel_custom, mask_voxel_custom = next(iter(train_loader))

    print("brain_voxel_custom shape = {}".format(brain_voxel_custom.shape))
    print("mask_voxel_custom shape = {}".format(mask_voxel_custom.shape))

    print("Creating Skull Strip Seg UNet3D model")

    # in_channels=1 for 1 medical image modality T2-weighted; out_channels=1 for skull vs non-skull
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet3d_model = UNet3D(in_channels=1, out_channels=1, init_features=4).to(device)

    print("Compiling UNet3D with Adam, FocalLoss, IoU, Accuracy")

    # compile model with Adam optimizer, focal loss, metrics
    optimizer = optim.Adam(unet3d_model.parameters(), lr=1e-1)
    focal_criterion = FocalLoss(gamma=0.7).to(device)
    accuracy = torchmetrics.Accuracy(task="binary").to(device)
    # (num_classes=2 for skull vs non-skull
    # iou_score = torchmetrics.IoU(num_classes=2)
    metrics = [iou_score, accuracy]

    # Define desired variables for tracking best validation performance
    best_val_loss = float("inf")
    best_val_metric = float("-inf")

    print("Training UNet3D across {} epochs".format(epochs))

    unet3d_model.train()
    step = 100
    for epoch in range(epochs):
        print("Epoch {}: Train across batch_idx and brain_data and target from train_loader".format(epoch))
        for batch_idx, (brain_data, target) in enumerate(train_loader):
            brain_data = brain_data.to(device)
            target = target.to(device)
            print("batch_idx = {}; brain_data len = {}; target len = {}".format(batch_idx, len(brain_data), len(target)))
            
            optimizer.zero_grad()
            output = unet3d_model(brain_data)
            loss = focal_criterion(output, target)
            loss.backward()
            optimizer.step()

        print("Running Evaluation")

        # validation
        unet3d_model.eval()
        val_loss = 0.0
        val_metric = 0.0
        with torch.no_grad():
            for val_brain_data, val_target in val_loader:
                val_brain_data = val_brain_data.to(device)
                val_target = val_target.to(device)

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

# References:
# - ChatGPT: https://chat.openai.com/c/78252db9-df19-429d-8473-c2b5a46c38e5

# Reference perplexity.ai for logic where subscriber sends confirmation to publisher when message is received:
# https://www.perplexity.ai/search/cf3ca7f0-a7f2-4395-8952-67198b09aef7?s=u

class ZMQSubscriber:
    def __init__(self, zmq_socket_address="tcp://127.0.0.1:5556"):
        self.zmq_socket_address = zmq_socket_address
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(self.zmq_socket_address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.running = False
        self.receive_thread = None
        self.data_to_process = None # store received data

    def start(self):
        self.running = True
        self.receive_thread = threading.Thread(target=self.receive_data)
        self.receive_thread.start()

    def stop(self):
        self.running = False
        if self.receive_thread:
            self.receive_thread.join()

    def receive_data(self):
        while self.running:
            try:
                # Receive the DataFrame bytes
                dataframe_bytes = self.socket.recv()

                # Deserialize the DataFrame using pickle
                received_dataframe = pickle.loads(dataframe_bytes)

                print("received_dataframe.head = {}".format(received_dataframe.head()))

                self.data_to_process = received_dataframe

                print("Going to send confirmation to NiFi PutZMQ Publisher")
                self.socket.send_string("Received")
            except Exception as e:
                print("Error receiving DataFrame: {}".format(e))

    def __del__(self):
        self.stop()
        self.socket.close()

def main():
    zmq_subscriber = ZMQSubscriber()

    nifti_csv_df = None
    epochs = 7

    try:
        print("Subscribing to {} to receive SkullStripSeg Prep DataFrame...".format(zmq_subscriber.zmq_socket_address))
        zmq_subscriber.start()

        while True:
            # check if there is data to process
            if zmq_subscriber.data_to_process is not None:
                nifti_csv_df = zmq_subscriber.data_to_process

                # pass the data to the training thread
                train_thread = threading.Thread(target=train_skull_strip_seg_model, args=(nifti_csv_df, epochs))
                train_thread.start()

                # reset the data_to_process to None to indicate processing is done
                zmq_subscriber.data_to_process = None
                print("Received data filepaths in pd df to train, finished training, exiting")
                break

            time.sleep(0.2) # sleep for 200 milliseconds

    except KeyboardInterrupt:
        print("Stopping the subscriber...")
        zmq_subscriber.stop()

def main_alt():
    nifti_csv_df = pd.read_csv("skull_strip_seg_prep.csv")
    train_skull_strip_seg_model(nifti_csv_df, epochs=3)

def main_unet():
    # Note: I get an out of gpu memory issue if I use gpu device, it was probably due to not properly handling batches
    # For now use CPU here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet3d_model = SimpleUNet3D(in_channels=1, out_channels=1)

    x = torch.randn((4, 1, 96, 128, 160))
    # x = x.unsqueeze(0)

    preds = unet3d_model(x)

    print(preds.shape)
    print(x.shape)

    assert preds.shape == x.shape

if __name__ == "__main__":
    # main()
    # main_alt()
    main_unet()
