import os
import zmq
import time
import pandas as pd
import pickle5 as pickle
import threading
import SimpleITK as sitk

from typing import Tuple

from tqdm import tqdm
import matplotlib.pyplot as plt

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
from brain_dataset import BrainMRIDataset

# Reference perplexity.ai for pytorch 3D UNet skull strip seg model
# https://www.perplexity.ai/search/0df235a1-27ba-4b67-bf7b-89c2500685c7?s=u

# Also reference perplexity.ai for pytorch train 3D UNet skull strip seg model
# https://www.perplexity.ai/search/cf74b6d5-9888-462b-9063-e90859bbf389
# Refer to section on "MRIDataset(Dataset)"

# Hyperparameters
LEARNING_RATE = 1e-1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
LOAD_MODEL = False
NUM_EPOCHS = 3

def iou_score(pred, target):
    intersection = torch.logical_and(pred, target).sum()
    union = torch.logical_or(pred, target).sum()
    iou = intersection / (union + 1e-7)
    return iou

# Simple Training for SimpleUNet3D function
def simple_train_unet3d(nifti_csv_data, epochs=3):
    X_train, X_val, y_train, y_val = train_test_split(nifti_csv_data["intensity_norm"].tolist(), nifti_csv_data["mask_index"].tolist(), test_size=0.1)

    print("X_train len = {}".format(len(X_train)))
    print("X_val len = {}".format(len(X_val)))

    print("Creating brain train & val datasets")
    
    brain_train_dataset = BrainMRIDataset(X_train, y_train)
    brain_val_dataset = BrainMRIDataset(X_val, y_val)

    print("brain_train_dataset len = {}".format(len(brain_train_dataset)))
    print("brain_val_dataset len = {}".format(len(brain_val_dataset)))

    print("Creating brain train & val dataloaders")

    train_loader = DataLoader(brain_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(brain_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Creating Skull Strip Seg UNet3D model")

    # in_channels=1 for 1 medical image modality T2-weighted; out_channels=1 for skull vs non-skull
    unet3d_model = SimpleUNet3D(in_channels=1, out_channels=1).to(DEVICE)
    # unet3d_model = SimpleUNet().to(device)

    print("Compiling UNet3D with Adam, FocalLoss, IoU, Accuracy")

    # compile model with Adam optimizer, focal loss, metrics
    bce_criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = optim.Adam(unet3d_model.parameters(), lr=LEARNING_RATE)

    # Define desired variables for tracking best validation performance
    best_val_loss = float("inf")

    print("Training UNet3D across {} epochs".format(epochs))

    unet3d_model.train()
    step = 100
    for epoch in range(epochs):
        print("Epoch {}: Train across batch_idx and brain_data and target from train_loader".format(epoch))
        for batch_idx, (brain_data, target) in enumerate(train_loader):
            brain_data = brain_data.to(DEVICE)
            target = target.to(DEVICE)
            print("batch_idx = {}; brain_data len = {}; target len = {}".format(batch_idx, len(brain_data), len(target)))
            
            optimizer.zero_grad()
            outputs = unet3d_model(brain_data.unsqueeze(1))
            loss = bce_criterion(outputs, target.unsqueeze(1))
            loss.backward()
            optimizer.step()

        print("Running Evaluation")

        # validation
        unet3d_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_brain_data, val_target in val_loader:
                val_brain_data = val_brain_data.to(DEVICE)
                val_target = val_target.to(DEVICE)

                val_output = unet3d_model(val_brain_data.unsqueeze(1))
                val_loss += bce_criterion(val_output, val_target.unsqueeze(1))
        
        # Calculate average validation loss and IoU score
        val_loss /= len(val_loader)

        # Update best validation performance and save model if it improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(unet3d_model.state_dict(), "best_unet3d_model_loss_{}.pt".format(step))

        print(f"--------------------Epoch {epoch+1}: Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")

        step += 100

        unet3d_model.train()

def train_unet3d(train_loader, model, optimizer, loss_fn, scaler):
    train_loop = tqdm(train_loader)

    for batch_idx, (brain_data, targets) in enumerate(train_loop):
        brain_data = brain_data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        # if debug:
        #     print("batch_idx = {}; brain_data len = {}; targets len = {}".format(batch_idx, len(brain_data), len(targets)))
            
        # forward float16 training, reducing VRAM and speed up training
        # with torch.cuda.amp.autocast():
            # brain_data.unsqueeze(1)
        # output = model(brain_data.float().unsqueeze(1))
        # loss = loss_fn(output, targets.float().unsqueeze(1))
        # train_iou = iou_score(output, targets.float().unsqueeze(1))

        optimizer.zero_grad()
        output = model(brain_data.unsqueeze(1))
        loss = loss_fn(output, targets.unsqueeze(1))
        train_iou = iou_score(output, targets.unsqueeze(1))

        # backward

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        train_loop.set_postfix(loss=loss.item())

def validate_unet3d():
        # validation
        unet3d_model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for val_brain_data, val_target in val_loader:
                val_brain_data = val_brain_data.to(DEVICE)
                val_target = val_target.to(DEVICE)

                val_output = unet3d_model(val_brain_data.unsqueeze(1))
                # val_output = torch.sigmoid(val_output)
                val_loss += bce_criterion(val_output, val_target.unsqueeze(1))
                val_iou += iou_score(val_output, val_target.unsqueeze(1))
        
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

def save_checkpoint(state, filename="best_unet3d_model.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
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

    model.train()

def mkdir_prep_dir(dirpath):
    """make preprocess directory if doesn't exist"""
    prep_dir = dirpath
    if not os.path.exists(prep_dir):
        os.makedirs(prep_dir)
    return prep_dir

# TODO (JG): Add check if dir not exist, create it, else it fails
def save_predictions_as_segs(loader, model, folder="saved_segs", device="cuda"):
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device).unsqueeze(1)
        y = y.unsqueeze(1)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        preds_np = preds.squeeze().cpu().numpy()
        preds_sitk = sitk.GetImageFromArray(preds_np)
        ground_sitk = sitk.GetImageFromArray(y.squeeze().cpu().numpy())

        mkdir_prep_dir(folder)

        sitk.WriteImage(preds_sitk, f"{folder}/pred_{idx}.nii.gz")
        sitk.WriteImage(ground_sitk, f"{folder}/{idx}.nii.gz")

    model.train()

def save_predictions_as_seg_slices(loader, model, folder="saved_seg_slices", device="cuda"):
    # nifti_percent_slices_save = 0.025
    # nifti_filepath_df_row = 0
    nifti_seg_2d_slice_divisor = 2
    nifti_data_type = "nfbs"
    nifti_csv_col_name = "skull_strip_seg"

    for idx, (x, y) in enumerate(loader):
        # check batch size > 1, then run predictions and select slice from 96 slices of an MRI per MRI
        # else after prediction, I noticed the order is different and causes an issue choosing slice
        if x.shape[0] > 1 and y.shape[0]:
            print("x.shape = {}".format(x.shape))
            print("y.shape = {}".format(y.shape))
            x = x.to(device=device).unsqueeze(1)
            print("unsequeeze x.shape = {}".format(x.shape))
            y = y.unsqueeze(1)
            print("unsequeeze y.shape = {}".format(y.shape))
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()

            print("preds.shape = {}".format(preds.shape))
            preds_np = preds.squeeze().cpu().numpy()
            preds_sitk = sitk.GetImageFromArray(preds_np)
            ground_sitk = sitk.GetImageFromArray(y.squeeze().cpu().numpy())

            print("ground_sitk.GetSize() = {}".format(ground_sitk.GetSize()))
            print("preds_sitk.GetSize() = {}".format(preds_sitk.GetSize()))        

            # if ground_sitk.GetSize() == preds_sitk.GetSize():
                # Create a figure and axis for visualization
            fig, ax = plt.subplots(1, 2, figsize=(14, 10))
            ax[0].set_title("NifTI {} 2D Image ID {} GroundT Slice = {}".format(idx, nifti_data_type, ground_sitk.GetSize()))
            # Display the 2D image slice
            print("ground_sitk.GetSize()[1] = {}".format(ground_sitk.GetSize()[1]))
            ax[0].imshow(ground_sitk[ground_sitk.GetSize()[1]//nifti_seg_2d_slice_divisor])

            print("preds_sitk.GetSize()[1] = {}".format(preds_sitk.GetSize()[1]))
            ax[1].set_title("NifTI {} 2D Image ID {} Mask Slice = {}".format(idx, nifti_data_type, preds_sitk.GetSize()))
            ax[1].imshow(preds_sitk[preds_sitk.GetSize()[1]//nifti_seg_2d_slice_divisor])

            # Save the 2D image slice as file
            saved_itk_image_dir = mkdir_prep_dir(folder)
            output_filename = "nifti_image_id_{}_slice_{}_{}.{}".format(idx, preds_sitk.GetSize()[1]//nifti_seg_2d_slice_divisor, nifti_csv_col_name, "png")
            output_filepath = os.path.join(saved_itk_image_dir, output_filename)
            print("Saving Image to path = {}".format(output_filepath))
            plt.savefig(output_filepath)

            # sitk.WriteImage(preds_sitk, f"{folder}/pred_{idx}.nii.gz")
            # sitk.WriteImage(ground_sitk, f"{folder}/{idx}.nii.gz")
        # else:
        #     print("GroundT != Preds SimpleITK voxel")

    model.train()



def advanced_train_unet3d(nifti_csv_data, epochs=3, debug=False):
    X_train, X_val, y_train, y_val = train_test_split(nifti_csv_data["intensity_norm"].tolist(), nifti_csv_data["mask_index"].tolist(), test_size=0.1)

    if debug:
        print("X_train len = {}".format(len(X_train)))
        print("X_val len = {}".format(len(X_val)))

    print("Creating brain train & val datasets")
    
    brain_train_dataset = BrainMRIDataset(X_train, y_train)
    brain_val_dataset = BrainMRIDataset(X_val, y_val)

    if debug:
        print("brain_train_dataset len = {}".format(len(brain_train_dataset)))
        print("brain_val_dataset len = {}".format(len(brain_val_dataset)))

    print("Creating brain train & val dataloaders")

    # Higher batch_size, we lose gpu memory
    train_loader = DataLoader(brain_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(brain_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Creating Skull Strip Seg UNet3D model")

    # in_channels=1 for 1 medical image modality T2-weighted; out_channels=1 for skull vs non-skull
    unet3d_model = SimpleUNet3D(in_channels=1, out_channels=1).to(device=DEVICE)

    print("Compiling UNet3D with Adam, BCEWithLogitsLoss, Dice Score, Accuracy")

    # compile model with Adam optimizer, focal loss, metrics
    optimizer = optim.Adam(unet3d_model.parameters(), lr=LEARNING_RATE)
    # focal_criterion = FocalLoss(gamma=0.7).to(device)
    # Use pytorch's built-in focal loss
    bce_criterion = nn.BCEWithLogitsLoss().to(device=DEVICE)

    step = 100
    if LOAD_MODEL:
        load_checkpoint(torch.load("best_unet3d_model_loss_{}.pt"))
        check_accuracy(val_loader, unet3d_model, device=DEVICE)

    # accuracy = torchmetrics.Accuracy(task="binary").to(device=DEVICE)
    # (num_classes=2 for skull vs non-skull
    # iou_score = torchmetrics.IoU(num_classes=2)
    # metrics = [iou_score, accuracy]

    scaler = torch.cuda.amp.GradScaler()

    # Define desired variables for tracking best validation performance
    # best_val_loss = float("inf")
    # best_val_iou = float("-inf")

    print("Training UNet3D across {} epochs".format(NUM_EPOCHS))

    # unet3d_model.train()
    step = 100
    for epoch in range(NUM_EPOCHS):
        if debug:
            print("Epoch {}: Train across batch_idx and brain_data and target from train_loader".format(NUM_EPOCHS))
        train_unet3d(train_loader, unet3d_model, optimizer, bce_criterion, scaler)

        # save model
        checkpoint = {
            "state_dict": unet3d_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        filename = "best_unet3d_model_{}.pth.tar".format(step)
        save_checkpoint(checkpoint, filename=filename)

        print("Running Evaluation")

        check_accuracy(val_loader, unet3d_model, device=DEVICE)

        # save_predictions_as_segs(
        #     val_loader, unet3d_model, folder="saved_segs", device=DEVICE
        # )

        save_predictions_as_seg_slices(
            val_loader, unet3d_model, folder="saved_seg_slices", device=DEVICE
        )

        step += 100

        # print(f"--------------------Epoch {epoch+1}: Train Loss: {loss:.4f} | Train IoU: {train_iou: .4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        # unet3d_model.train()

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

def adv_skull_strip_seg_unet_training():
    debug = False
    nifti_csv_df = pd.read_csv("nfbs_skull_strip_seg_prep.csv")
    advanced_train_unet3d(nifti_csv_df, epochs=7)

def simple_skull_strip_seg_unet_training():
    nifti_csv_df = pd.read_csv("nfbs_skull_strip_seg_prep.csv")
    simple_train_unet3d(nifti_csv_df, epochs=2)

def test_unet():
    # Note: I get an out of gpu memory issue if I use gpu device, it was probably due to not properly handling batches
    # For now use CPU here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet3d_model = SimpleUNet3D(in_channels=1, out_channels=1)

    batch_size = 4
    channels = 1
    depth = 96
    height = 128
    width = 160

    x = torch.randn((batch_size, channels, depth, height, width))

    preds = unet3d_model(x)

    print(preds.shape)
    print(x.shape)

    assert preds.shape == x.shape

if __name__ == "__main__":
    # test_unet()
    # simple_skull_strip_seg_unet_training()
    adv_skull_strip_seg_unet_training()
