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

from torch_model import SimpleUNet3D
from brain_dataset import StrokeMRIDataset

# Reference perplexity.ai for pytorch 3D UNet skull strip seg model
# https://www.perplexity.ai/search/0df235a1-27ba-4b67-bf7b-89c2500685c7?s=u

# Also reference perplexity.ai for pytorch train 3D UNet skull strip seg model
# https://www.perplexity.ai/search/cf74b6d5-9888-462b-9063-e90859bbf389
# Refer to section on "MRIDataset(Dataset)"

# Hyperparameters
LEARNING_RATE = 1e-1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
VAL_BATCH_SIZE = 1
LOAD_MODEL = False
NUM_EPOCHS = 2
DEBUG = True
TRAINING_TASKNAME = "Stroke Lesion Segmentation"

def iou_score(pred, target):
    intersection = torch.logical_and(pred, target).sum()
    union = torch.logical_or(pred, target).sum()
    iou = intersection / (union + 1e-7)
    return iou


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
# NOTE (JG): This save predictions as segs is like Aladdin's save_predictions_imgs(...), but
# it saves those images in the batch to one image. In that example, batch_size = 16, 
# so 16 objects (cars) were saved to an image.
# This approach would be difficult for our approach. So, what I will do is keep this 
# script to train the stroke lesion seg model here and then deploy it in NiFi, then save it individually.
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

def save_predictions_as_seg_slices(loader, model, dataset_name, folder="saved_seg_slices", device="cuda"):
    # nifti_percent_slices_save = 0.025
    # nifti_filepath_df_row = 0
    nifti_seg_2d_slice_divisor = 2
    nifti_data_type = dataset_name
    nifti_csv_col_name = "stroke_lesion_seg"

    for idx, (x, y) in enumerate(loader):
        if DEBUG:
            print("x.shape = {}".format(x.shape))
            print("y.shape = {}".format(y.shape))
        x = x.to(device=device).unsqueeze(1)
        y = y.unsqueeze(1)

        if DEBUG:
            print("unsequeeze x.shape = {}".format(x.shape))
            print("unsequeeze y.shape = {}".format(y.shape))

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            preds_np = preds.squeeze().cpu().numpy()
            ground_y_np = y.squeeze().cpu().numpy()

        if DEBUG:
            print("preds.shape = {}".format(preds.shape))
            print("preds_np.shape = {}".format(preds_np.shape))
            print("ground_y_np.shape = {}".format(ground_y_np.shape))

        fig, ax = plt.subplots(1, 2, figsize=(14, 10))
        ax[0].set_title("NifTI {} 2D Image ID {} GroundT Slice = {}".format(idx, nifti_data_type, ground_y_np.shape))
        ax[0].imshow(ground_y_np[ground_y_np.shape[0]//nifti_seg_2d_slice_divisor])

        if DEBUG:
            print("ground_y_np.shape[0] = {}".format(ground_y_np.shape[0]))
            print("preds_np.shape[0] = {}".format(preds_np.shape[0]))

        ax[1].set_title("NifTI {} 2D Image ID {} Pred Mask Slice = {}".format(idx, nifti_data_type, preds_np.shape))
        ax[1].imshow(preds_np[preds_np.shape[0]//nifti_seg_2d_slice_divisor])

        # Save the 2D image slice as file
        saved_itk_image_dir = mkdir_prep_dir(folder)
        output_filename = "{}_id_{}_slice_{}.{}".format(nifti_csv_col_name, idx, preds_np.shape[0]//nifti_seg_2d_slice_divisor, "png")
        output_filepath = os.path.join(saved_itk_image_dir, output_filename)
        plt.savefig(output_filepath)

        if DEBUG:
            print("Saved Image to path = {}".format(output_filepath))

    model.train()



def advanced_train_unet3d(nifti_csv_data, dataset_name="icpsr"):
    # TODO (JG): Check if I need to do preprocessing on "stroke_dwi_mask", I think we did this for "mask_index"
    # stroke_dwi_mask. Updated NiFi ResizeCropITKImage Py Processor and ExecuteDNNSkullStrippingSegmentation Processor
    X_train, X_val, y_train, y_val = train_test_split(nifti_csv_data["skull_strip_seg"].tolist(), nifti_csv_data["stroke_mask_index"].tolist(), test_size=0.1)

    if DEBUG:
        print("X_train len = {}".format(len(X_train)))
        print("X_val len = {}".format(len(X_val)))

    print("Creating stroke train & val datasets")
    
    brain_train_dataset = StrokeMRIDataset(X_train, y_train)
    brain_val_dataset = StrokeMRIDataset(X_val, y_val)

    if DEBUG:
        print("brain_train_dataset len = {}".format(len(brain_train_dataset)))
        print("brain_val_dataset len = {}".format(len(brain_val_dataset)))

    print("Creating brain train & val dataloaders")

    # Higher batch_size, we lose gpu memory
    train_loader = DataLoader(brain_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(brain_val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

    print("Creating Stroke Lesion Seg UNet3D model")

    # in_channels=1 for 1 medical image modality T2-weighted; out_channels=1 for skull vs non-skull
    unet3d_model = SimpleUNet3D(in_channels=1, out_channels=1).to(device=DEVICE)

    print("Compiling UNet3D with Adam, BCEWithLogitsLoss, Dice Score, Accuracy")

    # compile model with Adam optimizer, focal loss, metrics
    optimizer = optim.Adam(unet3d_model.parameters(), lr=LEARNING_RATE)
    # focal_criterion = FocalLoss(gamma=0.7).to(device)
    # Use pytorch's built-in focal loss
    bce_criterion = nn.BCEWithLogitsLoss().to(device=DEVICE)

    step = 100
    # if LOAD_MODEL:
    #     load_checkpoint(torch.load("best_unet3d_model_loss_{}.pt".format(step)), unet3d_model)
    #     check_accuracy(val_loader, unet3d_model, device=DEVICE)

    # accuracy = torchmetrics.Accuracy(task="binary").to(device=DEVICE)
    # (num_classes=2 for skull vs non-skull
    # iou_score = torchmetrics.IoU(num_classes=2)
    # metrics = [iou_score, accuracy]

    scaler = torch.cuda.amp.GradScaler()

    # Define desired variables for tracking best validation performance
    # best_val_loss = float("inf")
    # best_val_iou = float("-inf")

    print("Training UNet3D for {} on {} across {} epochs".format(TRAINING_TASKNAME, dataset_name, NUM_EPOCHS))

    # unet3d_model.train()
    step = 100
    for epoch in range(NUM_EPOCHS):
        if DEBUG:
            print("Epoch {}: Train across batch_idx and brain_data and target from train_loader".format(NUM_EPOCHS))
        train_unet3d(train_loader, unet3d_model, optimizer, bce_criterion, scaler)

        # save model
        checkpoint = {
            "state_dict": unet3d_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        mkdir_prep_dir("{}/models".format(dataset_name))
        filename = "{}/models/unet3d_stroke_lesion_seg_{}.pth.tar".format(dataset_name, step)
        save_checkpoint(checkpoint, filename=filename)

        print("Running Evaluation")

        check_accuracy(val_loader, unet3d_model, device=DEVICE)

        # save_predictions_as_segs(
        #     val_loader, unet3d_model, folder="saved_segs", device=DEVICE
        # )

        save_predictions_as_seg_slices(
            val_loader, unet3d_model, dataset_name, folder="{}/saved_seg_slices/{}".format(dataset_name, step), device=DEVICE
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
    dataset_name="icpsr"

    try:
        print("Subscribing to {} to receive StrokeLesionPrep Prep DataFrame...".format(zmq_subscriber.zmq_socket_address))
        zmq_subscriber.start()

        while True:
            # check if there is data to process
            if zmq_subscriber.data_to_process is not None:
                nifti_csv_df = zmq_subscriber.data_to_process

                # pass the data to the training thread
                train_thread = threading.Thread(target=advanced_train_unet3d, args=(nifti_csv_df, dataset_name))
                train_thread.start()

                # reset the data_to_process to None to indicate processing is done
                zmq_subscriber.data_to_process = None
                print("Received data filepaths in pd df to train, finished training, exiting")
                break

            time.sleep(0.2) # sleep for 200 milliseconds

    except KeyboardInterrupt:
        print("Stopping the subscriber...")
        zmq_subscriber.stop()

# def nfbs_skull_strip_seg_unet_training():
#     nifti_csv_df = pd.read_csv("nfbs/prep/nfbs_skull_strip_seg_prep.csv")
#     advanced_train_unet3d(nifti_csv_df, dataset_name="nfbs")

def icpsr_stroke_lesion_seg_unet_training():
    nifti_csv_df = pd.read_csv("icpsr/prep/icpsr_stroke_lesion_seg_prep.csv")
    advanced_train_unet3d(nifti_csv_df, dataset_name="icpsr")


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
    # nfbs_skull_strip_seg_unet_training()
    icpsr_stroke_lesion_seg_unet_training()