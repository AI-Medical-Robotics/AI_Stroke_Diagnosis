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
from matplotlib.font_manager import FontProperties

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
VAL_BATCH_SIZE = 1
LOAD_MODEL = False
NUM_EPOCHS = 6
DEBUG = False


TRAINING_TASKNAME = "Skull Stripping Segmentation"

USE_PRETRAINED_MODEL=False

MODEL_NAME = "Basic_UNet3D"


# Average Loss values Per Epoch for loss curve
train_bce_loss_avg_values = []
val_bce_loss_avg_values = []

# Average IoU values Per Epoch for segmentation performance metrics
train_iou_avg_values = []
val_iou_avg_values = []

# Average Dice values Per Epoch for segmentation performance metrics
train_dice_avg_values = []
val_dice_avg_values = []


def iou_score(pred, target):
    intersection = torch.logical_and(pred, target).sum()
    union = torch.logical_or(pred, target).sum()
    iou = intersection / (union + 1e-7)
    return iou

def dice_score(pred, target):
    dice_score = (2 * (pred * target).sum()) / (
        (pred + target).sum() + 1e-8
    )
    return dice_score



def save_checkpoint(state, filename="best_unet3d_model.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

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
# This approach would be difficult for our approach. So, what I will be is keep this 
# script to train the skull strip seg model here and then deploy it in NiFi, then save it individually.
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



def train_unet3d(train_loader, unet3d_model, optimizer, bce_criterion, step, epoch):
    train_loop = tqdm(train_loader)

    train_bce_loss_values = []
    train_iou_values = []
    train_dice_values = []

    print(f"Running train_unet3d {epoch}")
    for batch_idx, (brain_data, targets) in enumerate(train_loop):
        brain_data = brain_data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        # if debug:
        #     print("batch_idx = {}; brain_data len = {}; targets len = {}".format(batch_idx, len(brain_data), len(targets)))
            
        optimizer.zero_grad()
        output = unet3d_model(brain_data.unsqueeze(1))
        bce_loss = bce_criterion(output, targets.unsqueeze(1))

        voxel_skull_strip_pred = torch.sigmoid(output)
        voxel_skull_strip_pred = (voxel_skull_strip_pred > 0.5).float()

        train_iou = iou_score(voxel_skull_strip_pred, targets.unsqueeze(1))
        train_dice = dice_score(voxel_skull_strip_pred, targets.unsqueeze(1))

        bce_loss.backward()
        optimizer.step()

        train_bce_loss_values.append(bce_loss.item())
        train_iou_values.append(train_iou.item())
        train_dice_values.append(train_dice.item())

        # update tqdm loop
        train_loop.set_postfix(train_bce_loss=bce_loss.item(), train_iou=train_iou.item(), train_dice=train_dice.item())

        step += 1

    if NUM_EPOCHS > 1:
        print(f"NUM_EPOCHS = {NUM_EPOCHS}, so append avg metrics")
        # TODO (JG): Get average metrics per epoch
        train_bce_loss_avg_values.append( sum(train_bce_loss_values)/len(train_loader) )
        train_iou_avg_values.append( sum(train_iou_values)/len(train_loader) )
        train_dice_avg_values.append( sum(train_dice_values)/len(train_loader) )
    else:
        # TODO (JG): For Debug, Assign metrics lists to global metrics lists for steps across 1 epoch
        train_bce_loss_avg_values.extend(train_bce_loss_values)
        train_iou_avg_values.extend(train_iou_values)
        train_dice_avg_values.extend(train_dice_values)

    return step





def validate_unet3d(val_loader, unet3d_model, bce_criterion, epoch):
    val_loop = tqdm(val_loader)
    print(f"Running validate_unet3d {epoch}")

    val_bce_loss_values = []
    val_iou_values = []
    val_dice_values = []

    with torch.no_grad():
        for batch_idx, (prep_voxel, voxel_skull_strip_gt) in enumerate(val_loop):

            prep_voxel = prep_voxel.to(device=DEVICE).unsqueeze(1)
            voxel_skull_strip_gt = voxel_skull_strip_gt.to(device=DEVICE).unsqueeze(1)

            voxel_skull_strip_ext_fets = unet3d_model(prep_voxel)

            voxel_skull_strip_pred = torch.sigmoid(voxel_skull_strip_ext_fets)
            voxel_skull_strip_pred = (voxel_skull_strip_pred > 0.5).float()

            val_bce_loss = bce_criterion(voxel_skull_strip_ext_fets, voxel_skull_strip_gt)

            # val_iou = iou_score(voxel_skull_strip_ext_fets, voxel_skull_strip_gt)
            # val_dice = dice_score(voxel_skull_strip_ext_fets, voxel_skull_strip_gt)

            val_iou = iou_score(voxel_skull_strip_pred, voxel_skull_strip_gt)
            val_dice = dice_score(voxel_skull_strip_pred, voxel_skull_strip_gt)

            val_bce_loss_values.append(val_bce_loss.item())
            val_iou_values.append(val_iou.item())
            val_dice_values.append(val_dice.item())

            val_loop.set_postfix(val_bce_loss=val_bce_loss.item(), val_iou=val_iou.item(), val_dice=val_dice.item())

    if NUM_EPOCHS > 1:
        print(f"NUM_EPOCHS = {NUM_EPOCHS}, so append avg metrics")
        # TODO (JG): Get average metrics per epoch
        val_bce_loss_avg_values.append( sum(val_bce_loss_values)/len(val_loader) )
        val_iou_avg_values.append( sum(val_iou_values)/len(val_loader) )
        val_dice_avg_values.append( sum(val_dice_values)/len(val_loader) )
    else:
        # TODO (JG): For Debug, Assign metrics lists to global metrics lists for steps across 1 epoch
        val_bce_loss_avg_values.extend(val_bce_loss_values)
        val_iou_avg_values.extend(val_iou_values)
        val_dice_avg_values.extend(val_dice_values)



def train_skull_strip_seg_over_epochs(unet3d_model, optimizer, bce_criterion, train_loader, val_loader, dataset_name, dst_folder):
    print("Training UNet3D on {} across {} epochs".format(dataset_name, NUM_EPOCHS))

    unet3d_model.train()
    step = 0
    for epoch in range(1, NUM_EPOCHS+1):
        if DEBUG:
            print("Epoch {}: Train across batch_idx and brain_data and target from train_loader".format(NUM_EPOCHS))

        step = train_unet3d(train_loader, unet3d_model, optimizer, bce_criterion, step, epoch)

        # save model
        checkpoint = {
            "state_dict": unet3d_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        }

        mkdir_prep_dir(dst_folder)
        save_model_filepath = f"{dst_folder}/unet3d_skull_strip_seg_{step}.pth.tar"
        save_checkpoint(checkpoint, filename=save_model_filepath)

        validate_unet3d(val_loader, unet3d_model, bce_criterion, epoch)

    return step



# TODO (JG): Refer back to plot ICPSR method in SaveITKImageSlice
def save_mri_skull_strip_seg_slices(val_loader, unet3d_model, dataset_name, folder="save_mri_skull_strip_seg_slices"):
    nifti_seg_2d_slice_divisor = 2
    nifti_data_type = dataset_name
    nifti_csv_col_name = "skull_strip_seg"
    skull_classifier = ["skull strip", "not visible"]

    with torch.no_grad():
        for batch_idx, (prep_voxel, voxel_skull_strip_gt) in enumerate(val_loader):

            if DEBUG:
                print("prep_voxel.shape = {}".format(prep_voxel.shape))
                print("voxel_skull_strip_gt.shape = {}".format(voxel_skull_strip_gt.shape))
            prep_voxel = prep_voxel.to(device=DEVICE).unsqueeze(1)
            voxel_skull_strip_gt = voxel_skull_strip_gt.unsqueeze(1)

            if DEBUG:
                print("unsequeeze prep_voxel.shape = {}".format(prep_voxel.shape))
                print("unsequeeze voxel_skull_strip_gt.shape = {}".format(voxel_skull_strip_gt.shape))

            voxel_skull_strip_ext_fets = unet3d_model(prep_voxel)
            voxel_skull_strip_pred = torch.sigmoid(voxel_skull_strip_ext_fets)
            voxel_skull_strip_pred = (voxel_skull_strip_pred > 0.5).float()

            # Crop Skull From Brain Voxel
            voxel_brain_tissue_pred = (voxel_skull_strip_pred > 0.25).float()
            voxel_brain_tissue_crop = voxel_brain_tissue_pred * prep_voxel.float()
            
            voxel_brain_tissue_crop_np = voxel_brain_tissue_crop.squeeze().cpu().numpy()
            voxel_skull_strip_pred_np = voxel_skull_strip_pred.squeeze().cpu().numpy()
            voxel_prep_for_seg_np = prep_voxel.squeeze().cpu().numpy()
            voxel_skull_strip_gt_np = voxel_skull_strip_gt.squeeze().cpu().numpy()

            if DEBUG:
                print("voxel_brain_tissue_crop.shape = {}".format(voxel_brain_tissue_crop.shape))
                print("voxel_brain_tissue_crop_np.shape = {}".format(voxel_brain_tissue_crop_np.shape))
                print("voxel_skull_strip_pred.shape = {}".format(voxel_skull_strip_pred.shape))
                print("voxel_skull_strip_pred_np.shape = {}".format(voxel_skull_strip_pred_np.shape))
                print("voxel_skull_strip_gt_np.shape = {}".format(voxel_skull_strip_gt_np.shape))


            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            gt_slice_id = voxel_skull_strip_gt_np.shape[0]//nifti_seg_2d_slice_divisor
            fig.suptitle(f"{nifti_data_type.upper()}: Stroke MRI {batch_idx} 2D Slice {gt_slice_id} Skull Strip Segmentation")

            prep_slice_id = voxel_prep_for_seg_np.shape[0]//nifti_seg_2d_slice_divisor
            hrd_caption0 = f"Prep: Brain Skull Tissue"
            ax[0, 0].set_title(hrd_caption0)
            ax[0, 0].imshow(voxel_prep_for_seg_np[prep_slice_id])


            hrd_caption1 = f"GT Mask: Shape: {voxel_skull_strip_gt_np.shape}"
            ax[0, 1].set_title(hrd_caption1)
            ax[0, 1].imshow(voxel_skull_strip_gt_np[gt_slice_id])



            if DEBUG:
                print("voxel_skull_strip_gt_np.shape[0] = {}".format(voxel_skull_strip_gt_np.shape[0]))
                print("voxel_skull_strip_pred_np.shape[0] = {}".format(voxel_skull_strip_pred_np.shape[0]))

            pred_slice_id = voxel_skull_strip_pred_np.shape[0]//nifti_seg_2d_slice_divisor
            hrd_caption2 = f"Pred Mask: Shape: {voxel_skull_strip_pred_np.shape}"
            ax[1, 0].set_title(hrd_caption2)
            ax[1, 0].imshow(voxel_skull_strip_pred_np[pred_slice_id])

            crop_pred_slice_id = voxel_brain_tissue_crop_np.shape[0]//nifti_seg_2d_slice_divisor
            hrd_caption3 = f"Crop Pred: Brain Tissue"
            ax[1, 1].set_title(hrd_caption3)
            ax[1, 1].imshow(voxel_brain_tissue_crop_np[crop_pred_slice_id])

            # Adjust the spacing between subplots and figure edges
            plt.subplots_adjust(top=0.85)

            # Save the 2D image slice as file
            save_mri_skull_strip_seg_slices_dir = mkdir_prep_dir(folder)
            output_filename = "{}_{}_slice_id_{}.{}".format(nifti_csv_col_name, batch_idx, pred_slice_id, "jpg")
            output_filepath = os.path.join(save_mri_skull_strip_seg_slices_dir, output_filename)
            plt.savefig(output_filepath)
            plt.close()

            if DEBUG:
                print("Saved Image to path = {}".format(output_filepath))



def qual_eval_skull_strip_seg(unet3d_model, val_loader, dataset_name, model_filepath, dst_folder):
    unet3d_model.eval()
    
    if USE_PRETRAINED_MODEL:
        load_checkpoint(torch.load(model_filepath), unet3d_model)

    save_mri_skull_strip_seg_slices(val_loader, unet3d_model, dataset_name, folder=dst_folder)


def plot_unet3d_loss_curve(bce_loss_values, plot_title, plot_filename):
    f, ax = plt.subplots()

    ax.set_title(plot_title)
    ax.plot(bce_loss_values, color="blue", label="CE Loss")
    ax.grid(True)
    if NUM_EPOCHS > 1:
        labelx_text = "Epochs"
    else:
        labelx_text = "Steps Across Epochs"
    ax.set_xlabel(labelx_text)
    ax.set_ylabel("BCE Loss")
    ax.legend()
    plt.savefig(plot_filename)
    plt.close()


def plot_unet3d_2_loss_curves(train_bce_loss_values, val_bce_loss_values, plot_title, plot_filename):
    f, ax = plt.subplots()
    ax.set_title(plot_title)
    ax.plot(train_bce_loss_values, color="blue", label="Train BCE Loss")
    ax.plot(val_bce_loss_values, color="gold", label="Valid BCE Loss")
    ax.grid(True)
    if NUM_EPOCHS > 1:
        labelx_text = "Epochs"
    else:
        labelx_text = "Steps Across Epochs"
    ax.set_xlabel(labelx_text)
    ax.set_ylabel("BCE Loss")
    ax.legend()
    plt.savefig(plot_filename)
    plt.close()


def save_loss_values_to_pd_csv(filepath, column_name1, column_name2, train_bce_loss_values, val_bce_loss_values):
    loss_values_df = pd.DataFrame(columns = [column_name1, column_name2])

    val_bce_loss_values_cp = []
    val_bce_loss_values_cp.extend(val_bce_loss_values)

    # NOTE: there are more train losses, so setting remaining val losses to 0 for pd df
    val_bce_loss_values_cp += ["NA"] * (len(train_bce_loss_values) - len(val_bce_loss_values))
    loss_values_df[column_name1] = train_bce_loss_values
    loss_values_df[column_name2] = val_bce_loss_values_cp

    loss_values_df.to_csv(filepath, index=False)


def plot_unet3d_metric_score_curve(seg_metric_name, metric_score_values, plot_title, plot_filename):
    f, ax = plt.subplots()
    ax.set_title(plot_title)
    ax.plot(metric_score_values, color="blue", label=f"{seg_metric_name} Score")
    ax.grid(True)
    if NUM_EPOCHS > 1:
        labelx_text = "Epochs"
    else:
        labelx_text = "Steps Across Epochs"
    ax.set_xlabel(labelx_text)
    ax.set_ylabel(f"{seg_metric_name} Score")
    ax.legend()
    plt.savefig(plot_filename)
    plt.close()



def plot_unet3d_2_seg_metric_curves(iou_score_values, dice_score_values, plot_title, plot_filename):
    f, ax = plt.subplots()
    ax.set_title(plot_title)
    ax.plot(iou_score_values, color="blue", label="Valid IoU")
    ax.plot(dice_score_values, color="gold", label="Valid Dice")
    ax.grid(True)
    if NUM_EPOCHS > 1:
        labelx_text = "Epochs"
    else:
        labelx_text = "Steps Across Epochs"
    ax.set_xlabel(labelx_text)
    ax.set_ylabel("Valid IoU & Dice Scores")
    ax.legend()
    plt.savefig(plot_filename)
    plt.close()


def plot_unet3d_metrics_table(val_bce_loss_values, val_iou_score_values, val_dice_score_values, plot_filename, plot_title, col_prefix, epoch_steps=None):
    highest_dice_score = val_dice_score_values.index(max(val_dice_score_values))
    highest_iou_score = val_iou_score_values.index(max(val_iou_score_values))
    lowest_loss_score = val_bce_loss_values.index(min(val_bce_loss_values))

    if NUM_EPOCHS > 1:
        # Number of Epochs, mainly use this one
        epochs = [f"Epoch {idx}" for idx in range(NUM_EPOCHS)]
        epoch_colname = "Epoch"
    elif epoch_steps is not None:
        # Total Steps across all Epochs, it is total iterations after train_loader finishes; only testing for 1 epoch
        epochs = [f"Epoch Step {idx}" for idx in range(epoch_steps)] 
        epoch_colname = "Epoch (Step)"

    df = pd.DataFrame({
    epoch_colname: [epochs[highest_dice_score], epochs[highest_iou_score], epochs[lowest_loss_score]],
    f"{col_prefix} Dice Score": [f"{max(val_dice_score_values):.4f}", None, None],
    f"{col_prefix} IoU Score": [None, f"{max(val_iou_score_values):.4f}", None],
    f"{col_prefix} BCE Loss": [None, None, f"{min(val_bce_loss_values):.4f}"]
    })

    df[f"{MODEL_NAME}"] = df.columns[1:]

    df = df[[f"{MODEL_NAME}", epoch_colname, f"{col_prefix} Dice Score", f"{col_prefix} IoU Score", f"{col_prefix} BCE Loss"]]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axis("off")

    colors = [['white', 'lightgrey', 'white', 'lightgrey', 'white'],
              ['lightgrey', 'white', 'lightgrey', 'white', 'lightgrey'],
              ['white', 'lightgrey', 'white', 'lightgrey', 'white']]

    # Create the table
    font_properties = FontProperties(weight="bold")
    table = ax.table(cellText = df.values, cellColours=colors, colLabels = df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title(plot_title)

    # Adjust the spacing between subplots and figure edges
    plt.subplots_adjust(top=0.85)

    for (i, j), cell in table._cells.items():
        cell.set_text_props(fontproperties=font_properties)

    plt.savefig(plot_filename, bbox_inches = "tight")
    plt.close()

def save_2_metrics_scores_to_pd_csv(filepath, column_name1, column_name2, iou_score_values, dice_score_values):
    dice_score_values_cp = []
    dice_score_values_cp.extend(dice_score_values)

    metric_scores_df = pd.DataFrame(columns = [column_name1, column_name2])
    # NOTE: there are more train losses, so setting remaining val losses to 0 for pd df
    dice_score_values_cp += ["NA"] * (len(iou_score_values) - len(dice_score_values))
    metric_scores_df[column_name1] = iou_score_values
    metric_scores_df[column_name2] = dice_score_values_cp

    metric_scores_df.to_csv(filepath, index=False)

def save_metric_scores_to_pd_csv(filepath, column_name, metric_score_values):
    new_metric_score_df = pd.DataFrame(columns = [column_name])
    new_metric_score_df[column_name] = metric_score_values
    new_metric_score_df.to_csv(filepath, index=False)

def save_visualized_skull_strip_avg_metrics(unet3d_model, val_loader, dataset_name, step):

    mkdir_prep_dir(f"icpsr/models/{MODEL_NAME.lower()}/loss_curves/")
    plot_unet3d_loss_curve(train_bce_loss_avg_values, f"{MODEL_NAME} Train Avg Loss Curve", f"icpsr/models/{MODEL_NAME.lower()}/loss_curves/{MODEL_NAME.lower()}_train_avg_loss_curve.jpg")
    plot_unet3d_loss_curve(val_bce_loss_avg_values,  f"{MODEL_NAME} Valid Avg Loss Curve", f"icpsr/models/{MODEL_NAME.lower()}/loss_curves/{MODEL_NAME.lower()}_val_avg_loss_curve.jpg")
    plot_unet3d_2_loss_curves(train_bce_loss_avg_values, val_bce_loss_avg_values, f"{MODEL_NAME} Avg Loss Curves", f"icpsr/models/{MODEL_NAME.lower()}/loss_curves/{MODEL_NAME.lower()}_train_val_avg_loss_curves.jpg")
    
    save_loss_values_to_pd_csv(f"icpsr/models/{MODEL_NAME.lower()}/loss_curves/{MODEL_NAME.lower()}_train_val_loss_avg_values.csv", "train_avg_bce_loss", "val_avg_bce_loss", train_bce_loss_avg_values, val_bce_loss_avg_values)

    mkdir_prep_dir(f"icpsr/models/{MODEL_NAME.lower()}/iou_score_curves/")
    mkdir_prep_dir(f"icpsr/models/{MODEL_NAME.lower()}/dice_score_curves/")
    mkdir_prep_dir(f"icpsr/models/{MODEL_NAME.lower()}/seg_metric_score_curves/")


    plot_unet3d_metric_score_curve("Valid IoU", val_iou_avg_values, f"{MODEL_NAME} Valid Avg IoU Score Curve", f"icpsr/models/{MODEL_NAME.lower()}/iou_score_curves/{MODEL_NAME.lower()}_val_avg_iou_curve.jpg")
    plot_unet3d_metric_score_curve("Valid Dice", val_dice_avg_values, f"{MODEL_NAME} Valid Avg Dice Score Curve", f"icpsr/models/{MODEL_NAME.lower()}/dice_score_curves/{MODEL_NAME.lower()}_val_avg_dice_curve.jpg")

    plot_unet3d_2_seg_metric_curves(val_iou_avg_values, val_dice_avg_values, f"{MODEL_NAME} Valid Avg IoU & Dice Score Curves", f"icpsr/models/{MODEL_NAME.lower()}/seg_metric_score_curves/{MODEL_NAME.lower()}_val_avg_iou_dice_curves.jpg")

    save_2_metrics_scores_to_pd_csv(f"icpsr/models/{MODEL_NAME.lower()}/seg_metric_score_curves/{MODEL_NAME.lower()}_val_avg_iou_dice_scores.csv", "val_avg_iou_scores", "val_avg_dice_scores", val_iou_avg_values, val_dice_avg_values)

    mkdir_prep_dir(f"icpsr/models/{MODEL_NAME.lower()}/seg_metric_tables/")

    # Table for Average BCE Loss, IoU, Dice
    plot_unet3d_metrics_table(val_bce_loss_avg_values, val_iou_avg_values, val_dice_avg_values, f"icpsr/models/{MODEL_NAME.lower()}/seg_metric_tables/{MODEL_NAME.lower()}_val_avg_dice_iou_bceloss.jpg", plot_title = f"{MODEL_NAME} Valid Avg Loss, IoU & Dice Score Metrics", col_prefix = "Val", epoch_steps=step)



def train_skull_strip_segmentation(nifti_csv_data, dataset_name="icpsr"):
    X_train, X_val, y_train, y_val = train_test_split(nifti_csv_data["intensity_norm"].tolist(), nifti_csv_data["skull_mask_index"].tolist(), test_size=0.3)

    if DEBUG:
        print("X_train len = {}".format(len(X_train)))
        print("X_val len = {}".format(len(X_val)))

    print("Creating brain train & val datasets")
    
    brain_train_dataset = BrainMRIDataset(X_train, y_train)
    brain_val_dataset = BrainMRIDataset(X_val, y_val)

    if DEBUG:
        print("brain_train_dataset len = {}".format(len(brain_train_dataset)))
        print("brain_val_dataset len = {}".format(len(brain_val_dataset)))

    print("Creating brain train & val dataloaders")

    # Higher batch_size, we lose gpu memory
    train_loader = DataLoader(brain_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(brain_val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

    print("Creating Skull Strip Seg UNet3D model")

    # in_channels=1 for 1 medical image modality T2-weighted; out_channels=1 for skull vs non-skull
    unet3d_model = SimpleUNet3D(in_channels=1, out_channels=1).to(device=DEVICE)

    print("Compiling UNet3D with Adam, BCEWithLogitsLoss")

    # compile model with Adam optimizer, focal loss, metrics
    optimizer = optim.Adam(unet3d_model.parameters(), lr=LEARNING_RATE)
    # focal_criterion = FocalLoss(gamma=0.7).to(device)
    # Use pytorch's built-in focal loss
    bce_criterion = nn.BCEWithLogitsLoss().to(device=DEVICE)

    mkdir_prep_dir(f"{dataset_name}/models/{MODEL_NAME.lower()}/saved_weights")

    if not USE_PRETRAINED_MODEL:
        step = train_skull_strip_seg_over_epochs(unet3d_model, optimizer, bce_criterion, train_loader, val_loader, dataset_name, dst_folder=f"{dataset_name}/models/{MODEL_NAME.lower()}/saved_weights/")
    else:
        timestamp = time.time()
        step = f"pretrained_{timestamp}"

        model_filepath = "/home/bizon/src/AI_Stroke_Diagnosis/DataPipeline/src/backend/torch/skull-strip-seg/icpsr/models/basic_unet3d_best_sofar/saved_weights/unet3d_skull_strip_seg_1692.pth.tar"

    mkdir_prep_dir("{}/models/{}/saved_seg_slices/{}".format(dataset_name, MODEL_NAME.lower(), step))

    qual_eval_skull_strip_seg(unet3d_model, val_loader, dataset_name, model_filepath, dst_folder="{}/models/{}/saved_seg_slices/{}".format(dataset_name, MODEL_NAME.lower(), step))

    save_visualized_skull_strip_avg_metrics(unet3d_model, val_loader, dataset_name, step)


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

def main_with_nifi_zmq_sub():
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

def main():
    nifti_csv_df = pd.read_csv("/media/bizon/projects_1/data/ICPSR_38464_Stroke_Data_NiFi/data_prep_icpsr_seg/icpsr_skull_stroke_lesion_seg_prep.csv")
    train_skull_strip_segmentation(nifti_csv_df, dataset_name="icpsr")


def nfbs_skull_strip_seg_unet_training():
    nifti_csv_df = pd.read_csv("nfbs/prep/nfbs_skull_strip_seg_prep.csv")
    advanced_train_unet3d(nifti_csv_df, dataset_name="nfbs")

def icpsr_skull_strip_seg_unet_training():
    nifti_csv_df = pd.read_csv("icpsr/prep/icpsr_skull_strip_seg_prep.csv")
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
    # simple_skull_strip_seg_unet_training()
    # nfbs_skull_strip_seg_unet_training()
    # icpsr_skull_strip_seg_unet_training()
    main()
