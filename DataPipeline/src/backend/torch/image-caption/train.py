import os
import zmq
import time
import pandas as pd
import pickle5 as pickle
import threading

from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import torch
# import torchmetrics
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from voxel_cap_model import CNN3DtoLSTM

# from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from utils import save_checkpoint, load_checkpoint, print_examples

from mri_cap_dataset import StrokeMRIVocabulary, StrokeMRICapDataset, MRICapCollate

import nltk.translate.bleu_score

# Hyperparameters
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
VAL_BATCH_SIZE = 1
LOAD_MODEL = False
NUM_EPOCHS = 6
DEBUG = False

# Any metrics that require Max, Min will require being trained
USE_PRETRAINED_MODEL=False

CAPTION_TYPE = "short_caption"
# CAPTION_TYPE = "long_caption"

TRAINING_TASKNAME = f"Stroke Lesion MRI {CAPTION_TYPE.upper()}"

MODEL_NAME = "CNN3DToLSTM"

# caption_name = "Medical History"

caption_name = "Lesion Type"

# Hyperparameters
embed_size = 256
hidden_size = 256
num_layers = 1

# Avg Loss values Per Epoch for loss curve
train_ce_loss_avg_values = []
val_ce_loss_avg_values = []

# Min Loss values Per Epoch for loss curve
train_ce_loss_min_values = []
val_ce_loss_min_values = []

# Avg BLEU scores Per Epoch
# train_bleu_score_avg_values = []
val_bleu_score_avg_values = []

# Max BLEU scores Per Epoch
# train_bleu_score_max_values = []
val_bleu_score_max_values = []

def mkdir_prep_dir(dirpath):
    """make preprocess directory if doesn't exist"""
    prep_dir = dirpath
    if not os.path.exists(prep_dir):
        os.makedirs(prep_dir)
    return prep_dir

def extract_captions(nifti_csv_data):
    all_captions = []
    # imgid_cap_map = {}

    if caption_name == "Medical History":
        target_colname = "voxid_to_prep_caption"
    elif caption_name == "Lesion Type":
        target_colname = "voxid_to_clinical_lesions"


    for i in range(len(nifti_csv_data)):
        with open(nifti_csv_data[target_colname].iloc[i], "rb") as file:
            voxid_to_prep_caps = pickle.load(file)

            # voxel_id = voxid_to_prep_caps[0]
            # prep_clinical_captions_list = voxid_to_prep_caps[1]

            voxel_id = list(voxid_to_prep_caps.keys())[0]
            prep_clinical_captions_list = list(voxid_to_prep_caps.values())[0]

            if CAPTION_TYPE == "short_caption":
                clinical_label = prep_clinical_captions_list[0]
                # print(f"short caption: clinical_label = {clinical_label}")
                prep_captions_str = clinical_label
            elif CAPTION_TYPE == "long_caption":
                caption = prep_clinical_captions_list[1]
                # print(f"long caption: caption = {caption}")
                prep_captions_str = caption

            all_captions.append(prep_captions_str)
    
    # TODO (JG): Plot distribution between ischemic and hemorrhagic

    return all_captions

def tokenize_captions(all_captions):
    tokenizer = get_tokenizer("basic_english")
    flattened_captions = [caption for sublist in all_captions for caption in sublist]
    tokens = (tokenizer(caption) for caption in flattened_captions)
    vocab = build_vocab_from_iterator(tokens, specials=["<unk>"])
    vocab_size = len(vocab)
    return tokenizer, vocab, vocab_size
    
def get_captions_max(all_captions):
    return max(len(caption) for caption in all_captions)


def calc_bleu_score(voxcap_dnn, stroke_vocab, prep_voxel, voxel_captions, i, batch_size):
    # gather tokens in torch tensors for caption pred and gt, 
    # store them into list, compute bleu score

    image_caption_pred_list, image_caption_gt_list = voxcap_dnn.caption_image(prep_voxel, stroke_vocab, gt_caption=voxel_captions, keep_eos=True)

    bleu_smooth = nltk.translate.bleu_score.SmoothingFunction().method1
    bleu_score = nltk.translate.bleu_score.corpus_bleu([image_caption_gt_list], [image_caption_pred_list], smoothing_function=bleu_smooth)

    return bleu_score



def save_checkpoint(state, filename="best_cnn3d_lstm_imgcap.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def print_examples(voxcap_dnn, voxel_extfets, DEVICE, stroke_vocab, index):
    # TODO (JG): Grount Truth Captions
    print(f"Ex {index} MRI Caption Pred: " + " ".join(
        voxcap_dnn.caption_image(voxel_extfets, stroke_vocab)
    ))

# NOTE (JG): Couldnt compute BLEU score during training, but worked fine it seems during validation.
# In this function, we use val_loader
def save_mri_sliced_captions(val_loader, voxcap_dnn, dataset_name, stroke_vocab, folder="save_mri_sliced_captions"):
    val_loop = tqdm(val_loader)
    # nifti_percent_slices_save = 0.025
    # nifti_filepath_df_row = 0
    nifti_seg_2d_slice_divisor = 2
    nifti_data_type = dataset_name
    nifti_csv_col_name = "stroke_mri_caption"

    print(f"Running save_mri_sliced_captions")
    with torch.no_grad():
        for index, (prep_voxel, voxel_captions) in enumerate(val_loop):
            # print(f"voxel_captions = {voxel_captions}")

            # print("Setting userprep_voxel, voxel_captions on gpu device")
            prep_voxel = prep_voxel.to(device=DEVICE)
            voxel_captions = voxel_captions.to(device=DEVICE)

            prep_voxel_np = prep_voxel.squeeze().cpu().numpy()

            # outputs = voxcap_dnn(prep_voxel, voxel_captions[:-1])

            image_caption_pred_list, image_caption_gt_list = voxcap_dnn.caption_image(prep_voxel, stroke_vocab, gt_caption=voxel_captions, keep_eos=False)
            image_caption_pred_str = " ".join(image_caption_pred_list)
            image_caption_gt_str = " ".join(image_caption_gt_list)

            caption = f"Pred: {image_caption_pred_str}\n GT: {image_caption_gt_str}"
            # print(f"MRI {index} Caption: {caption}")

            # TODO (JG): Double check silent plot save, so it doesnt open up
            slice_id = prep_voxel_np.shape[0]//nifti_seg_2d_slice_divisor
            fig, ax = plt.subplots(figsize=(8, 8))
            # fig, ax = plt.subplots()
            ax.text(0.5, -0.1, caption, wrap = True, ha="center", va="center", transform=ax.transAxes, fontsize=12, fontweight="bold")
            ax.set_title(f"{caption_name}: {nifti_data_type.upper()} Stroke MRI {index} Caption of 2D Slice ID {slice_id}")
            ax.imshow(prep_voxel_np[slice_id])



            saved_mri_cap_slice_dir = mkdir_prep_dir(folder)
            output_filename = "{}_{}_slice_id_{}.{}".format(nifti_csv_col_name, index, slice_id, "jpg")
            output_filepath = os.path.join(saved_mri_cap_slice_dir, output_filename)
            plt.savefig(output_filepath)
            plt.close()
            # print(f"Saved Stroke MRI Caption 2D Slice Viz: {saved_mri_cap_slice_dir}")
            # val_loop.set_postfix(save_mri_sliced_captions=index)

# add stroke_vocab
# NOTE (JG): Couldnt compute BLEU score during training, but worked fine it seems during validation
def train_cnn3d_lstm(train_loader, voxcap_dnn, optimizer, ce_criterion, step, stroke_vocab, epoch):

    train_loop = tqdm(train_loader)

    train_ce_loss_values = []
    train_bleu_scores = []

    print(f"Running train_cnn3d_lstm epoch {epoch}")
    for batch_idx, (prep_voxel, voxel_captions) in enumerate(train_loop):
        prep_voxel = prep_voxel.to(device=DEVICE)
        voxel_captions = voxel_captions.to(device=DEVICE)
        # if debug:
        #     print("batch_idx = {}; prep_voxel len = {}; voxel_captions len = {}".format(batch_idx, len(prep_voxel), len(voxel_captions)))
            
        optimizer.zero_grad()
        outputs = voxcap_dnn(prep_voxel, voxel_captions[:-1])

        # print(f"voxcap_dnn outputs.shape = {outputs.shape}")
        loss = ce_criterion(outputs.reshape(-1, outputs.shape[2]), voxel_captions.reshape(-1))

        # train_bleu_score = calc_bleu_score(voxcap_dnn, stroke_vocab, prep_voxel, voxel_captions, batch_idx, train_loader.batch_size)

        loss.backward(loss)
        optimizer.step()

        train_ce_loss_values.append(loss.item())
        # train_bleu_scores.append(train_bleu_score)

        # update tqdm loop
        train_loop.set_postfix(train_ce_loss=loss.item())

        step += 1

    if NUM_EPOCHS > 1:
        train_ce_loss_avg_values.append( sum(train_ce_loss_values)/len(train_loader) )
        # train_bleu_score_avg_values.append( sum(train_bleu_scores)/len(train_loader) )

        train_ce_loss_min_values.append( min(train_ce_loss_values) )
        # train_bleu_score_max_values.append( max(train_bleu_scores) )
    else:
        train_ce_loss_avg_values.extend(train_ce_loss_values)
        # train_bleu_score_avg_values.extend(train_bleu_scores)

    return step



def validate_cnn3d_lstm(val_loader, voxcap_dnn, ce_criterion, stroke_vocab, epoch):
    val_loop = tqdm(val_loader)

    val_ce_loss_values = []
    val_bleu_scores = []

    # bleu_score = torchmetrics.text.BLEUScore()

    print(f"Running validate_cnn3d_lstm {epoch}")
    with torch.no_grad():
        for batch_idx, (prep_voxel, voxel_captions) in enumerate(val_loop):

            # print(f"voxel_captions = {voxel_captions}")
            # print("Setting userprep_voxel, voxel_captions on gpu device")
            prep_voxel = prep_voxel.to(device=DEVICE)
            voxel_captions = voxel_captions.to(device=DEVICE)

            outputs = voxcap_dnn(prep_voxel, voxel_captions[:-1])
            # print(f"voxcap_dnn outputs.shape = {outputs.shape}")

            loss = ce_criterion(outputs.reshape(-1, outputs.shape[2]), voxel_captions.reshape(-1))
            
            bleu_score = calc_bleu_score(voxcap_dnn, stroke_vocab, prep_voxel, voxel_captions, batch_idx, val_loader.batch_size)

            # Save the validation loss values to list
            val_ce_loss_values.append(loss.item())

            val_bleu_scores.append(bleu_score)
            
            val_loop.set_postfix(val_ce_loss=loss.item(), val_bleu_score=bleu_score)

            # print_examples(voxcap_dnn, prep_voxel, DEVICE, stroke_vocab, batch_idx)


    if NUM_EPOCHS > 1:
        val_ce_loss_avg_values.append( sum(val_ce_loss_values)/len(val_loader) )
        val_bleu_score_avg_values.append( sum(val_bleu_scores)/len(val_loader) )

        val_ce_loss_min_values.append( min(val_ce_loss_values) )
        val_bleu_score_max_values.append( max(val_bleu_scores) )
    else:
        val_ce_loss_avg_values.extend(val_ce_loss_values)
        val_bleu_score_avg_values.extend(val_bleu_scores)


def train_voxcap_over_epochs(voxcap_dnn, optimizer, ce_criterion, train_loader, val_loader, stroke_vocab, dataset_name, dst_folder):
    print("Training {} for {} on {} across {} epochs".format(MODEL_NAME, TRAINING_TASKNAME, dataset_name, NUM_EPOCHS))
   
    step = 0
    
    voxcap_dnn.train()

    for epoch in range(1, NUM_EPOCHS+1):
        # following line is to see a couple test cases

        step = train_cnn3d_lstm(train_loader, voxcap_dnn, optimizer, ce_criterion, step, stroke_vocab, epoch)

        checkpoint = {
            "state_dict": voxcap_dnn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        }
        mkdir_prep_dir(dst_folder)
        save_model_filepath=f"{dst_folder}/cnn3d_lstm_voxcap_{step}.pth.tar"
        save_checkpoint(checkpoint, filename=save_model_filepath)

        validate_cnn3d_lstm(val_loader, voxcap_dnn, ce_criterion, stroke_vocab, epoch)

    return step

def qual_eval_voxcap(voxcap_dnn, step, val_loader, stroke_vocab, dataset_name, model_filepath, dst_folder):
    voxcap_dnn.eval()

    # for epoch in range(NUM_EPOCHS):
    
    if USE_PRETRAINED_MODEL:
        load_checkpoint(torch.load(model_filepath), voxcap_dnn)
    elif model_filepath is None:
        print("model_filepath not provided, using trained model in qualitative evaluation")


    save_mri_sliced_captions(val_loader, voxcap_dnn, dataset_name, stroke_vocab, folder=dst_folder)
    # NOTE (JG): If we used epochs here, step inc by 100
    # step += 100

def plot_cnn3d_lstm_loss_curve(ce_loss_values, plot_title, plot_filename):
    f, ax = plt.subplots()

    ax.set_title(plot_title)
    ax.plot(ce_loss_values, color="blue", label="CE Loss")
    ax.grid(True)
    if NUM_EPOCHS > 1:
        labelx_text = "Epochs"
    else:
        labelx_text = "Steps Across Epochs"
    ax.set_ylabel("CE Loss")
    ax.legend()
    plt.savefig(plot_filename)
    plt.close()

def plot_cnn3d_lstm_2_loss_curves(train_ce_loss_values, val_ce_loss_values, plot_title, plot_filename):
    f, ax = plt.subplots()
    ax.set_title(plot_title)
    ax.plot(train_ce_loss_values, color="blue", label="Train CE Loss")
    ax.plot(val_ce_loss_values, color="gold", label="Valid CE Loss")
    ax.grid(True)
    if NUM_EPOCHS > 1:
        labelx_text = "Epochs"
    else:
        labelx_text = "Steps Across Epochs"
    ax.set_ylabel("CE Loss")
    ax.legend()
    plt.savefig(plot_filename)
    plt.close()

def save_loss_values_to_pd_csv(filepath, column_name1, column_name2, train_ce_loss_values, val_ce_loss_values):
    loss_values_df = pd.DataFrame(columns = [column_name1, column_name2])

    val_ce_loss_values_cp = []
    val_ce_loss_values_cp.extend(val_ce_loss_values)

    # NOTE: there are more train losses, so setting remaining val losses to 0 for pd df
    val_ce_loss_values_cp += ["NA"] * (len(train_ce_loss_values) - len(val_ce_loss_values))

    loss_values_df[column_name1] = train_ce_loss_values
    loss_values_df[column_name2] = val_ce_loss_values_cp

    loss_values_df.to_csv(filepath, index=False)


def plot_cnn3d_lstm_metric_score_curve(seg_metric_name, metric_score_values, plot_title, plot_filename):
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


def save_metric_scores_to_pd_csv(filepath, column_name, metric_score_values):
    new_metric_score_df = pd.DataFrame(columns = [column_name])
    new_metric_score_df[column_name] = metric_score_values
    new_metric_score_df.to_csv(filepath, index=False)



def plot_cnn3d_lstm_metrics_table(val_ce_loss_values, val_bleu_score_values, plot_filename, plot_title, col_prefix, epoch_steps=None):
    # highest_dice_score = val_dice_score_values.index(max(val_dice_score_values))
    highest_bleu_score = val_bleu_score_values.index(max(val_bleu_score_values))
    lowest_loss_score = val_ce_loss_values.index(min(val_ce_loss_values))

    if NUM_EPOCHS > 1:
        # Number of Epochs, mainly use this one
        epochs = [f"Epoch {idx}" for idx in range(NUM_EPOCHS)]
        epoch_colname = "Epoch"
    elif epoch_steps is not None:
        # Total Steps across all Epochs, it is total iterations after train_loader finishes; only testing for 1 epoch
        epochs = [f"Epoch Step {idx}" for idx in range(epoch_steps)] 
        epoch_colname = "Epoch (Step)"




    if len(col_prefix) == 1:
        df = pd.DataFrame({
        epoch_colname: [epochs[highest_bleu_score], epochs[lowest_loss_score]],
        f"{col_prefix[0]} BLEU Score": [f"{max(val_bleu_score_values):.4f}", None],
        f"{col_prefix[0]} CE Loss": [None, f"{min(val_ce_loss_values):.4f}"]
        })

        df[f"{MODEL_NAME}"] = df.columns[1:]


        # idx 0: Avg
        df = df[[f"{MODEL_NAME}", epoch_colname, f"{col_prefix[0]} BLEU Score", f"{col_prefix[0]} CE Loss"]]
    elif len(col_prefix) == 2:
        df = pd.DataFrame({
        epoch_colname: [epochs[highest_bleu_score], epochs[lowest_loss_score]],
        f"{col_prefix[1]} BLEU Score": [f"{max(val_bleu_score_values):.4f}", None],
        f"{col_prefix[0]} CE Loss": [None, f"{min(val_ce_loss_values):.4f}"]
        })

        df[f"{MODEL_NAME}"] = df.columns[1:]


        # idx 0: Min; idx 1: Max
        df = df[[f"{MODEL_NAME}", epoch_colname, f"{col_prefix[1]} BLEU Score", f"{col_prefix[0]} CE Loss"]]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axis("off")

    colors = [['white', 'lightgrey', 'white', 'lightgrey'],
              ['lightgrey', 'white', 'lightgrey', 'white']]

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



def plot_cnn3d_lstm_bleu_score_curve(bleu_score_values, plot_title, plot_filename):
    f, ax = plt.subplots()
    ax.set_title(plot_title)
    ax.plot(bleu_score_values, color="blue", label="BLEU Score")
    ax.legend()
    plt.savefig(plot_filename)
    plt.close()

def save_bleu_scores_to_pd_csv(filepath, column_name, bleu_score_values):
    new_bleu_score_df = pd.DataFrame(columns = [column_name])
    new_bleu_score_df[column_name] = bleu_score_values
    new_bleu_score_df.to_csv(filepath, index=False)


# CAPTION_TYPE
def save_visualized_stroke_captioning_avg_metrics(val_loader, dataset_name, step):

    mkdir_prep_dir(f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/loss_curves/")
    plot_cnn3d_lstm_loss_curve(train_ce_loss_avg_values, f"{MODEL_NAME} Train Avg Loss Curve", f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/loss_curves/{MODEL_NAME.lower()}_train_avg_loss_curve.jpg")
    plot_cnn3d_lstm_loss_curve(val_ce_loss_avg_values,  f"{MODEL_NAME} Valid Avg Loss Curve", f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/loss_curves/{MODEL_NAME.lower()}_val_avg_loss_curve.jpg")
    plot_cnn3d_lstm_2_loss_curves(train_ce_loss_avg_values, val_ce_loss_avg_values, f"{MODEL_NAME} Avg Loss Curves", f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/loss_curves/{MODEL_NAME.lower()}_train_val_avg_loss_curves.jpg")
    
    save_loss_values_to_pd_csv(f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/loss_curves/{MODEL_NAME.lower()}_train_val_avg_loss_values.csv", "train_ce_loss", "val_ce_loss", train_ce_loss_avg_values, val_ce_loss_avg_values)

    # TODO (JG): Figure out BLEU Score later
    mkdir_prep_dir(f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/bleu_score_curves/")

    plot_cnn3d_lstm_metric_score_curve("Valid BLEU", val_bleu_score_avg_values, f"{MODEL_NAME} Valid Avg BLEU Score Curve", f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/bleu_score_curves/{MODEL_NAME.lower()}_val_avg_bleu_curve.jpg")
    

    save_metric_scores_to_pd_csv(f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/bleu_score_curves/{MODEL_NAME.lower()}_val_avg_bleu_scores.csv", "val_avg_bleu_score", val_bleu_score_avg_values)

    mkdir_prep_dir(f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/seg_metric_tables/")

    # Table for Average BCE Loss, IoU, Dice
    plot_cnn3d_lstm_metrics_table(val_ce_loss_avg_values, val_bleu_score_avg_values, f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/seg_metric_tables/{MODEL_NAME.lower()}_val_avg_bleu_celoss.jpg", plot_title = f"{MODEL_NAME} Valid Avg Loss, BLEU Score Metrics", col_prefix = ["Avg"], epoch_steps=step)


def save_visualized_stroke_captioning_best_metrics(val_loader, dataset_name, step):
    plot_cnn3d_lstm_loss_curve(train_ce_loss_min_values, f"{MODEL_NAME} Train Min Loss Curve", f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/loss_curves/{MODEL_NAME.lower()}_train_min_loss_curve.jpg")
    plot_cnn3d_lstm_loss_curve(val_ce_loss_min_values,  f"{MODEL_NAME} Valid Min Loss Curve", f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/loss_curves/{MODEL_NAME.lower()}_val_min_loss_curve.jpg")
    plot_cnn3d_lstm_2_loss_curves(train_ce_loss_min_values, val_ce_loss_min_values, f"{MODEL_NAME} Min Loss Curves", f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/loss_curves/{MODEL_NAME.lower()}_train_val_min_loss_curves.jpg")
    
    save_loss_values_to_pd_csv(f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/loss_curves/{MODEL_NAME.lower()}_train_val_min_loss_values.csv", "train_ce_loss", "val_ce_loss", train_ce_loss_min_values, val_ce_loss_min_values)

    plot_cnn3d_lstm_metric_score_curve("Valid BLEU", val_bleu_score_max_values, f"{MODEL_NAME} Valid Max BLEU Score Curve", f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/bleu_score_curves/{MODEL_NAME.lower()}_val_max_bleu_curve.jpg")
    
    save_metric_scores_to_pd_csv(f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/bleu_score_curves/{MODEL_NAME.lower()}_val_max_bleu_scores.csv", "val_max_bleu_score", val_bleu_score_max_values)

    mkdir_prep_dir(f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/seg_metric_tables/")

    # Table for Average BCE Loss, IoU, Dice
    plot_cnn3d_lstm_metrics_table(val_ce_loss_min_values, val_bleu_score_max_values, f"icpsr/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/seg_metric_tables/{MODEL_NAME.lower()}_val_max_bleu_min_celoss.jpg", plot_title = f"{MODEL_NAME} Valid Min Loss, Max BLEU Score Metrics", col_prefix = ["Min", "Max"], epoch_steps=step)



def train_stroke_mri_captioning(nifti_csv_data, dataset_name="icpsr"):
    if caption_name == "Medical History":
        X_train, X_val, y_train, y_val = train_test_split(nifti_csv_data["voxid_to_prep_vox"].tolist(), nifti_csv_data["voxid_to_prep_caption"].tolist(), test_size=0.4)
    elif caption_name == "Lesion Type":
        X_train, X_val, y_train, y_val = train_test_split(nifti_csv_data["voxid_to_prep_vox"].tolist(), nifti_csv_data["voxid_to_clinical_lesions"].tolist(), test_size=0.4)

    if DEBUG:
        print("X_train len = {}".format(len(X_train)))
        print("X_val len = {}".format(len(X_val)))

    print("Creating stroke train & val datasets")
    
    all_captions = extract_captions(nifti_csv_data)
    # print("all_captions = {}".format(all_captions))

    stroke_vocab = StrokeMRIVocabulary(freq_threshold=5)
    stroke_vocab.build_vocabulary(all_captions)

    # tokenizer, vocab, vocab_size = tokenize_captions(all_captions)

    # captions_max_len = get_captions_max(all_captions)
    # print(f"captions_max_len = {captions_max_len}")

    mricap_train_dataset = StrokeMRICapDataset(X_train, y_train, caption_type = CAPTION_TYPE)
    mricap_val_dataset = StrokeMRICapDataset(X_val, y_val, caption_type = CAPTION_TYPE)

    vocab_size = len(stroke_vocab)
    # print(f"vocab_size = {vocab_size}")
    mricap_pad_idx = stroke_vocab.stoi["<PAD>"]

    if DEBUG:
        print("mricap_train_dataset len = {}".format(len(mricap_train_dataset)))
        print("mricap_val_dataset len = {}".format(len(mricap_val_dataset)))

    print("Creating brain train & val dataloaders")

    # Higher batch_size, we lose gpu memory
    train_loader = DataLoader(mricap_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=MRICapCollate(mri_pad_idx = mricap_pad_idx))
    val_loader = DataLoader(mricap_val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=MRICapCollate(mri_pad_idx = mricap_pad_idx))
    # train_loader = DataLoader(mricap_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(mricap_val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

    # step=0

    # initialize model, loss, etc; captions_max_len = embed_size
    cnn3dtolstm_model = CNN3DtoLSTM(embed_size, hidden_size, vocab_size, num_layers).to(DEVICE)
    # model = CNN3DtoRNN(captions_max_len, hidden_size, vocab_size, num_layers).to(DEVICE)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=mricap_train_dataset.mri_vocab.stoi["<PAD>"])
    # criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(cnn3dtolstm_model.parameters(), lr=LEARNING_RATE)

    torch.backends.cudnn.benchmark = True
    load_model = False

    # if load_model:
    #     step = load_checkpoint(torch.load("cnn3d_lstm_voxcap_{}.pth.tar"), model, optimizer)


    mkdir_prep_dir(f"{dataset_name}/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/saved_weights")

    if not USE_PRETRAINED_MODEL:
        step = train_voxcap_over_epochs(cnn3dtolstm_model, optimizer, ce_criterion, train_loader, val_loader, stroke_vocab, dataset_name, dst_folder=f"{dataset_name}/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/saved_weights/")
        model_filepath = None
    else:
        timestamp = time.time()
        step = f"pretrained_{timestamp}"

        model_filepath = "/home/bizon/src/AI_Stroke_Diagnosis/DataPipeline/src/backend/torch/image-caption/icpsr/models/cnn3dtolstm_bk110523_v4/long_caption/saved_weights/cnn3d_lstm_voxcap_1351.pth.tar"


    qual_eval_voxcap(cnn3dtolstm_model, step, val_loader, stroke_vocab, dataset_name, model_filepath, dst_folder=f"{dataset_name}/models/{MODEL_NAME.lower()}/{CAPTION_TYPE.lower()}/saved_seg_slices/{step}")


    save_visualized_stroke_captioning_avg_metrics(val_loader, dataset_name, step)

    save_visualized_stroke_captioning_best_metrics(val_loader, dataset_name, step)


def main_lesion_cap():
    nifti_csv_df = pd.read_csv("/media/bizon/projects_1/data/ICPSR_38464_Stroke_Data_NiFi/join_pd_dataframes/data_prep_icpsr_lesion_cap/voxelid_lesion_cap_prep.csv")
    train_stroke_mri_captioning(nifti_csv_df, dataset_name="icpsr")


def main_med_hist_cap():
    nifti_csv_df = pd.read_csv("/media/bizon/projects_1/data/ICPSR_38464_Stroke_Data_NiFi/join_pd_dataframes/data_prep_icpsr_clinical_cap/voxelid_clinical_caption_prep.csv")
    train_stroke_mri_captioning(nifti_csv_df, dataset_name="icpsr")

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

                self.data_to_process = received_dataframe

                # Sending confirmation happens late, I realized when the data is passed from
                # this thread to the part of the program that will get the dataframe to load 
                # images for training, this part send confirm part is too slow. Maybe bring
                # training here?
                print("Going to send confirmation to NiFi PutZMQ ImgCapPrep Publisher")
                self.socket.send_string("Received")
            except Exception as e:
                print("Error receiving DataFrame: {}".format(e))

    def __del__(self):
        self.stop()
        self.socket.close()

def main_with_nifi_zmq_sub():
    zmq_subscriber = ZMQSubscriber()

    img_cap_csv_df = None

    try:
        print("Subscribing to {} to receive ImgCap Prep DataFrame...".format(zmq_subscriber.zmq_socket_address))
        zmq_subscriber.start()

        while True:
            # check if there is data to process
            if zmq_subscriber.data_to_process is not None:
                img_cap_csv_df = zmq_subscriber.data_to_process

                # pass the data to the training thread
                train_thread = threading.Thread(target=train_cnn_lstm, args=(img_cap_csv_df,))
                train_thread.start()

                # reset the data_to_process to None to indicate processing is done
                zmq_subscriber.data_to_process = None
                print("Received data filepaths in pd df to train, finished training, exiting")
                # break

            time.sleep(0.2) # sleep for 200 milliseconds

    except KeyboardInterrupt:
        print("Stopping the subscriber...")
        zmq_subscriber.stop()
        # zmq_subscriber.running = False

if __name__ == "__main__":
    main_lesion_cap()
    # main_med_hist_cap()
