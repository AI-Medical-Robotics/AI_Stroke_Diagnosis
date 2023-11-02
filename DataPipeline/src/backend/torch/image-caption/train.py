import os
import zmq
import time
import pandas as pd
import pickle5 as pickle
import threading

from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from voxel_cap_model import CNN3DtoRNN

# from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from utils import save_checkpoint, load_checkpoint, print_examples

from mri_cap_dataset import StrokeMRIVocabulary, StrokeMRICapDataset, MRICapCollate

from nltk.translate.bleu_score import corpus_bleu

# Hyperparameters
LEARNING_RATE = 1e-1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
VAL_BATCH_SIZE = 1
LOAD_MODEL = False
NUM_EPOCHS = 2
DEBUG = False
TRAINING_TASKNAME = "Stroke Lesion MRI Captioning"

USE_PRETRAINED_MODEL=False

# Hyperparameters
embed_size = 256
hidden_size = 256
num_layers = 1
learning_rate = 3e-4

# Loss values for loss curve
train_ce_loss_values = []
val_ce_loss_values = []

train_bleu_score_values = []
val_bleu_score_values = []

def mkdir_prep_dir(dirpath):
    """make preprocess directory if doesn't exist"""
    prep_dir = dirpath
    if not os.path.exists(prep_dir):
        os.makedirs(prep_dir)
    return prep_dir

def extract_captions(nifti_csv_data):
    all_captions = []
    # imgid_cap_map = {}

    for i in range(len(nifti_csv_data)):
        with open(nifti_csv_data["voxid_to_prep_caption"].iloc[i], "rb") as file:
            imgid_to_prep_caps = pickle.load(file)

            voxel_id = imgid_to_prep_caps[0]
            prep_captions_str = imgid_to_prep_caps[1]
            # voxel_id = list(imgid_to_prep_caps.keys())[0]
            # prep_captions_str = list(imgid_to_prep_caps.values())[0]

            # imgid_cap_map[voxel_id] = prep_captions_str

            all_captions.append(prep_captions_str)
    
    return all_captions

# def extract_features(nifti_csv_data):
#     pass

def tokenize_captions(all_captions):
    tokenizer = get_tokenizer("basic_english")
    flattened_captions = [caption for sublist in all_captions for caption in sublist]
    tokens = (tokenizer(caption) for caption in flattened_captions)
    vocab = build_vocab_from_iterator(tokens, specials=["<unk>"])
    vocab_size = len(vocab)
    return tokenizer, vocab, vocab_size
    
def get_captions_max(all_captions):
    return max(len(caption) for caption in all_captions)


def calc_bleu_score(caption_pred, caption_gt):
    # gather tokens in torch tensors for caption pred and gt, 
    # store them into list, compute bleu score
    caption_gt_list = []
    caption_pred_list = []

    count = 0

    print(f"caption_gt.tolist() = {caption_gt.tolist()}")
    print(f"caption_pred.tolist() = {caption_pred.tolist()}")

    for gt_idx in caption_gt.tolist():
        caption_gt_list.append(gt_idx[0])

    for pred_idx in caption_pred.tolist():
        caption_pred_list.append(pred_idx[0])

    bleu_score = nltk.translate.bleu_score.corpus_bleu([caption_gt_list], [caption_pred_list])
    print(f"BLEU Score: {bleu_score}")
    return bleu_score

def train_cnn3d_rnn(train_loader, voxcap_dnn, optimizer, loss_criterion, step):

    train_loop = tqdm(train_loader)

    train_loss_values = []
    train_bleu_scores = []

    print("Running train_cnn3d_rnn")
    for batch_idx, (prep_voxel, voxel_captions) in enumerate(train_loop):
        prep_voxel = prep_voxel.to(device=DEVICE)
        voxel_captions = voxel_captions.to(device=DEVICE)
        # if debug:
        #     print("batch_idx = {}; prep_voxel len = {}; voxel_captions len = {}".format(batch_idx, len(prep_voxel), len(voxel_captions)))
            
        # print(f"prep_voxel.shape = {prep_voxel.shape}")
        # print(f"prep_voxel.unsqueeze(0).shape = {prep_voxel.unsqueeze(0).shape}")
        # print(f"prep_voxel.unsqueeze(0).unsqueeze(0).shape = {prep_voxel.unsqueeze(0).unsqueeze(0).shape}")

        optimizer.zero_grad()
        outputs = voxcap_dnn(prep_voxel, voxel_captions[:-1])

        # print(f"voxcap_dnn outputs.shape = {outputs.shape}")
        loss = loss_criterion(outputs.reshape(-1, outputs.shape[2]), voxel_captions.reshape(-1))

        train_bleu_score = calc_bleu_score(outputs.reshape(-1, outputs.shape[2]), voxel_captions.reshape(-1))

        loss.backward(loss)
        optimizer.step()

        train_loss_values.append(loss.item())
        train_bleu_scores.append(train_bleu_score)

        # update tqdm loop
        train_loop.set_postfix(train_ce_loss=loss.item())

        step += 1

    train_ce_loss_values.append(min(train_loss_values))
    train_bleu_score_values.append(min(train_bleu_scores))

    return step


def save_checkpoint(state, filename="best_cnn3d_rnn_imgcap.pth.tar"):
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

def save_mri_sliced_captions(loader, voxcap_dnn, dataset_name, stroke_vocab, folder="save_mri_sliced_captions"):
    # nifti_percent_slices_save = 0.025
    # nifti_filepath_df_row = 0
    nifti_seg_2d_slice_divisor = 2
    nifti_data_type = dataset_name
    nifti_csv_col_name = "stroke_mri_caption"
    caption_name = "Medical History"

    with torch.no_grad():
        for index, (prep_voxel, voxel_captions) in enumerate(loader):
            # print(f"voxel_captions = {voxel_captions}")

            # print("Setting userprep_voxel, voxel_captions on gpu device")
            prep_voxel = prep_voxel.to(device=DEVICE)
            voxel_captions = voxel_captions.to(device=DEVICE)

            prep_voxel_np = prep_voxel.squeeze().cpu().numpy()

            # outputs = voxcap_dnn(prep_voxel, voxel_captions[:-1])

            image_caption_pred_list, image_caption_gt_list = voxcap_dnn.caption_image(prep_voxel, stroke_vocab, gt_caption=voxel_captions)
            image_caption_pred_str = " ".join(image_caption_pred_list)
            image_caption_gt_str = " ".join(image_caption_gt_list)

            caption = f"MRI {index} Caption: Pred: {image_caption_pred_str}; GT: {image_caption_gt_str}"
            print(caption)

            # TODO (JG): Double check silent plot save, so it doesnt open up
            slice_id = prep_voxel_np.shape[0]//nifti_seg_2d_slice_divisor
            # fig, ax = plt.subplots(figsize=(14, 10))
            fig, ax = plt.subplots()
            ax.text(0.5, -0.1, caption, ha="center", va="center", transform=ax.transAxes)
            ax.set_title("{}: {} Stroke MRI Caption of 2D Slice ID {} ".format(caption_name, nifti_data_type.upper(), slice_id))
            ax.imshow(prep_voxel_np[slice_id])

            saved_mri_cap_slice_dir = mkdir_prep_dir(folder)
            output_filename = "{}_{}_slice_id_{}.{}".format(nifti_csv_col_name, index, slice_id, "jpg")
            output_filepath = os.path.join(saved_mri_cap_slice_dir, output_filename)
            plt.savefig(output_filepath)
            plt.close()
            print(f"Saved Stroke MRI Caption 2D Slice Viz: {saved_mri_cap_slice_dir}")

def validate_cnn3d_rnn(val_loader, voxcap_dnn, loss_criterion, stroke_vocab):
    val_loop = tqdm(val_loader)

    val_loss_values = []
    val_bleu_scores = []

    print("Running validate_cnn3d_rnn")
    with torch.no_grad():
        for batch_idx, (prep_voxel, voxel_captions) in enumerate(val_loop):

            # print("Setting userprep_voxel, voxel_captions on gpu device")
            prep_voxel = prep_voxel.to(device=DEVICE)
            voxel_captions = voxel_captions.to(device=DEVICE)

            outputs = voxcap_dnn(prep_voxel, voxel_captions[:-1])
            # print(f"voxcap_dnn outputs.shape = {outputs.shape}")

            loss = loss_criterion(outputs.reshape(-1, outputs.shape[2]), voxel_captions.reshape(-1))
            
            val_bleu_score = calc_bleu_score(outputs.reshape(-1, outputs.shape[2]), voxel_captions.reshape(-1))

            # Save the validation loss values to list
            val_loss_values.append(loss.item())

            val_bleu_scores.append(val_bleu_score)
            
            val_loop.set_postfix(val_ce_loss=loss.item())

            # print_examples(voxcap_dnn, prep_voxel, DEVICE, stroke_vocab, batch_idx)

    val_ce_loss_values.append(min(val_loss_values))
    val_bleu_score_values.append(min(val_bleu_scores))

def train_voxcap_over_epochs(voxcap_dnn, optimizer, loss_criterion, step, train_loader, val_loader, stroke_vocab, dataset_name="icpsr"):
    voxcap_dnn.train()

    for epoch in range(NUM_EPOCHS):
        # following line is to see a couple test cases

        step = train_cnn3d_rnn(train_loader, voxcap_dnn, optimizer, loss_criterion, step)

        checkpoint = {
            "state_dict": voxcap_dnn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        }
        mkdir_prep_dir("icpsr/models/")
        filename="icpsr/models/cnn3d_rnn_voxcap_{}.pth.tar".format(step)
        save_checkpoint(checkpoint, filename=filename)

        validate_cnn3d_rnn(val_loader, voxcap_dnn, loss_criterion, stroke_vocab)

def qual_eval_voxcap(voxcap_dnn, step, val_loader, stroke_vocab, dataset_name="icpsr"):
    voxcap_dnn.eval()

    # for epoch in range(NUM_EPOCHS):

    model_filepath = "/home/bizon/src/AI_Stroke_Diagnosis/DataPipeline/src/backend/torch/image-caption/icpsr/models/cnn3d_rnn_voxcap_2041.pth.tar"
    
    if USE_PRETRAINED_MODEL:
        load_checkpoint(torch.load(model_filepath), voxcap_dnn)

    save_mri_sliced_captions(val_loader, voxcap_dnn, dataset_name, stroke_vocab, folder="{}/save_mri_sliced_captions/".format(dataset_name))
    # NOTE (JG): If we used epochs here, step inc by 100
    step += 100

def plot_cnn3d_rnn_loss_curve(ce_loss_values, plot_title, plot_filename):
    plt.title(plot_title)
    plt.plot(ce_loss_values, label="CE Loss")
    plt.legend()
    plt.savefig(plot_filename)
    plt.show()

def plot_cnn3d_rnn_bleu_score_curve(bleu_score_values, plot_title, plot_filename):
    plt.title(plot_title)
    plt.plot(bleu_score_values, label="BLEU Score")
    plt.legend()
    plt.savefig(plot_filename)
    plt.show()



def train_stroke_mri_captioning(nifti_csv_data, dataset_name="icpsr"):
    X_train, X_val, y_train, y_val = train_test_split(nifti_csv_data["voxid_to_prep_vox"].tolist(), nifti_csv_data["voxid_to_prep_caption"].tolist(), test_size=0.3)

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

    mricap_train_dataset = StrokeMRICapDataset(X_train, y_train)
    mricap_val_dataset = StrokeMRICapDataset(X_val, y_val)

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

    step=0

    # initialize model, loss, etc; captions_max_len = embed_size
    model = CNN3DtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(DEVICE)
    # model = CNN3DtoRNN(captions_max_len, hidden_size, vocab_size, num_layers).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    torch.backends.cudnn.benchmark = True
    load_model = False

    # if load_model:
    #     step = load_checkpoint(torch.load("cnn3d_rnn_voxcap_{}.pth.tar"), model, optimizer)

    train_voxcap_over_epochs(model, optimizer, criterion, step, train_loader, val_loader, stroke_vocab)

    # qual_eval_voxcap(model, step, val_loader, stroke_vocab)

    mkdir_prep_dir("icpsr/models/loss_curves/")
    plot_cnn3d_rnn_loss_curve(train_ce_loss_values, "CNN3DToLSTM Train Loss Curve", "icpsr/models/loss_curves/cnn3d_to_lstm_train_loss_curve.jpg")
    plot_cnn3d_rnn_loss_curve(val_ce_loss_values,  "CNN3DToLSTM Valid Loss Curve", "icpsr/models/loss_curves/cnn3d_to_lstm_val_loss_curve.jpg")

    mkdir_prep_dir("icpsr/models/bleu_score_curves/")
    plot_cnn3d_rnn_bleu_score_curve(train_bleu_score_values, "CNN3DToLSTM Train BLEU Score Curve", "icpsr/models/bleu_score_curves/cnn3d_to_lstm_train_bleu_curve.jpg")
    plot_cnn3d_rnn_bleu_score_curve(train_bleu_score_values, "CNN3DToLSTM Valid BLEU Score Curve", "icpsr/models/bleu_score_curves/cnn3d_to_lstm_val_bleu_curve.jpg")


def main():
    nifti_csv_df = pd.read_csv("/media/bizon/projects_1/data/ICPSR_38464_Stroke_Data_NiFi/join_pd_dataframes/data_prep_icpsr_cap/voxelid_caption_prep.csv")
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
    main()
