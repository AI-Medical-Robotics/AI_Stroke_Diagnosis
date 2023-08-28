import os
import zmq
import time
import pandas as pd
import pickle5 as pickle
import threading

from tqdm import tqdm

import numpy as np

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import torch
import torch.nn as nn
import torch.optim as optim
from model import CNNtoLSTM

from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from utils import save_checkpoint, load_checkpoint, print_examples

# Reference perplexity.ai for preparing data for pytorch and training pytorch:
# https://www.perplexity.ai/search/b1046a4a-c702-481b-b719-ace5706e024e?s=u

class ImgCapPrepDataset(Dataset):
    def __init__(self, imgid_keys, imgid_cap_map, ext_features, tokenizer, vocab, caps_max_len, vocab_size):
        self.imgid_keys = imgid_keys
        self.imgid_cap_map = imgid_cap_map
        self.ext_features = ext_features
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.caps_max_len = caps_max_len
        self.vocab_size = vocab_size
        self.padding_value = 0
    
    def __len__(self):
        return len(self.imgid_keys)

    def __getitem__(self, idx):
        img_id = self.imgid_keys[idx]
        captions_sample_list = self.imgid_cap_map[img_id]

        print("captions_sample_list type = {}".format(type(captions_sample_list)))

        X1, X2, y = [], [], []
        for caption_words in captions_sample_list:
            print("caption_words type = {}".format(type(caption_words)))
            print("caption_words = {}".format(caption_words))
            # flattened_captions = [caption for caption in caption_words]
            # print("flattened_captions type = {}".format(type(flattened_captions)))

            caption_word_tokens = self.tokenizer(caption_words)
            print("caption_word_tokens type = {}".format(type(caption_word_tokens)))
            print("caption_word_tokens = {}".format(caption_word_tokens))

            seq = [self.vocab[token_str] for token_str in caption_word_tokens]
            print("seq type = {}".format(type(seq)))
            print("seq = {}".format(seq))
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                print("in_seq type = {}".format(type(in_seq)))
                print("in_seq = {}".format(in_seq))
                # print("in_seq[0] = {}".format(in_seq[0]))
                in_seq = torch.tensor([in_seq]) # Convert in_seq to tensor
                # in_seq = torch.unsqueeze(in_seq, 0) # Add dim to match pad_sequence

                # manually ensure all padded sequences have same length by slicing them to caps_max_len
                in_seq = [seq[:self.caps_max_len] + [self.padding_value] * (self.caps_max_len - len(seq)) for seq in in_seq]
                in_seq = pad_sequence(in_seq, batch_first=True, padding_value=0)[0]

                out_seq = torch.tensor(out_seq)
                out_seq = F.one_hot(out_seq, num_classes=self.vocab_size)

                X1.append(self.ext_features[img_id])
                X2.append(in_seq)
                y.append(out_seq)

        X1 = torch.stack(X1)
        X2 = torch.cat(X2)
        y = torch.stack(y)

        return X1, X2, y


def split_train_test(imgid_cap_map, split_ratio=0.90):
    image_ids = list(imgid_cap_map.keys())
    split = int(len(image_ids) * split_ratio)
    train = image_ids[:split]
    test = image_ids[split:]
    return train, test

def extract_captions(img_cap_csv_df):
    all_captions = []
    imgid_cap_map = {}

    for i in range(len(img_cap_csv_df)):
        with open(img_cap_csv_df.imgid_to_prep_captions.iloc[i], "rb") as file:
            imgid_to_prep_caps = pickle.load(file)

            # image_id = imgid_to_prep_caps[0]
            # prep_captions_str = imgid_to_prep_caps[1]

            image_id = list(imgid_to_prep_caps.keys())[0]
            prep_captions_str = list(imgid_to_prep_caps.values())[0]

            imgid_cap_map[image_id] = prep_captions_str

            print("check1: imgid_to_prep_caps len = {}".format(len(imgid_to_prep_caps)))
            print("check1: imgid_to_prep_caps type = {}".format(type(imgid_to_prep_caps)))
            # print("image_id = {}".format(image_id))
            # print("prep_captions_str = {}".format(prep_captions_str))

            all_captions.append(prep_captions_str)

    return all_captions, imgid_cap_map

def extract_features(img_cap_csv_df):
    imgid_extfeatures_map = {}

    for i in range(len(img_cap_csv_df)):
        with open(img_cap_csv_df.imgid_to_feature.iloc[i], "rb") as file:
            imgid_to_ext_features = pickle.load(file)

            # image_id = imgid_to_ext_features[0]
            # ext_features = imgid_to_ext_features[1]

            image_id = list(imgid_to_ext_features.keys())[0]
            ext_features = list(imgid_to_ext_features.values())[0]

            imgid_extfeatures_map[image_id] = ext_features

            print("check1: imgid_to_ext_features len = {}".format(len(imgid_to_ext_features)))
            print("check1: imgid_to_ext_features type = {}".format(type(imgid_to_ext_features)))
            print("image_id = {}".format(image_id))
            print("ext_features len = {}".format(len(ext_features)))

    return imgid_extfeatures_map


# TODO (JG): Update to use pytorch tokenizer
def tokenize_captions(all_captions):
    tokenizer = get_tokenizer("basic_english")
    flattened_captions = [caption for sublist in all_captions for caption in sublist]
    tokens = (tokenizer(caption) for caption in flattened_captions)
    vocab = build_vocab_from_iterator(tokens, specials=["<unk>"])
    vocab_size = len(vocab)
    return tokenizer, vocab, vocab_size

def get_captions_max(all_captions):
    return max(len(caption) for caption in all_captions)


# def eval_cnn_lstm(model, results_plot_path, eval_train=True):
#     model.evaluating(results_plot_path, train=eval_train)

def train_cnn_lstm(img_cap_csv_df, batch_size = 64, save_model = True):
    # NiFi data flow already loads the dataset, preprocesses the data and then we now subscribe to ZMQ to 
    # receive the pandas dataframe with preprocessed data filepaths
    print("Entering the training thread train_cnn_lstm")

    print("Passed ZMQ Subscriber's img_cap_csv_df to train_cnn_lstm for training soon:")
    print(img_cap_csv_df.head())

    all_captions, imgid_cap_map = extract_captions(img_cap_csv_df)
    print("all_captions = {}".format(all_captions))

    ext_features = extract_features(img_cap_csv_df)

    tokenizer, vocab, vocab_size = tokenize_captions(all_captions)

    captions_max_len = get_captions_max(all_captions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    # embed_size = 256
    # hidden_size = 256
    # num_layers = 1
    learning_rate = 3e-4
    num_epochs = 10

    writer = SummaryWriter("runs/flickr")

    train, test = split_train_test(imgid_cap_map, split_ratio=0.90)

    img_cap_prep_dataset = ImgCapPrepDataset(imgid_keys=train, imgid_cap_map=imgid_cap_map, ext_features=ext_features, tokenizer=tokenizer, vocab=vocab, caps_max_len=captions_max_len, vocab_size=vocab_size)

    imgcap_prep_loader = DataLoader(img_cap_prep_dataset, batch_size= batch_size, shuffle=True)


    img_cap_model = CNNtoLSTM(captions_max_len, vocab_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(img_cap_model.parameters(), lr=learning_rate)
    step = 0

    img_cap_model.train()

    for epoch in range(num_epochs):
        # following line is to see a couple test cases
        # print_examples(img_cap_model, device, img_cap_prep_dataset)
        
        if save_model:
            checkpoint = {
                "state_dict": img_cap_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in tqdm(
            enumerate(imgcap_prep_loader), total=len(imgcap_prep_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = img_cap_model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape[-1])

            writer.add_scalar("CNNtoLSTM Training Loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

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

def main():
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
