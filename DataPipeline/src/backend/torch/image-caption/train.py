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

import torch.optim as optim
from model import CNNtoLSTM

from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from utils import save_checkpoint, load_checkpoint, print_examples

def ImgCapPrepDataset(Dataset):
    def __init__(self, imgid_keys, imgid_cap_map, ext_features, tokenizer, caps_max_len, vocab_size):
        self.imgid_keys = imgid_keys
        self.imgid_cap_map = imgid_cap_map
        self.ext_features = ext_features
        self.tokenizer = tokenizer
        self.caps_max_len = caps_max_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return len(self.imgid_keys)

    def __getitem__(self, idx):
        img_id = self.imgid_keys[idx]
        captions = self.imgid_cap_map[img_id]

        X1, X2, y = [], [], []
        for caption in captions:
            seq = self.tokenizer(caption)
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequence([torch.tensor(in_seq)], batch_first=True, padding_value=0)
                out_seq = torch.tensor(out_seq)

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

            image_id = imgid_to_prep_caps[0]
            prep_captions_str = imgid_to_prep_caps[1]

            imgid_cap_map[image_id] = prep_captions_str

            self.logger.info("check1: imgid_to_prep_caps len = {}".format(len(imgid_to_prep_caps)))
            self.logger.info("image_id = {}".format(image_id))
            self.logger.info("prep_captions_str = {}".format(prep_captions_str))

            all_captions.append(prep_captions_str)

    return all_captions, imgid_cap_map

# TODO (JG): Update to use pytorch tokenizer
def tokenize_captions(all_captions):
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(all_captions)
    vocab = build_vocab_from_iterator([tokens], specials=["<unk>"])
    vocab_size = len(vocab)
    # print("vocab_size = {}".format(vocab_size))
    return tokenizer, vocab_size

def get_captions_max(self, all_captions):
    return max(len(caption.split()) for caption in all_captions)


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

    tokenizer, vocab_size = tokenize_captions(all_captions)

    captions_max_len = get_captions_max(all_captions)

    # Hyperparameters
    # embed_size = 256
    # hidden_size = 256
    # num_layers = 1
    learning_rate = 3e-4
    num_epochs = 10

    writer = SummaryWriter("runs/flickr")

    train, test = split_train_test(imgid_cap_map, split_ratio=0.90)

    img_cap_prep_dataset = ImgCapPrepDataset(train, imgid_cap_map, ext_features, tokenizer, captions_max_len, vocab_size)

    imgcap_prep_loader = DataLoader(img_cap_prep_dataset, batch_size= batch_size, shuffle=True)


    img_cap_model = CNNtoLSTM(captions_max_len, vocab_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(img_cap_model.parameters(), lr=learning_rate)
    step = 0

    img_cap_model.train()

    for epoch in range(num_epochs):
        # following line is to see a couple test cases
        print_examples(img_cap_model, device, img_cap_prep_dataset)
        
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
