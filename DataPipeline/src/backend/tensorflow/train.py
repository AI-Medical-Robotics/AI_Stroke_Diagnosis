import os
import zmq
import time
import pandas as pd
import pickle5 as pickle
import threading
import SimpleITK as sitk

from tqdm import tqdm

import numpy as np
from model import unet_2d, unet_3d
# from load import load_atlas, load_stroke


def eval_unet2d(model, results_plot_path, eval_train=True):
    model.evaluating(results_plot_path, train=eval_train)

def train_unet2d(nifti_csv_df):
    # NiFi data flow already loads the dataset, preprocesses the data and then we now subscribe to ZMQ to 
    # receive the pandas dataframe with preprocessed data filepaths
    print("Entering the training thread train_unet2d")
    skull_strip_seg_unet2d = unet_2d()
    # raw_index_imgs = list()
    # mask_index_imgs = list()

    # am I loading too much in memory?
    # print("Loading NFBS Data Prepped Images len = {}: mask_index_imgs".format(len(nifti_csv_df)))
    # for i in tqdm(range(len(nifti_csv_df))):
    #     input_mask_index_img = sitk.ReadImage(nifti_csv_df.mask_index.iloc[i])
    #     mask_index_imgs.append(input_mask_index_img)

    print("Loading NFBS Data Prepped Images: raw_index_imgs")
    # for i in tqdm(range(len(nifti_csv_df))):
    #     # Load the image using ITK
    #     input_raw_index_img = sitk.ReadImage(nifti_csv_df.raw_index.iloc[i])
    #     raw_index_imgs.append(input_raw_index_img)
        
    print("Loaded NFBS Data Prepped Images: raw_index_imgs len = {}".format(len(nifti_csv_df.raw_index.tolist())))

    print("Passing NFBS Data Prepped Image path: mask_index_imgs len = {}".format(len(nifti_csv_df.mask_index.tolist())))

    # X_train,X_test,y_train,y_test = skull_strip_seg_unet2d.split(raw_index_imgs, mask_index_imgs)
    X_train,X_test,y_train,y_test = skull_strip_seg_unet2d.split(nifti_csv_df.raw_index.tolist(), nifti_csv_df.mask_index.tolist())

    # train_stroke_seg.py; pass
    skull_strip_seg_unet2d.training(nifti_csv_df.raw_index.tolist(), epochs=5)

    mri_train_results_plot_path = "{}/src/datasets/NFBS_Dataset_NiFi/results/{}/unet2d/plot_metrics/".format(os.path.expanduser("~"), "train")
    # mri_train_results_plot_path = "stroke_results/train/unet2d/plot_metrics/"
    eval_unet2d(skull_strip_seg_unet2d, mri_train_results_plot_path, eval_train=True)

    # mri_test_results_plot_path = "stroke_results/test/unet2d/plot_metrics/"
    # eval_unet2d(stroke_lesion_mask_unet2d, mri_test_results_plot_path, eval_train=False)


# References:
# - ChatGPT: https://chat.openai.com/c/78252db9-df19-429d-8473-c2b5a46c38e5

# Reference perplexity.ai for logic where subscriber sends confirmation to publisher when message is received:
# https://www.perplexity.ai/search/cf3ca7f0-a7f2-4395-8952-67198b09aef7?s=u

class ZMQSubscriber:
    def __init__(self, zmq_socket_address="tcp://127.0.0.1:5555"):
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

                # Store the received DataFrame for processing
                if self.data_to_process is not None:
                    print("ZMQ Subscriber - Received DataFrame:")
                    print(received_dataframe.head())
                    self.running = False
                    return self.data_to_process

                # Sending confirmation happens late, I realized when the data is passed from
                # this thread to the part of the program that will get the dataframe to load 
                # images for training, this part send confirm part is too slow. Maybe bring
                # training here?
                print("Going to send confirmation to NiFi PutZMQ Publisher")
                self.socket.send_string("Received")
            except Exception as e:
                print("Error receiving DataFrame: {}".format(e))

    def receive_data_then_train(self):
        self.running = True
        nifti_csv_df_to_process = self.receive_data()
        print("Going to train UNet on NIfTI NFBS data for skull strip seg")
        train_unet2d(nifti_csv_df_to_process)

    def __del__(self):
        self.stop()
        self.socket.close()

def main():
    zmq_subscriber = ZMQSubscriber()

    nifti_csv_df = None

    try:
        print("Subscribing to {} to receive DataFrame...".format(zmq_subscriber.zmq_socket_address))
        # zmq_subscriber.start()

        # Not using threading right now.
        zmq_subscriber.receive_data_then_train()

        # while True:
        #     # check if there is data to process
        #     if zmq_subscriber.data_to_process is not None:
        #         nifti_csv_df = zmq_subscriber.data_to_process

        #         if nifti_csv_df is not None:
        #             print("Passed ZMQ Subscriber's nifti_csv_df to training:")
        #             print(nifti_csv_df.head())
        #             zmq_subscriber.stop()
        #             break

                # pass the data to the training thread
                # train_thread = threading.Thread(target=train_unet2d, args=(nifti_csv_df,))
                # train_thread.start()

                # # reset the data_to_process to None to indicate processing is done
                # zmq_subscriber.data_to_process = None
                # print("Received data filepaths in pd df to train, finished training, exiting")
                # break

            # time.sleep(0.2) # sleep for 200 milliseconds

        # train_unet2d(nifti_csv_df)

    except KeyboardInterrupt:
        print("Stopping the subscriber...")
        # zmq_subscriber.stop()
        zmq_subscriber.running = False

    # zmq_subscriber.stop()
    zmq_subscriber.running = False

if __name__ == "__main__":
    main()
