import os
import zmq
import pandas as pd
import pickle
import threading

from tqdm import tqdm

import numpy as np
from model import unet_2d, unet_3d
# from load import load_atlas, load_stroke

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

                # Store the received DataFrame for processing
                self.data_to_process = received_dataframe

                print("Received DataFrame:")
                print(received_dataframe.head())
            except Exception as e:
                print("Error receiving DataFrame: {}".format(e))

    def __del__(self):
        self.stop()
        self.socket.close()

def eval_unet2d(model, results_plot_path, eval_train=True):
    model.evaluating(results_plot_path, train=eval_train)

def train_unet2d(nifti_csv_df):
    # NiFi data flow already loads the dataset, preprocesses the data and then we now subscribe to ZMQ to 
    # receive the pandas dataframe with preprocessed data filepaths
    skull_strip_seg_unet2d = unet_2d()
    raw_index_imgs = list()
    mask_index_imgs = list()

    for i in tqdm(range(len(nifti_csv_df))):
        # Load the image using ITK
        input_raw_index_img = sitk.ReadImage(nifti_csv_df.raw_index.iloc[i])
        raw_index_imgs.append(input_raw_index_img)

        input_mask_index_img = sitk.ReadImage(nifti_csv_df.mask_index.iloc[i])
        mask_index_imgs.append(input_mask_index_img)

    X_train,X_test,y_train,y_test = model.split(raw_index_imgs, mask_index_imgs)

    # train_stroke_seg.py; pass
    model.training(raw_index_imgs, 100)

    mri_train_results_plot_path = "{}/src/datasets/NFBS_Dataset_NiFi/results/{}/unet2d/plot_metrics/".format(os.path.expanduser("~"), "train")
    # mri_train_results_plot_path = "stroke_results/train/unet2d/plot_metrics/"
    eval_unet2d(skull_strip_seg_unet2d, mri_train_results_plot_path, eval_train=True)

    # mri_test_results_plot_path = "stroke_results/test/unet2d/plot_metrics/"
    # eval_unet2d(stroke_lesion_mask_unet2d, mri_test_results_plot_path, eval_train=False)

def main():
    zmq_subscriber = ZMQSubscriber()

    try:
        print("Subscribing to {} to receive DataFrame...".format(zmq_subscriber.zmq_socket_address))
        zmq_subscriber.start()

        while True:
            # check if there is data to process
            if zmq_subscriber.data_to_process is not None:
                nifti_csv_df = zmq_subscriber.data_to_process

                # pass the data to the training thread
                train_thread = threading.Thread(target=train_unet2d, args=(nifti_csv_df,))
                train_thread.start()

                # reset the data_to_process to None to indicate processing is done
                zmq_subscriber.data_to_process = None
                print("Received data filepaths in pd df to train, finished training, exiting")
                break

            time.sleep(0.2) # sleep for 200 milliseconds

    except KeyboardInterrupt:
        print("Stopping the subscriber...")
        zmq_subscriber.stop()


    # train_unet2d()


if __name__ == "__main__":
    main()
