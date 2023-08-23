import os
import random
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


import uuid
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Activation, ReLU, LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Reference perplexity.ai for switching segmentation_models iou_score and focal_loss to keras:
# - https://www.perplexity.ai/search/b787fae0-f8c6-43ee-b6c5-3c7ad3f96ff3?s=u

from tensorflow.keras.metrics import MeanIoU
# from tensorflow.keras.losses import BinaryFocalCrossentropy

# import segmentation_models as sm
# from segmentation_models.metrics import iou_score
# focal_loss = sm.losses.cce_dice_loss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class unet_3d:
    def __init__(self):
        self.print_dl_info()
    
    def print_dl_info(self):
        print(tf.__version__)
        print(keras.__version__)
        device_name = tf.test.gpu_device_name()
        if device_name != "/device:GPU:0":
            raise SystemError("GPU device not found")
        print("Found GPU at: {}".format(device_name))
        pass

    def conv3D_block(self, input, num_filters=3, kernel_size=3):
        x = layers.Conv3D(num_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv3D(num_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        return x

    # encode3D_block
    def encode3D_block(self, input, num_filters):
        x = conv3D_block(input, num_filters=num_filters)
        p = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        return x, p

    def upconv3D_block(self, input, num_filters):
        x = layers.Conv3DTranspose(num_filters, kernel_size=2, strides=2, padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        return x

    def build_DAGMNet3D(self, input_shape=(128,128,128,1), num_classes=1, num_filters=32):
        """
            conv3D_block creates a block of two 3D convolutional layers with batch normalization and ReLU activation.
            upconv3D_block creates 3D transposed convolution layer with batch normalization and ReLU activation.
            build_DAGMNet takes input_shape, number of classes, and creates DAGMNet with five down-sampling blocks
                and four up-sampling blocks.
        """
        inputs = layers.Input(shape=input_shape)

        # Encode 3D blocks 1 - 5
        conv1 = self.conv3D_block(inputs, num_filters=num_filters)
        pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = self.conv3D_block(pool1, num_filters=num_filters*2)
        pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = self.conv3D_block(pool2, num_filters=num_filters*4)
        pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = self.conv3D_block(pool3, num_filters=num_filters*8)
        pool4 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv4)

        conv5 = self.conv3D_block(pool4, num_filters=num_filters*16)

        # UpConv 3D blocks 1 - 4
        # Up 1
        up6 = self.upconv3D_block(conv5, num_filters=num_filters*8)
        concat6 = layers.concatenate([up6, conv4], axis=-1)
        conv6 = self.conv3D_block(concat6, num_filters=num_filters*8)

        # Up 2
        up7 = self.upconv3D_block(conv6, num_filters=num_filters*4)
        concat7 = layers.concatenate([up7, conv3], axis=-1)
        conv7 = self.conv3D_block(concat7, num_filters=num_filters*4)

        # Up 3
        up8 = self.upconv3D_block(conv7, num_filters=num_filters*2)
        concat8 = layers.concatenate([up8, conv2], axis=-1)
        conv8 = self.conv3D_block(concat8, num_filters=num_filters*2)

        # Up 3
        up9 = self.upconv3D_block(conv8, num_filters=num_filters)
        concat9 = layers.concatenate([up9, conv1], axis=-1)
        conv9 = self.conv3D_block(concat9, num_filters=num_filters) 

        if num_classes == 1: # Binary
            activation = "sigmoid"
        else:
            activation = "softmax"

        print("activation = {}".format(activation))
        outputs = layers.Conv3D(num_classes, kernel_size=1, activation=activation)(conv9)

        model = keras.Model(inputs=inputs, outputs=outputs)

        return model

    # Referenced perplexity.ai for focal_loss implementation since its not available in tensorflow 2.6.0: 
    # # https://www.perplexity.ai/search/ce25e8f4-2e4c-4569-be0a-8838d50e17c4?s=u
    def focal_loss(self, y_true, y_pred, gamma=2.0, alpha=0.25):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_loss = -alpha * tf.pow(1 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)


    def training(self, images, labels, epochs=5, learning_rate=1e-4, batch_size=1):
        """
        training(...) pipeline loads the data and splits it into training and validation
        sets using "train_test_split". Then it calculates class weights using
        "class_weight.compute_class_weight" to handle imbalanced data.

        "build_DAGMNet3D(...)" function builds the model, then we compile it with
        Adam optimizer and "SparseCategoricalCrossentropy" loss function. The
        "EarlyStopping" callback is defined to stop training if the validation loss
        doesn't improve after 10 epochs.

        The model is trained on the "model.fit" on the training and validation sets.
        The test set is evaluated using "model.evaluate" and predictions are generated
        using "model.predict".

        "classification_report" and "confusion_matrix" from "sklearn.metrics" are
        generated for evaluation.
        """
        # Load and split data into training and validation sets
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            images, labels, test_size=0.3, random_state=42
        )

        # Calculate class weights for imbalanced data
        # class_weights = class_weight.compute_class_weight(
        #     'balanced', np.unique(y_train), y_train
        # )

        # Define model
        print("images.shape = {}".format(images.shape))
        print("labels.shape = {}".format(labels.shape))
        # (128, 128, 128, 1)
        self.unet3d = self.build_DAGMNet3D(input_shape=(128, 128, 128, 1), num_classes=1)

        # Define optimizer and loss function
        optimizer = Adam(learning_rate=learning_rate)
        # loss_fn = keras.losses.SparseCategoricalCrossentropy()

        # Compile model: focal_loss
        # self.unet3d.compile(optimizer=optimizer, loss="binary_crossentropy", 
        #     metrics=[iou_score, Precision(), Recall(), BinaryAccuracy()])

        self.unet3d.compile(optimizer=optimizer, loss=self.focal_loss,
                          metrics=[MeanIoU(num_classes=1), "accuracy"])

        # Define callbacks
        # early_stopping = EarlyStopping(
        #     monitor='val_loss', patience=10, restore_best_weights=True
        # )

        callbacks = callbacks = [
            ModelCheckpoint("best_stroke_lesion_mask_unet3d_epochs{}.h5".format(epochs), verbose=1, save_best_only=True,
                           metrics=[MeanIoU(num_classes=1), "accuracy"])
        ]

        # Train model
        self.DAGMNet3D_result = self.unet3d.fit(self.x_train, self.y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(self.x_val, self.y_val),
                            # class_weights=class_weights,
                            callbacks=callbacks)

        # Evaluate model on test set
        # test_loss, test_iou, test_prec, test_rec, test_bin_acc = self.unet3d.evaluate(x_test, y_test)
        # print("Test Loss: {}; Test IOU: {}; Test Prec: {}; Test Rec: {}; Test Bin Acc: {}".format(test_loss, test_iou, test_prec, test_rec, test_bin_acc))
        # print("Test Loss: {}; Test IOU: {}; Test Accuracy: {}".format(test_loss, test_iou, test_acc ))

        # Generate predictions
        # y_pred = self.unet3d.predict(x_test)
        # y_pred = np.argmax(y_pred, axis=-1)

        # Generate classification report and confusion matrix
        # target_names = ['Non-Lesion', 'Lesion']
        # print(classification_report(y_test, y_pred, target_names=target_names))
        # print(confusion_matrix(y_test, y_pred))

        return self.DAGMNet3D_result

    def evaluating(self, results_plot_path, train=True, batch_size=48):
        """
        evaluating(...) pipeline loads the trained DAGMNet model and the validation dataset.
        The model is then evaluated on the validation dataset using various evaluation metrics.
        We then calculate the evaluation metrics over the validation dataset and print them.
        We use TensorFlow metrics for accuracy, precision, recall; Scikit-learn's for f1 score,
        and MedPy's binary metrics for F1 score, dice coefficient and Hausdorff distance.
        """
        if not os.path.exists(results_plot_path):
            os.makedirs(results_plot_path)
        
        if train:
            train_loss, train_iou, train_acc, val_loss, val_iou, val_acc = self.DAGMNet3D_result.history.values()
        else:
            # TODO: Need to account for x_test and y_test. This case will fail
            print("Num of Raw 3D MRIs (Patients) For Testing = {}".format(len(self.x_test)))
            # test_gen = self.data_gen(self.X_test, self.y_test, batch_size=batch_size, train=train)
            test_loss, test_iou, test_acc = self.unet3d.evaluate(x_test, y_test)
        
        f = plt.figure(figsize=(20,8))

        
        plt.subplot(1,3,1)
        plt.title("Stroke Lesion {} Loss".format(self.unet3d.name))
        if train:
            plt.plot(train_loss, label="Training")
            plt.plot(val_loss, label="Validation")
        else:
            plt.plot(test_loss, label="Testing")
        plt.legend()
        plt.grid()
        
        plt.subplot(1,3,2)
        plt.title("Stroke Lesion {} Accuracy".format(self.unet3d.name))
        if train:
            plt.plot(train_acc, label="Training")
            plt.plot(val_acc, label="Validation")
        else:
            plt.plot(test_acc, label="Testing")
        plt.legend()
        plt.grid()
        
        plt.subplot(1,3,3)
        plt.title("Stroke Lesion {} IoU".format(self.unet3d.name))
        if train:
            plt.plot(train_iou, label="Training")
            plt.plot(val_iou, label="Validation")
        else:
            plt.plot(test_iou, label="Testing")
        plt.legend()
        plt.grid()

        plt.show()
        
        if train:
            eval_stage = "train_val"
        else:
            eval_stage = "test"
        save_plot_metrics_file = "{}/{}_{}_eval_{}.jpg".format(results_plot_path, eval_stage, self.unet3d.name, uuid.uuid4())
        f.savefig(save_plot_metrics_file, bbox_inches="tight")
        
        plt.close(f)
        return save_plot_metrics_file





    # def inference(self, mri_3d_path, results_seg_path):
    #     """
    #     """
        # load the trained model
        # model = self.build_DAGMNet3D(input_shape=(192, 192, 64, 1), num_classes=2)
        # model.load_weights("path/to/trained/weights.h5")

        # model = load_model("best_stroke_lesion_mask_unet3d_epochs5.h5", custom_objects={
        #     "categorical_crossentropy_plus_dice_loss": focal_loss,
        #     "iou_score": iou_score})

        # tf.saved_model.load('saved_model_path')

        # Get the input and output tensors


class unet_2d:
    def __init__(self):
        self.print_dl_info()
    
    def print_dl_info(self):
        print(tf.__version__)
        print(keras.__version__)
        device_name = tf.test.gpu_device_name()
        if device_name != "/device:GPU:0":
            raise SystemError("GPU device not found")
        print("Found GPU at: {}".format(device_name))
        pass

    def split(self, resized_img, resized_mask):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            resized_img, resized_mask, test_size=0.05)
        return self.X_train, self.X_val, self.y_train, self.y_val

    def data_gen(self, mri_3d_list, brain_3d_mask_list, batch_size, train=True):
        '''Custom data generator to take 3D MRIs, split them into list of MRI slices per 3D MRI
            to feed MRIs to model'''
        patients_mri_slices = [] # 125 * 96 = 12K
        patients_mask_slices = []
        # loop num patients had 3D MRIs = 125; get each voxel (3D image)
            # each 3D MRI has 96 MRI 2D slices
        print("datagen: len(mri_3d_list) = ", len(mri_3d_list))
        # need to make sure the list of raw_mri_slices and brain_mask_slices are sorted.
        # maybe add a file associated with it
        for patient_i in range(len(mri_3d_list)):
#         for patient_i in range(125):
            raw_mri_voxel = nib.load(mri_3d_list[patient_i]).get_data()
            raw_mri_slices = [raw_mri_voxel[i] for i in range(len(raw_mri_voxel))]
            patients_mri_slices.extend(raw_mri_slices)
            
            brain_mask_voxel = nib.load(brain_3d_mask_list[patient_i]).get_data()
            brain_mask_slices = [brain_mask_voxel[i] for i in range(len(brain_mask_voxel))]
            patients_mask_slices.extend(brain_mask_slices)

        print("len(patients_mri_slices) = {}".format(len(patients_mri_slices)))
        print("len(patients_mask_slices) = {}".format(len(patients_mask_slices)))
    
        c = 0
        # List of training MRIs
        n = [i for i in range(len(patients_mri_slices))]
        random.shuffle(n)
#         print("n = {}".format(n))
        
        while(True):
            # adding extra dimensions as conv2d takes file of size 4
#             img = np.zeros((batch_size, 233, 189, 1)).astype("float")
#             mask = np.zeros((batch_size, 233, 189, 1)).astype("float")
            img = np.zeros((batch_size, 128, 160, 1)).astype("float32")
            mask = np.zeros((batch_size, 128, 160, 1)).astype("float32")
            
            for i in range(c, c+batch_size):
                train_mri_img = patients_mri_slices[n[i]]
                train_mri_img = np.expand_dims(train_mri_img, -1)
                # print("train_mri_img.shape = {}".format(train_mri_img.shape))
                
                train_brain_mask = patients_mask_slices[n[i]]
                train_brain_mask = np.expand_dims(train_brain_mask, -1)
                # print("train_brain_mask.shape = {}".format(train_brain_mask.shape))
                
                img[i-c] = train_mri_img
                mask[i-c] = train_brain_mask
            c += batch_size
            if(c+batch_size >= len(patients_mri_slices)):
                if train:
                    c=0
                    random.shuffle(n)
                else: # in cases we use data_gen for testing
                    break
            
            yield img,mask
    
    def conv2D_block(self, input, filters=3, kernel_size=3):
        x = Conv2D(filters, kernel_size, padding = 'same')(input)
#         x = Activation('relu')(x)
        x = Activation(LeakyReLU())(x)
        x = BatchNormalization()(x) # Not in the original network
        
        x = Conv2D(filters, kernel_size, padding='same')(x)
#         x = Activation('relu')(x) # Not in the original network
        x = Activation(LeakyReLU())(x)
        x = BatchNormalization()(x)
        return x
    
    def encoder2D_block(self, input, filters):
        '''encoder block conv block followed by maxpooling'''
        x = self.conv2D_block(input, filters)
        p = MaxPooling2D((2,2))(x)
        dp = Dropout(0.4)(p)
        return x, dp
    
    def decoder2D_block(self, input, skip_features, filters):
        '''decoder block skip features get input from encoder for concatenation'''
        x = Conv2DTranspose(filters, (2,2), strides=2, padding='same')(input)
        x = concatenate([x, skip_features])
        x = Dropout(0.4)(x)
        x = self.conv2D_block(x, filters)
        return x
    
    def build2D_unet(self, input_shape, n_classes, filters=64):
        inputs = Input(input_shape)
        
        conv_dwns1, p1 = self.encoder2D_block(inputs, filters)
        conv_dwns2, p2 = self.encoder2D_block(p1, filters*2)
        conv_dwns3, p3 = self.encoder2D_block(p2, filters*4)
        conv_dwns4, p4 = self.encoder2D_block(p3, filters*8)
        
        bridge1 = self.conv2D_block(p4, filters*16)
        
        conv_ups1 = self.decoder2D_block(bridge1, conv_dwns4, filters*8)
        conv_ups2 = self.decoder2D_block(conv_ups1, conv_dwns3, filters*4)
        conv_ups3 = self.decoder2D_block(conv_ups2, conv_dwns2, filters*2)
        conv_ups4 = self.decoder2D_block(conv_ups3, conv_dwns1, filters)
        
        if n_classes == 1: # Binary
            activation = "sigmoid"
        else:
            activation = "softmax"
        
        # Change the activation based on n_classes
        outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(conv_ups4)
        print(activation)
        
        model = Model(inputs, outputs, name="2D-UNet")
        return model
    
    # Referenced perplexity.ai for focal_loss implementation since its not available in tensorflow 2.6.0: 
    # # https://www.perplexity.ai/search/ce25e8f4-2e4c-4569-be0a-8838d50e17c4?s=u
    def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_loss = -alpha * tf.pow(1 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)

    # training function part of preprocessing class
    def training(self, train_t1w_raw_ind, epochs=100, learn_rate=1e-2, batch_size=48):

        print("Num of Raw 3D MRIs (Patients) For Training = {}".format(len(self.X_train)))
        print("Num of Raw 3D MRIs (Patients) For Validation = {}".format(len(self.X_val)))
#         raw_mri_slices = nib.load(train_t1w_raw[0]).get_data()
        raw_mri_slices = nib.load(train_t1w_raw_ind[0]).get_data()
        middle_mri_slice = raw_mri_slices.shape[0]//2
        print("Raw MRI Slice = {}".format(middle_mri_slice))
        print("Raw MRI Slice - Shape = {}".format(raw_mri_slices[middle_mri_slice].shape))
        
        # Get image height from raw MRI slice 0
        IMG_HEIGHT=raw_mri_slices[middle_mri_slice].shape[0]
        print("Raw MRI Slice - IMG_HEIGHT = {}".format(IMG_HEIGHT))

        IMG_WIDTH=raw_mri_slices[middle_mri_slice].shape[1]
        print("Raw MRI Slice - IMG_WIDTH = {}".format(IMG_WIDTH))

        try:
            IMG_CHANNELS = raw_mri_slices[middle_mri_slice].shape[3]
        except IndexError:
            IMG_CHANNELS = 1
        print("Raw MRI Slice - IMG_CHANNELS = {}".format(IMG_CHANNELS))

        # Raw MRI Slice 0 - input_shape
        input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

#         epochs=7 # epochs = 60
        train_gen = self.data_gen(self.X_train, self.y_train, batch_size=batch_size)
        val_gen = self.data_gen(self.X_val, self.y_val, batch_size=batch_size)
        channels=1
        
        self.unet2d = self.build2D_unet(input_shape, n_classes=1)


        # learning rate = 3E-4, 1E-5
        self.unet2d.summary()
        self.unet2d.compile(optimizer=Adam(learning_rate=learn_rate), loss=self.focal_loss,
                          metrics=[MeanIoU(num_classes=1), "accuracy"])
        # fitting the model
        callbacks = callbacks = [
            ModelCheckpoint("best_stroke_lesion_mask_unet2d_epochs{}.h5".format(epochs), verbose=1, save_best_only=True,
                           metrics=[MeanIoU(num_classes=1), "accuracy"])
        ]
        self.unet2d_result = self.unet2d.fit(train_gen, steps_per_epoch=16, epochs=epochs,
                               validation_data = val_gen, validation_steps=16,
                               initial_epoch=0, callbacks=callbacks)
        return self.unet2d_result
        
    def evaluating(self, results_plot_path, train=True, batch_size=48):
        if not os.path.exists(results_plot_path):
            os.makedirs(results_plot_path)
        
        if train:
            train_loss, train_iou, train_acc, val_loss, val_iou, val_acc = self.unet2d_result.history.values()
        else:
            # TODO: Need to account for x_test and y_test. This case will fail
            print("Num of Raw 3D MRIs (Patients) For Testing = {}".format(len(self.X_test)))
            test_gen = self.data_gen(self.X_test, self.y_test, batch_size=batch_size, train=train)
            test_loss, test_iou, test_acc = self.unet2d.evaluate(test_gen)
        
        f = plt.figure(figsize=(20,8))
        
        plt.subplot(1,3,1)
        plt.title("Stroke Lesion {} Loss".format(self.unet2d.name))
        if train:
            plt.plot(train_loss, label="Training")
            plt.plot(val_loss, label="Validation")
        else:
            plt.plot(test_loss, label="Testing")
        plt.legend()
        plt.grid()
        
        plt.subplot(1,3,2)
        plt.title("Stroke Lesion {} Accuracy".format(self.unet2d.name))
        if train:
            plt.plot(train_acc, label="Training")
            plt.plot(val_acc, label="Validation")
        else:
            plt.plot(test_acc, label="Testing")
        plt.legend()
        plt.grid()
        
        plt.subplot(1,3,3)
        plt.title("Stroke Lesion {} IoU".format(self.unet2d.name))
        if train:
            plt.plot(train_iou, label="Training")
            plt.plot(val_iou, label="Validation")
        else:
            plt.plot(test_iou, label="Testing")
        plt.legend()
        plt.grid()

        plt.show()
        
        if train:
            eval_stage = "train_val"
        else:
            eval_stage = "test"
        save_plot_metrics_file = "{}/{}_{}_eval_{}.jpg".format(results_plot_path, eval_stage, self.unet2d.name, uuid.uuid4())
        f.savefig(save_plot_metrics_file, bbox_inches="tight")
        
        plt.close(f)
        return save_plot_metrics_file
        
#     def inference(self, mri_3d_path, results_seg_path):
#         """Generates prediction for a sample image and does the plotting: img_path refers to mri_3d_path"""
#         if not os.path.exists(results_seg_path):
#             os.makedirs(results_seg_path)
        
#         # applying bias correction
#         n4 = N4BiasFieldCorrection()
#         n4.inputs.dimension = 3
#         n4.inputs.shrink_factor = 3
#         n4.inputs.n_iterations = [20, 10, 10, 5]
#         n4.inputs.input_image = mri_3d_path
#         n4.inputs.output_image = mri_3d_path
#         res = n4.run()
#         print("Applied Bias Correction")
        
#         # resizing and cropping
#         target_shape = np.array((96,128,160))
#         new_resolution = [2,]*3
#         new_affine = np.zeros((4,4))
#         new_affine[:3,:3] = np.diag(new_resolution)
#         # putting point 0,0,0 in the middle of the new volume - this
#         # could be refined in the future
#         new_affine[:3,3] = target_shape*new_resolution/2.*-1
#         new_affine[3,3] = 1.
#         downsampled_and_cropped_nii = resample_img(mri_3d_path, target_affine=new_affine,
#                     target_shape=target_shape, interpolation="nearest")
#         downsampled_and_cropped_nii.to_filename(mri_3d_path)
#         mri_voxel = sitk.ReadImage(mri_3d_path)
#         print("Resized & Cropped to mri_voxel.shape")
        
#         # intensity normalizing
#         resacleFilter = sitk.RescaleIntensityImageFilter()
#         resacleFilter.SetOutputMaximum(255)
#         resacleFilter.SetOutputMinimum(0)
#         mri_voxel = resacleFilter.Execute(mri_voxel)
#         sitk.WriteImage(mri_voxel, mri_3d_path)
#         print("Applied Intensity Normalization")
        
#         # getting predictions: do predictions on all of MRI 2D slices of a patients MRI 3D voxel
#         orig_mri_voxel = nib.load(mri_3d_path).get_data()
        
#         patient_orig_mri_slices = [] # 125 * 96 = 12K
#         # loop num patients had 3D MRIs = 125; get each voxel (3D image)
#             # each 3D MRI has 96 MRI 2D slices

#         orig_mri_slices = [orig_mri_slice for orig_mri_slice in tqdm(orig_mri_voxel)]
#         print("len(orig_mri_slices) = {}".format(len(orig_mri_slices)))
#         patient_orig_mri_slices.extend(orig_mri_slices)
            
#         patients_mri_slices_testset = np.array(patient_orig_mri_slices)
#         patients_mri_slices_testset = np.expand_dims(patients_mri_slices_testset, axis = 3)
        
# #         orig_mri_voxel = np.expand_dims(orig_mri_voxel, -1) # accounts for the np.zeros similar to saw in data_gen, but previous 5d (batch, 96, 128, 160, 1)
# #         orig_mri_voxel = np.expend_dims(orig_mri_voxel, 0) # we need new 4d (batch, 128, 160, 1); I think just need to comment out
#         model = load_model("best_stroke_lesion_mask_unet2d.h5", custom_objects={
#             "categorical_crossentropy_plus_dice_loss": focal_loss,
#             "iou_score": iou_score})
#         pred_mri_slices = model.predict(patients_mri_slices_testset)
#         print("len(pred_mri_slices) = {}".format(len(pred_mri_slices)))
#         pred_mri_slices = np.squeeze(pred_mri_slices)
#         orig_mri_voxel = nib.load(mri_3d_path).get_data()
#         print("Performed Stroke Lesion Segmentation")
        
#         # converting prediction to nifti file
#         func = nib.load(mri_3d_path)
#         ni_mri_voxel = nib.Nifti1Image(pred_mri_slices, func.affine)
#         nib.save(ni_mri_voxel, "pred_T1w_stroke_lesion_mask.nii.gz")
#         pred_mri_voxel = nib.load("pred_T1w_stroke_lesion_mask.nii.gz")
#         print("Converted Prediction to Nifti")
        
#         # creating binary mask and stripping stroke lesion from raw image
#         pred_mask = math_img("img > 0.25", img = pred_mri_voxel)
#         crop = pred_mask.get_data()*orig_mri_voxel
#         print("Created Binary Mask & Stripped stroke lesion from Raw Img")
#         print("pred_mri_voxel.shape[0] = {}".format(pred_mri_voxel.shape[0]))
        
#         # plotting predictions
#         pred_mri_voxel = nib.load("pred_T1w_stroke_lesion_mask.nii.gz").get_data()
        
#         steps = int(len(orig_mri_voxel)/2)
#         print("Run inference {} times based on 'len(orig_mri_voxel)/2'".format(steps))
#         for i in range(0, len(orig_mri_voxel), steps): # raw_mri_voxel[i]
#             fig, ax = plt.subplots(1,3,figsize=(15,10))
#             ax[0].set_title("T1w Original MRI Slice {} (cropped)".format(i))
#             ax[0].imshow(orig_mri_voxel[i])
#             ax[1].set_title("T1w Stroke Lesion Segmentation MRI Slice {}".format(i))
#             ax[1].imshow(pred_mri_voxel[i])
#             ax[2].set_title("T1w Stroke Lesion Stripped MRI Slice {}".format(i))
#             ax[2].imshow(crop[i])

#             save_plot_metrics_file = "{}/{}_{}_pred_mri_slice_{}_uuid_{}.jpg".format(results_seg_path, "stroke_seg", model.name, i, uuid.uuid4())
#             fig.savefig(save_plot_metrics_file, bbox_inches="tight")

#         plt.show()
        
#         # converting skull stripped to nifti file
#         ni_img = nib.Nifti1Image(crop, func.affine)
#         nib.save(ni_img, "pred_T1w_stroke_lesion_mask.nii.gz")
#         print("Predicted files stores as : pred_T1w_stroke_lesion_mask.nii.gz")
        
    # def plotting(self, filename):
    #     """Plots an interactive plot"""
    #     img = nii.mean_img(filepath)
    #     plotting.view_img(img, bg_img = img)
