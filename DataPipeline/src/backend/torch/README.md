# PyTorch DL for Stroke Diagnosis

3D UNet for each segmentation.

## Skull Stripping Segmentation

Heres the path to our PyTorch 3D UNet Skull Stripping Segmentation DL Training Pipeline: [AI_Stroke_Diagnosis/DataPipeline/src/backend/torch/skull-strip-seg](skull-strip-seg)

- brain_dataset.py
- torch_model.py
- train.py

### NFBS Dataset

After training on NFBS dataset, each 3D UNet model gets saved per epoch and we see each one is about 1.1GB. Here are the performance metrics:

- BCE Logits Loss, Accuracy and Dice Score Metrics for 3D UNet

~~~
Compiling UNet3D with Adam, BCEWithLogitsLoss, Dice Score, Accuracy
Training UNet3D on nfbs across 7 epochs
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56/56 [01:25<00:00,  1.52s/it, loss=0.022]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 25284436/25559040 with Acc 98.93
Dice Score: 0.5988669395446777
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56/56 [01:25<00:00,  1.52s/it, loss=0.00735]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 25350257/25559040 with Acc 99.18
Dice Score: 0.5812737345695496
/home/bizon/src/AI_Stroke_Diagnosis/DataPipeline/src/backend/torch/skull-strip-seg/train.py:286: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, ax = plt.subplots(1, 2, figsize=(14, 10))
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56/56 [01:23<00:00,  1.50s/it, loss=0.00712]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 25410100/25559040 with Acc 99.42
Dice Score: 0.6849814653396606
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56/56 [01:24<00:00,  1.51s/it, loss=0.00571]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 25492664/25559040 with Acc 99.74
Dice Score: 0.8914580941200256
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56/56 [01:24<00:00,  1.50s/it, loss=0.00465]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 25501619/25559040 with Acc 99.78
Dice Score: 0.9084030389785767
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56/56 [01:23<00:00,  1.50s/it, loss=0.00653]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 25389822/25559040 with Acc 99.34
Dice Score: 0.7877423763275146
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56/56 [01:24<00:00,  1.50s/it, loss=0.00533]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 25501578/25559040 with Acc 99.78
Dice Score: 0.9135783314704895
~~~

![](skull-strip-seg/saved_seg_slices/brain_ssseg_id_0_slice_48_skull_strip_seg.png)

### ICPSR 38464 Dataset

After training on ICPSR 38464 dataset, each 3D UNet model gets saved per epoch and we see each one is about 1.1GB. Here are the performance metrics:

- BCE Logits Loss, Accuracy and Dice Score Metrics for 3D UNet

~~~
Compiling UNet3D with Adam, BCEWithLogitsLoss, Dice Score, Accuracy
Training UNet3D on icpsr across 7 epochs
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 846/846 [03:38<00:00,  3.88it/s, loss=0.0157]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 236423428/237895680 with Acc 99.38
Dice Score: 0.5113473534584045
/home/bizon/src/AI_Stroke_Diagnosis/DataPipeline/src/backend/torch/skull-strip-seg/train.py:207: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, ax = plt.subplots(1, 2, figsize=(14, 10))
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 846/846 [03:37<00:00,  3.89it/s, loss=0.00894]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 236988598/237895680 with Acc 99.62
Dice Score: 0.5275589227676392
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 846/846 [03:35<00:00,  3.92it/s, loss=0.0113]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 237066952/237895680 with Acc 99.65
Dice Score: 0.5891886949539185
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 846/846 [03:36<00:00,  3.91it/s, loss=0.0055]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 236993849/237895680 with Acc 99.62
Dice Score: 0.45991578698158264
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 846/846 [03:36<00:00,  3.91it/s, loss=0.00493]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 237070120/237895680 with Acc 99.65
Dice Score: 0.5947967767715454
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 846/846 [03:35<00:00,  3.93it/s, loss=0.00501]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 237116484/237895680 with Acc 99.67
Dice Score: 0.5226009488105774
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 846/846 [04:33<00:00,  3.09it/s, loss=0.00534]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 236830325/237895680 with Acc 99.55
Dice Score: 0.6142829060554504
~~~

![](skull-strip-seg/icpsr/saved_seg_slices/700/brain_ssseg_id_38_slice_10_skull_strip_seg.png)

## Stroke Lesion Segmentation

Now that we saved skull stripped segmentation data preprocessed MRI voxels from our brain dataset (ICPSR38464), we'll try training our 3D UNet on Stroke Lesion Segmentation of varying sizes of stroke lesions.

- General Stroke Lesion Segmentation (trained on Ischemic, Hemorrhagic, etc)

Specifically trained on:

- Ischemic Stroke Lesion Segmentation
- Hemorrhagic Stroke Lesion Segmentation

### ICPSR 38464 Dataset

- General Stroke Lesion Segmentation



- Ischemic Stroke Lesion Segmentation

- Hemorrhagic Stroke Lesion Segmentation

## Stroke Lesion Image Captioning

- Medical History per 3D MRI Voxel (or 2D image slice sample)