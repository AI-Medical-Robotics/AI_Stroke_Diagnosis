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
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1088/1088 [04:38<00:00,  3.90it/s, loss=0.0127]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 78927249/79298560 with Acc 99.53
Dice Score: 0.24076980352401733
/home/bizon/src/AI_Stroke_Diagnosis/DataPipeline/src/backend/torch/skull-strip-seg/train.py:286: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, ax = plt.subplots(1, 2, figsize=(14, 10))
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1088/1088 [04:39<00:00,  3.90it/s, loss=0.00772]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 78898421/79298560 with Acc 99.50
Dice Score: 0.5353542566299438
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1088/1088 [04:39<00:00,  3.89it/s, loss=0.00587]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 79028701/79298560 with Acc 99.66
Dice Score: 0.5896857976913452
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1088/1088 [04:37<00:00,  3.91it/s, loss=0.00795]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 79042141/79298560 with Acc 99.68
Dice Score: 0.5159244537353516
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1088/1088 [04:39<00:00,  3.90it/s, loss=0.00333]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 79085863/79298560 with Acc 99.73
Dice Score: 0.6671044826507568
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1088/1088 [04:38<00:00,  3.91it/s, loss=0.027]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 78981264/79298560 with Acc 99.60
Dice Score: 0.3892383277416229
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1088/1088 [04:38<00:00,  3.91it/s, loss=0.0057]
=> Saving Checkpoint
Running Evaluation
Acc Ratio 79055849/79298560 with Acc 99.69
Dice Score: 0.6671208739280701
~~~

![](skull-strip-seg/icpsr/saved_seg_slices/700/brain_ssseg_id_38_slice_10_skull_strip_seg.png)

## Stroke Lesion Segmentation

- General Stroke Lesion Segmentation (trained on Ischemic, Hemorrhagic, etc)

Specifically trained on:

- Ischemic Stroke Lesion Segmentation
- Hemorrhagic Stroke Lesion Segmentation

## Stroke Lesion Image Captioning

- Medical History per 3D MRI Voxel (or 2D image slice sample)