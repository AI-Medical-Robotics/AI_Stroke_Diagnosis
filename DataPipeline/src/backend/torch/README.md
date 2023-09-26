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
Epoch 1:
=> Saving Checkpoint
Running Evaluation
BCE With Logits Loss: 0.0166
Acc Ratio 25281064/25559040 with Acc 98.91
Dice Score: 0.5600466728210449


Epoch 2:
=> Saving Checkpoint
Running Evaluation
BCE With Logits Loss: 0.012
Acc Ratio 25311801/25559040 with Acc 99.03
Dice Score: 0.7302404046058655

Epoch 3:
=> Saving Checkpoint
Running Evaluation
BCE With Logits Loss: 0.00735
Acc Ratio 25444218/25559040 with Acc 99.55
Dice Score: 0.7939558029174805
~~~

![](skull-strip-seg/saved_seg_slices/brain_ssseg_id_0_slice_48_skull_strip_seg.png)

### ICPSR 38464 Dataset

coming soon... some results

## Stroke Lesion Segmentation

- General Stroke Lesion Segmentation (trained on Ischemic, Hemorrhagic, etc)

Specifically trained on:

- Ischemic Stroke Lesion Segmentation
- Hemorrhagic Stroke Lesion Segmentation

## Stroke Lesion Image Captioning

- Medical History per 3D MRI Voxel (or 2D image slice sample)