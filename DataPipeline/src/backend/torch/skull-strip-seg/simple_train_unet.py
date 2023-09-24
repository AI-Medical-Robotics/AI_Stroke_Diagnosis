import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import SimpleITK as sitk

from torch_model import SimpleUNet

class BrainMRIDataset(torch.utils.data.Dataset):
    def __init__(self, brain_voxel_list, brain_mask_list):
        self.voxel_paths = brain_voxel_list
        self.mask_paths = brain_mask_list

    def __len__(self):
        return len(self.voxel_paths)

    def __getitem__(self, idx):
        print("idx = {}".format(idx))

        # sitk to torch tensor dims (channels, depth, height, width)
        voxel = sitk.ReadImage(self.voxel_paths)
        voxel_array = sitk.GetArrayFromImage(voxel)
        voxel_tensor = torch.tensor(voxel_array).float()
        print("voxel shape = {}".format(voxel.shape))

        mask_voxel = sitk.ReadImage(self.mask_paths)
        mask_voxel_array = sitk.GetArrayFromImage(mask_voxel)
        mask_voxel_tensor = torch.from_numpy(mask_voxel_array).float()
        print("mask_voxel shape = {}".format(mask_voxel.shape))
        
        return voxel_tensor, mask_voxel_tensor

def train_skull_strip_seg_model(nifti_csv_data, epochs=3):
    X_train, X_val, y_train, y_val = train_test_split(nifti_csv_data["intensity_norm"].tolist(), nifti_csv_data["mask_index"].tolist(), test_size=0.1)

    print("X_train len = {}".format(len(X_train)))
    print("X_val len = {}".format(len(X_val)))

    print("Creating brain train & val datasets")
    
    brain_train_dataset = BrainMRIDataset(X_train, y_train)
    brain_val_dataset = BrainMRIDataset(X_val, y_val)

    print("brain_train_dataset len = {}".format(len(brain_train_dataset)))
    print("brain_val_dataset len = {}".format(len(brain_val_dataset)))

    print("Creating brain train & val dataloaders")

    train_loader = DataLoader(brain_train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(brain_val_dataset, batch_size=4, shuffle=False)

    print("Creating Skull Strip Seg UNet3D model")

    # in_channels=1 for 1 medical image modality T2-weighted; out_channels=1 for skull vs non-skull
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet3d_model = UNet3D(in_channels=1, out_channels=1, init_features=4).to(device)
    # unet3d_model = SimpleUNet().to(device)

    print("Compiling UNet3D with Adam, FocalLoss, IoU, Accuracy")

    # compile model with Adam optimizer, focal loss, metrics
    bce_criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(unet3d_model.parameters(), lr=0.001)

    # Define desired variables for tracking best validation performance
    best_val_loss = float("inf")
    best_val_metric = float("-inf")

    print("Training UNet3D across {} epochs".format(epochs))

    unet3d_model.train()
    step = 100
    for epoch in range(epochs):
        print("Epoch {}: Train across batch_idx and brain_data and target from train_loader".format(epoch))
        for batch_idx, (brain_data, target) in enumerate(train_loader):
            brain_data = brain_data.to(device)
            target = target.to(device)
            print("batch_idx = {}; brain_data len = {}; target len = {}".format(batch_idx, len(brain_data), len(target)))
            
            optimizer.zero_grad()
            outputs = unet3d_model(brain_data.unsqueeze(1))
            loss = bce_criterion(outputs, target.unsqueeze(1))
            loss.backward()
            optimizer.step()

        print("Running Evaluation")

        # validation
        unet3d_model.eval()
        val_loss = 0.0
        val_metric = 0.0
        with torch.no_grad():
            for val_brain_data, val_target in val_loader:
                val_brain_data = val_brain_data.to(device)
                val_target = val_target.to(device)

                val_output = unet3d_model(val_brain_data.unsqueeze(1))
                val_loss += bce_criterion(val_output, val_target.unsqueeze(1))
        
        # Calculate average validation loss and IoU score
        val_loss /= len(val_loader)

        # Update best validation performance and save model if it improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(unet3d_model.state_dict(), "best_unet3d_model_loss_{}.pt".format(step))

        print(f"Epoch {epoch+1}: Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")

        step += 100

        unet3d_model.train()

def main_alt():
    nifti_csv_df = pd.read_csv("skull_strip_seg_prep.csv")
    train_skull_strip_seg_model(nifti_csv_df, epochs=3)

if __name__ == "__main__":
    # main()
    main_alt()
