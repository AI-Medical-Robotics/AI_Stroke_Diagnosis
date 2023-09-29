import torch
import SimpleITK as sitk

# Reference perplexity.ai for pytorch 3D UNet skull strip seg model
# https://www.perplexity.ai/search/0df235a1-27ba-4b67-bf7b-89c2500685c7?s=u

# Also reference perplexity.ai for pytorch train 3D UNet skull strip seg model
# https://www.perplexity.ai/search/cf74b6d5-9888-462b-9063-e90859bbf389
# Refer to section on "MRIDataset(Dataset)"

class BrainMRIDataset(torch.utils.data.Dataset):
    def __init__(self, brain_voxel_list, skull_mask_list, debug=False):
        self.brain_voxel_paths = brain_voxel_list
        self.skull_mask_paths = skull_mask_list
        self.debug = debug

    def __len__(self):
        return len(self.brain_voxel_paths)

    def __getitem__(self, idx):
        if self.debug:
            print("idx = {}".format(idx))

        # sitk to torch tensor dims (channels, depth, height, width)
        if self.debug:
            print("self.brain_voxel_paths[idx] = {}".format(self.brain_voxel_paths[idx]))
        voxel = sitk.ReadImage(self.brain_voxel_paths[idx])
        voxel_array = sitk.GetArrayFromImage(voxel)
        voxel_tensor = torch.tensor(voxel_array).float()
        if self.debug:
            print("voxel_tensor shape = {}".format(voxel_tensor.shape))
            print("self.skull_mask_paths[idx] = {}".format(self.skull_mask_paths[idx]))

        mask_voxel = sitk.ReadImage(self.skull_mask_paths[idx])
        mask_voxel_array = sitk.GetArrayFromImage(mask_voxel)
        mask_voxel_tensor = torch.from_numpy(mask_voxel_array).float()

        if self.debug:
            print("mask_voxel_tensor shape = {}".format(mask_voxel_tensor.shape))
        
        return voxel_tensor, mask_voxel_tensor

