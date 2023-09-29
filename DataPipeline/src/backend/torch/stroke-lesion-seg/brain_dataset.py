import torch
import SimpleITK as sitk

# Reference perplexity.ai for pytorch 3D UNet skull strip seg model
# https://www.perplexity.ai/search/0df235a1-27ba-4b67-bf7b-89c2500685c7?s=u

# Also reference perplexity.ai for pytorch train 3D UNet skull strip seg model
# https://www.perplexity.ai/search/cf74b6d5-9888-462b-9063-e90859bbf389
# Refer to section on "MRIDataset(Dataset)"

class StrokeMRIDataset(torch.utils.data.Dataset):
    def __init__(self, brain_voxel_list, stroke_mask_list, debug=False):
        self.brain_voxel_paths_ = brain_voxel_list
        self.stroke_mask_paths_ = stroke_mask_list
        self.debug = debug

    def __len__(self):
        return len(self.brain_voxel_paths_)

    def __getitem__(self, idx):
        if self.debug:
            print("idx = {}".format(idx))

        # sitk to torch tensor dims (channels, depth, height, width)
        if self.debug:
            print("self.brain_voxel_paths_[idx] = {}".format(self.brain_voxel_paths_[idx]))
        brain_voxel = sitk.ReadImage(self.brain_voxel_paths_[idx])
        brain_voxel_array = sitk.GetArrayFromImage(brain_voxel)
        brain_voxel_tensor = torch.tensor(brain_voxel_array).float()
        if self.debug:
            print("brain_voxel_tensor shape = {}".format(brain_voxel_tensor.shape))
            print("self.stroke_mask_paths_[idx] = {}".format(self.stroke_mask_paths_[idx]))

        stroke_mask_voxel = sitk.ReadImage(self.stroke_mask_paths_[idx])
        stroke_mask_voxel_array = sitk.GetArrayFromImage(stroke_mask_voxel)
        stroke_mask_voxel_tensor = torch.from_numpy(stroke_mask_voxel_array).float()

        if self.debug:
            print("stroke_mask_voxel_tensor shape = {}".format(stroke_mask_voxel_tensor.shape))
        
        return brain_voxel_tensor, stroke_mask_voxel_tensor

