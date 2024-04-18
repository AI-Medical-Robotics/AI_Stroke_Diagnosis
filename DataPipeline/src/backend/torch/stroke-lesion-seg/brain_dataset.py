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

    def get_stroke_lesion_mris(self, idx):
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

    def __getitem__(self, idx):
        if self.debug:
            print("idx = {}".format(idx))

        brain_voxel_tensor, stroke_mask_voxel_tensor = self.get_stroke_lesion_mris(idx)

        return brain_voxel_tensor, stroke_mask_voxel_tensor


class StrokeFilterMRIDataset(torch.utils.data.Dataset):
    def __init__(self, brain_voxel_list, stroke_mask_list, stroke_type="both", debug=False):
        self.brain_voxel_paths_ = brain_voxel_list
        self.stroke_mask_paths_ = stroke_mask_list
        self.stroke_type = stroke_type
        self.brain_voxel_paths_, self.stroke_mask_paths_ = self.get_filtered_stroke_lesion_mris()
        self.stroke_type = stroke_type
        self.debug = debug

    def __len__(self):
        return len(self.brain_voxel_paths_)

    def get_filtered_stroke_lesion_mris(idx):
        # TODO (JG): Instead of using __getitem__ index, we'll iterate over lists and check our clinical labels if they equal our stroke_type

        # sitk to torch tensor dims (channels, depth, height, width)
        if self.debug:
            print("self.voxid_to_prep_vox_paths_[idx] = {}".format(self.voxid_to_prep_vox_paths_[idx]))
        with open(self.voxid_to_prep_vox_paths_[idx], "rb") as file:
            pickle_voxid_to_prep_vox = pickle.load(file)
        
        voxel_id_from_prepvox = pickle_voxid_to_prep_vox[0]
        prep_voxel_filepath = pickle_voxid_to_prep_vox[1]
        prep_voxel = sitk.ReadImage(prep_voxel_filepath)
        prep_voxel_array = sitk.GetArrayFromImage(prep_voxel)
        prep_voxel_tensor = torch.tensor(prep_voxel_array).float()


        # Will have a voxid and filepath
        stroke_mask_voxel = sitk.ReadImage(self.stroke_mask_paths_[idx])
        stroke_mask_voxel_array = sitk.GetArrayFromImage(stroke_mask_voxel)
        stroke_mask_voxel_tensor = torch.from_numpy(stroke_mask_voxel_array).float()


        if self.debug:
            print("self.voxid_to_cap_paths_[idx] = {}".format(self.voxid_to_cap_paths_[idx]))

        with open(self.voxid_to_cap_paths_[idx], "rb") as file:
            pickle_voxid_to_prepcaps = pickle.load(file)

        # check the clinical label, use it to filter list that we passed into Dataset, then use filtered list
        voxel_id_from_caps = list(pickle_voxid_to_prepcaps.keys())[0]
        prep_clinical_captions_list = list(pickle_voxid_to_prepcaps.values())[0]

        # if self.caption_type == "short_caption":
        clinical_label = prep_clinical_captions_list[0]
            # print(f"short caption: clinical_label = {clinical_label}")
        prep_captions_str = clinical_label
        # elif self.caption_type == "long_caption":
        #     caption = prep_clinical_captions_list[1]
        #     # print(f"long caption: caption = {caption}")
        #     prep_captions_str = caption

        return brain_voxel_tensor, stroke_mask_voxel_tensor

    def get_stroke_lesion_mris(self, idx):
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


    def __getitem__(self, idx):
        if self.debug:
            print("idx = {}".format(idx))


        brain_voxel_tensor, stroke_mask_voxel_tensor = self.get_stroke_lesion_mris(idx)

        return brain_voxel_tensor, stroke_mask_voxel_tensor

