import spacy # for tokenizer
import torch
import SimpleITK as sitk

import pickle5 as pickle

from torch.nn.utils import rnn

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Reference perplexity.ai for preparing data for pytorch and training pytorch:
# https://www.perplexity.ai/search/b1046a4a-c702-481b-b719-ace5706e024e?s=u

spacy_eng = spacy.load("en_core_web_sm")

class StrokeMRIVocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        tokenized_text = spacy_eng.tokenizer(text)
        tokenized_corpus = []

        for token in tokenized_text:
            tokenized_corpus.append(token.text.lower())

        return tokenized_corpus

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

        # print(f"self.itos = {self.itos}")

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        numericalized_corpus = []

        for token in tokenized_text:
            if token in self.stoi:
                numericalized_corpus.append(self.stoi[token])
            else:
                numericalized_corpus.append(self.stoi["<UNK>"])

        return numericalized_corpus

class StrokeMRICapDataset(torch.utils.data.Dataset):
    def __init__(self, voxid_to_prep_vox_list, voxid_to_cap_list, freq_threshold=5, debug=False):
        self.voxid_to_prep_vox_paths_ = voxid_to_prep_vox_list
        self.voxid_to_cap_paths_ = voxid_to_cap_list
        self.debug = debug
        # self.tokenizer = get_tokenizer("basic_english")
        # self.torch_vocab = self.build_vocabulary()
        self.all_captions = self.get_all_captions()

        self.mri_vocab = StrokeMRIVocabulary(freq_threshold)
        self.mri_vocab.build_vocabulary(self.all_captions)

    def get_all_captions(self):
        captions_tokens_corpus = []
        # voxid_cap_map = {}

        for i in range(len(self.voxid_to_cap_paths_)):
            with open(self.voxid_to_cap_paths_[i], "rb") as file:
                voxid_to_prep_caps = pickle.load(file)

                voxel_id = voxid_to_prep_caps[0]
                prep_captions_str = voxid_to_prep_caps[1]
                # voxel_id = list(voxid_to_prep_caps.keys())[0]
                # prep_captions_str = list(voxid_to_prep_caps.values())[0]

                # voxid_cap_map[voxel_id] = prep_captions_str

                # if i == 0:
                #     print(f"prep_captions_str = {prep_captions_str}")

                # for caption in prep_captions_str:
                if i == 0:
                    print(f"prep_captions_str = {prep_captions_str}")

                captions_tokens_corpus.append(prep_captions_str)
        
        # print(f"captions_tokens_corpus = {captions_tokens_corpus}")
        return captions_tokens_corpus

    # def build_vocabulary(self):
    #     captions_tokens_corpus = self.get_all_captions()

    #     # print(f"captions_tokens_corpus = {captions_tokens_corpus}")

    #     vocab = build_vocab_from_iterator(captions_tokens_corpus, specials=["<unk>"], special_first=True)

    #     return vocab

    # def numericalize_captions(self, all_captions_per_voxel):
    #     print(f"all_captions_per_voxel = {all_captions_per_voxel}")
    #     # tokenize 
    #     # tokens_corpus = [sentence.split() for sentence in all_captions_per_voxel]
    #     # print(f"tokens_corpus = {tokens_corpus}")

    #     # Convert the tokenized corpus to numerical data
    #     numerical_corpus = [[self.torch_vocab[token] for token in sentence] for sentence in all_captions_per_voxel]

    #     # print(f"numerical_corpus = {numerical_corpus}")
    #     return numerical_corpus

    def __len__(self):
        return len(self.voxid_to_prep_vox_paths_)

    def __getitem__(self, idx):
        if self.debug:
            print("idx = {}".format(idx))

        # sitk to torch tensor dims (channels, depth, height, width)
        if self.debug:
            print("self.voxid_to_prep_vox_paths_[idx] = {}".format(self.voxid_to_prep_vox_paths_[idx]))
        with open(self.voxid_to_prep_vox_paths_[idx], "rb") as file:
            pickle_voxid_to_prep_vox = pickle.load(file)
        
        voxel_id_from_prepvox = pickle_voxid_to_prep_vox[0]
        prep_voxel_filepath = pickle_voxid_to_prep_vox[1]
        # print(f"voxel_id_from_prepvox = {voxel_id_from_prepvox}")
        # print(f"prep_voxel_filepath = {prep_voxel_filepath}")

        prep_voxel = sitk.ReadImage(prep_voxel_filepath)
        prep_voxel_array = sitk.GetArrayFromImage(prep_voxel)
        prep_voxel_tensor = torch.tensor(prep_voxel_array).float()
        # print("prep_voxel_tensor shape = {}".format(prep_voxel_tensor.shape))
        prep_voxel_tensor = prep_voxel_tensor.unsqueeze(0)
        # print("prep_voxel_tensor shape = {}".format(prep_voxel_tensor.shape))

        if self.debug:
            print("self.voxid_to_cap_paths_[idx] = {}".format(self.voxid_to_cap_paths_[idx]))

        with open(self.voxid_to_cap_paths_[idx], "rb") as file:
            pickle_voxid_to_prepcaps = pickle.load(file)

        voxel_id_from_caps = pickle_voxid_to_prepcaps[0]
        prep_captions_str = pickle_voxid_to_prepcaps[1]

        # voxel_id_from_caps = list(pickle_voxid_to_prepcaps.keys())[0]
        # TODO (JG): numericalize captions
        # prep_captions_str = list(pickle_voxid_to_prepcaps.values())[0]
        # numerical_vocab, vocab_size, numerical_corpus = self.numericalize_captions(prep_captions_str)
        
        numericalized_caption = [self.mri_vocab.stoi["<SOS>"]]
        # TODO (JG): Ensure there is one caption per voxel ID; For ex, could be 5 captions per voxel id, but need to separately iterate through them
        numericalized_caption = self.mri_vocab.numericalize(prep_captions_str)

        numericalized_caption.append(self.mri_vocab.stoi["<EOS>"])

        # print(f"voxel_id_from_caps = {voxel_id_from_caps}")
        # print(f"prep_captions_str = {prep_captions_str}")
        # print(f"numericalized_caption = {numericalized_caption}")

        numericalized_caption_tensor = torch.tensor(numericalized_caption)

        # print(f"numericalized_caption_tensor.shape = {numericalized_caption_tensor.shape}")

        # if voxel_id_from_caps in voxel_id_from_prepvox:
        #     print("Voxel ID from UNet3D ext features matches Voxel ID from prep captions")
        # else:
        #     print("WARNING: Voxel ID from UNet3D ext features unequal to Voxel ID from prep captions")


        return prep_voxel_tensor, numericalized_caption_tensor

class MRICapCollate:
    def __init__(self, mri_pad_idx):
        self.pad_idx = mri_pad_idx

    def __call__(self, batch):
        voxels = []
        count = 1
        for voxel in batch:
            # if count == 1:
                # print(f"voxel[0].unsqueeze(0) = {voxel[0].unsqueeze(0)}")
                # print(f"voxel[0].unsqueeze(0).shape = {voxel[0].unsqueeze(0).shape}")
                # print(f"voxel[0] = {voxel[0]}")
                # print(f"voxel[0].shape = {voxel[0].shape}")
            voxels.append(voxel[0].unsqueeze(0))
            count += 1
        
        voxels = torch.cat(voxels, dim=0)
        # print(f"voxels.shape = {voxels.shape}")
        # voxels = voxels.unsqueeze(0)
        # print(f"voxel.unsqueeze(0).shape = {voxel.unsqueeze(0).shape}")
        numericalized_cap_labels = [label[1] for label in batch]
        numericalized_cap_labels = rnn.pad_sequence(numericalized_cap_labels, batch_first = False, padding_value=self.pad_idx)
        # print(f"numericalized_cap_labels.shape = {numericalized_cap_labels.shape}")
        return voxels, numericalized_cap_labels
