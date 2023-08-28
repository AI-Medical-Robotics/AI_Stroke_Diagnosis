import torch
import torch.nn as nn
# import statistics
# import torchvision.models as models

# Reference perplexity.ai:
# https://www.perplexity.ai/search/01326834-6a79-43be-9e49-c040e0c02b66

# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size, train_CNN=False):
#         super(EncoderCNN, self).__init__()
#         self.train_CNN = train_CNN
#         self.inception = models.inception_v3(pretrained=True, aux_logits=False)
#         self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
#         self.relu = nn.ReLU()
#         self.times = []
#         self.dropout = nn.Dropout(0.5)

#         def forward(self, images):
#             features = self.inception(images)

#             # Included following code in train.py. Which is more appropriate?
#             for name, param in self.inception.named_parameters():
#                 if "fc.weight" in name or "fc.bias" in name:
#                     param.requires_grad = True
#                 else:
#                     param.requires_grad = self.train_CNN

#             return self.dropout(self.relu(features))

# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
#         super(DecoderRNN, self).__init()
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
#         self.linear = nn.Linear(hidden_size, vocab_size)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, features, captions):
#         embeddings = self.dropout(self.embed(captions))
#         embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
#         hiddens, _ = self.lstm(embeddings)
#         outputs = self.linear(hiddens)
#         return outputs

class CNNtoLSTM(nn.Module):
    def __init__(self, caps_max_len, vocab_size):
        super(CNNtoLSTM, self).__init__()
        
        # Encoder model - image feature layers
        self.fe1 = nn.Dropout(0.4)
        self.fe2 = nn.Linear(4096, 256)

        # Encoder model - sequence feature layers
        self.se1 = nn.Embedding(vocab_size, 256, padding_idx=0)
        self.se2 = nn.Dropout(0.4)
        self.se3 = nn.LSTM(256, 256)

        # Decoder model
        self.decoder1 = nn.Linear(256, 256)
        self.decoder2 = nn.Linear(256, vocab_size)

    def forward(self, inputs1, inputs2):
        fe1 = self.fe1(inputs1)
        fe2 = self.fe2(fe1)

        se1 = self.se1(inputs2)
        se2 = self.se2(se1)
        se3, _ = self.se3(se2)

        decoder1 = torch.add(fe2, se3)
        decoder2 = self.decoder1(decoder1)
        outputs = self.decoder2(decoder2)
        return outputs
