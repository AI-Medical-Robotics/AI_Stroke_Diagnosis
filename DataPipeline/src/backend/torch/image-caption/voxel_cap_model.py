import torch
import torch.nn as nn
import statistics
import torchvision.models as models
# TODO (JG): Fix UNet3D to RNN forward missing problem coming from UNet3D part.

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)



class SimpleUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_features=[64, 128, 256, 512], debug=False):
        super(SimpleUNet3D, self).__init__()
        self.debug = debug

        self.upsampling3d = nn.ModuleList()
        self.downsampling3d = nn.ModuleList()

        self.pool3d = nn.MaxPool3d(kernel_size=2, stride=2)

        # Downsampling encoder part of UNET
        for feature in hidden_features:
            self.downsampling3d.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Upsampling decoder part of UNET
        for feature in reversed(hidden_features):
            # we do ConvTranspose2d for up, then DoubleConv for 2 convs
            self.upsampling3d.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.upsampling3d.append(DoubleConv(feature*2, feature))

        self.bottleneck3d = DoubleConv(hidden_features[-1], hidden_features[-1]*2)

        self.final_conv3d = nn.Conv3d(hidden_features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections3d = []

        # we loop for downsampling
        for down3d in self.downsampling3d:
            x = down3d(x)
            # for skip connections, ordering is important here, first is highest resolution, last is smallest resolution
            skip_connections3d.append(x)
            x = self.pool3d(x)

        x = self.bottleneck3d(x)

        # now go backwards to make skip connections easier, reverse list, go with skip connections with highest res to lowest res
        skip_connections3d = skip_connections3d[::-1]

        # we loop taking steps of 2 for upsampling with DoubleConv
        for idx in range(0, len(self.upsampling3d), 2):
            # upsample with ConvTranspose2d
            x = self.upsampling3d[idx](x)
            # Divide idx by 2 to get just idx since we're doing step by 2 above
            skip_connection = skip_connections3d[idx//2]

            # Soln to cases when not divisible by 2 issue
                # NOTE: in the UNET paper, they did cropping, but we'll do resizing for this issue
            if x.shape != skip_connection.shape:
                # we check x from outward part of upsampling, ifnequal resize our x using skip_connection resolutions just by channels, ignore h & w
                if self.debug:
                    print("x.shape = {}".format(x.shape))
                    print("skip_connection.shape = {}".format(skip_connection.shape))
                    print("skip_connection.shape[2:] = {}".format(skip_connection.shape[2:]))
                # x = TF.resize(x, size=skip_connection.shape[2:])
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="trilinear", align_corners=False)
                # monai_resize3d = monai.transforms.Resize(spatial_size=skip_connection.shape[2:])
                # x = monai_resize3d(x)

            if self.debug:
                print("Concatenating skip connections")
            # Concatenate skip connections and add them along channel dimensions (ex: we have batch, channel, h, w)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # Then run through DoubleConv
            x = self.upsampling3d[idx+1](concat_skip)

        return self.final_conv3d(x)



class SimpleCNN3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_features=[64, 128, 256, 512], debug=False):
        super(SimpleCNN3D, self).__init__()
        self.debug = debug

        self.downsampling3d = nn.ModuleList()

        self.pool3d = nn.MaxPool3d(kernel_size=2, stride=2)

        for feature in hidden_features:
            self.downsampling3d.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.final_conv3d = nn.Conv3d(hidden_features[-1], out_channels, kernel_size=1)


    def forward(self, x):
        # we loop for downsampling
        for down3d in self.downsampling3d:
            x = down3d(x)
            # for skip connections, ordering is important here, first is highest resolution, last is smallest resolution
            x = self.pool3d(x)

        return self.final_conv3d(x)




class EncoderCNN3D(nn.Module):
    def __init__(self, embed_size, hidden_size, train_CNN=False):
        super(EncoderCNN3D, self).__init__()
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_CNN = train_CNN
        self.embed_size = embed_size
        # self.resnet_orig = models.resnet50(pretrained=True)
        # self.unet3d = SimpleUNet3D(in_channels=1, out_channels=embed_size)
        self.cnn3d = SimpleCNN3D(in_channels=1, out_channels=embed_size)
        self.final_conv3d_in_chs = self.cnn3d.final_conv3d.in_channels

        # Get all layers except last one, then change last one
        # layers = list(self.unet3d.children())[:-1]
        # self.unet3d = nn.Sequential(*layers)
        # print("self.final_conv3d_in_chs = {}".format(self.final_conv3d_in_chs))
        # self.conv3d_embed = nn.Conv3d(self.final_conv3d_in_chs, embed_size, kernel_size=1)

        # self.fc_embed = nn.Linear(self.final_conv3d_in_chs, embed_size)

        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # print("EncoderCNN3D forward:")
        # print("images.shape = {}".format(images.shape))
        features = self.cnn3d(images)
        # print("after CNN3D: features.shape = {}".format(features.shape))
        features = features.view(features.size(0), -1)
        # print("after view: features.shape = {}".format(features.shape))
        fc_embed = nn.Linear(in_features=features.shape[-1], out_features=self.embed_size).to(self.DEVICE)
        features = fc_embed(features)
        # print("after nn.Linear: features.shape = {}".format(features.shape))
        features = self.relu(features)
        # print("after relu: features.shape = {}".format(features.shape))

        # Included following code in train.py. Which is more appropriate?
        for name, param in self.cnn3d.named_parameters():
            # if "fc.weight" in name or "fc.bias" in name:
            #     param.requires_grad = True
            # else:
            param.requires_grad = self.train_CNN

        return self.dropout(features)

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        # print(f"after embed: embeddings.shape = {embeddings.shape}")
        embeddings = self.dropout(embeddings)
        # print(f"features.shape = {features.shape}")
        # print(f"features.unsqueeze(0).shape = {features.unsqueeze(0).shape}")
        # print(f"after dropout: embeddings.shape = {embeddings.shape}")
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class CNN3DtoLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNN3DtoLSTM, self).__init__()
        self.encoderCNN3D = EncoderCNN3D(embed_size, hidden_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        # print("CNN3DtoLSTM: images.shape = {}".format(images.shape))
        # print("CNN3DtoLSTM: captions.shape = {}".format(captions.shape))
        features = self.encoderCNN3D(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, gt_caption=None, max_length=50, keep_eos=True):
        result_caption = []
        image_caption_gt = []

        with torch.no_grad():
            x = self.encoderCNN3D(image).unsqueeze(0)
            # print(f"x.shape = {x.shape}")
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                # print(f"after decoderRNN: hiddens.shape = {hiddens.shape}")
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                # print(f"after decoderRNN linear: output.shape = {output.shape}")
                predicted = output.argmax(1)
                # print(f"after argmax(1) pred = {predicted}")

                if not keep_eos:
                    if vocabulary.itos[predicted.item()] == "<EOS>":
                        break

                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)
                # print(f"after decoderRNN embed: x.shape = {x.shape}")
        
                if keep_eos:
                    if vocabulary.itos[predicted.item()] == "<EOS>":
                        break

        image_caption_pred = [vocabulary.itos[idx] for idx in result_caption]
        if gt_caption is None:
            return image_caption_pred
        
        # print(f"vocabulary.itos = {vocabulary.itos}")
        # print(f"gt_caption.tolist() = {gt_caption.tolist()}")

        for gt_idx in gt_caption.tolist():
            # print(f"gt_idx[0] = {gt_idx[0]}")
            # print(f"vocabulary.itos[gt_idx[0]] = {vocabulary.itos[gt_idx[0]]}")
            if not keep_eos:
                if vocabulary.itos[gt_idx[0]] == "<EOS>":
                    break
            image_caption_gt.append(vocabulary.itos[gt_idx[0]])


        return image_caption_pred, image_caption_gt
