import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import monai

# Reference perplexity.ai for pytorch skull strip seg model
# https://www.perplexity.ai/search/cf74b6d5-9888-462b-9063-e90859bbf389

# Reference perplexity.ai for pytorch 3D UNet skull strip seg model
# https://www.perplexity.ai/search/0df235a1-27ba-4b67-bf7b-89c2500685c7?s=u

# Add SeLU instead of ReLU
class SESeLUBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SESeLUBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size = 1, bias = False)
        self.selu = nn.SELU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size = 1, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.selu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out

class SESeLUDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SESeLUDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.SELU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.SELU(inplace=True),
            SESeLUBlock(in_channels=out_channels)
        )

    def forward(self, x):
        return self.conv(x)

# Add attention gates using the Squeeze-and-Excitation (SE) block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size = 1, bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size = 1, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out

class SEDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(in_channels=out_channels)
        )

    def forward(self, x):
        return self.conv(x)

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

# NOTE: SimpleUNet3D works now on some test data
# Referenced Aladdin's 2D UNet PyTorch tutorial video and adapted it for my 3D UNet:
    # "PyTorch Image Segmentation Tutorial with U-NET: everything from scratch baby"
    # https://youtu.be/IHq1t7NxS8k?si=OoExNDXHms8J3rB2
# Last 2 epochs out of 5 epochs showed improved Dice Scores for SimpleUNet3D
    # Also didnt remove skull during preprocessing, but plan to later
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

class AttSEUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_features=[64, 128, 256, 512], debug=False):
        super(AttSEUNet3D, self).__init__()
        self.debug = debug

        self.upsampling3d = nn.ModuleList()
        self.downsampling3d = nn.ModuleList()

        self.pool3d = nn.MaxPool3d(kernel_size=2, stride=2)

        # Downsampling encoder part of UNET
        for feature in hidden_features:
            self.downsampling3d.append(SEDoubleConv(in_channels, feature))
            in_channels = feature

        # Upsampling decoder part of UNET
        for feature in reversed(hidden_features):
            # we do ConvTranspose2d for up, then SEDoubleConv for 2 convs
            self.upsampling3d.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.upsampling3d.append(SEDoubleConv(feature*2, feature))

        self.bottleneck3d = SEDoubleConv(hidden_features[-1], hidden_features[-1]*2)

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

        # we loop taking steps of 2 for upsampling with SEDoubleConv
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
            # Then run through SEDoubleConv
            x = self.upsampling3d[idx+1](concat_skip)

        return self.final_conv3d(x)


class AttSESeLUUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_features=[64, 128, 256, 512], debug=False):
        super(AttSESeLUUNet3D, self).__init__()
        self.debug = debug

        self.upsampling3d = nn.ModuleList()
        self.downsampling3d = nn.ModuleList()

        # self.avg_max_pool3d = nn.AdaptiveMaxPool3d(2)
        self.pool3d = nn.MaxPool3d(kernel_size=2, stride=2)

        # Downsampling encoder part of UNET
        for feature in hidden_features:
            self.downsampling3d.append(SESeLUDoubleConv(in_channels, feature))
            in_channels = feature

        # Upsampling decoder part of UNET
        for feature in reversed(hidden_features):
            # we do ConvTranspose2d for up, then SESeLUDoubleConv for 2 convs
            self.upsampling3d.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.upsampling3d.append(SESeLUDoubleConv(feature*2, feature))

        self.bottleneck3d = SESeLUDoubleConv(hidden_features[-1], hidden_features[-1]*2)

        self.final_conv3d = nn.Conv3d(hidden_features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections3d = []

        # we loop for downsampling
        for down3d in self.downsampling3d:
            x = down3d(x)
            # for skip connections, ordering is important here, first is highest resolution, last is smallest resolution
            skip_connections3d.append(x)
            x = self.avg_max_pool3d(x)

        x = self.bottleneck3d(x)

        # now go backwards to make skip connections easier, reverse list, go with skip connections with highest res to lowest res
        skip_connections3d = skip_connections3d[::-1]

        # we loop taking steps of 2 for upsampling with SESeLUDoubleConv
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
            # Then run through SESeLUDoubleConv
            x = self.upsampling3d[idx+1](concat_skip)

        return self.final_conv3d(x)


def test():
    # batch, chs, h, w, z
    x = torch.randn((3, 1, 161, 161, 161))
    model = SimpleUNet3D(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

def test_attunet3d():
    # batch, chs, h, w, z
    x = torch.randn((3, 1, 161, 161, 161))
    model = AttSEUNet3D(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    # test()
    test_attunet3d()
