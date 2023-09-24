import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import monai

# Reference perplexity.ai for pytorch skull strip seg model
# https://www.perplexity.ai/search/cf74b6d5-9888-462b-9063-e90859bbf389

# Reference perplexity.ai for pytorch 3D UNet skull strip seg model
# https://www.perplexity.ai/search/0df235a1-27ba-4b67-bf7b-89c2500685c7?s=u

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

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2))
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
# UNet3D doesnt work right now because of input to output size issues
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super(UNet3D, self).__init__()
        self.conv1 = DoubleConv(in_channels=in_channels, out_channels=init_features)
        self.down1 = Down(in_channels=init_features, out_channels=init_features * 2)
        self.down2 = Down(in_channels=init_features * 2, out_channels=init_features * 4)
        self.down3 = Down(in_channels=init_features * 4, out_channels=init_features * 8)
        self.down4 = Down(in_channels=init_features * 8, out_channels=init_features * 16)
        self.up1 = Up(in_channels=init_features * 16, out_channels=init_features * 8)
        self.up2 = Up(in_channels=init_features * 8, out_channels=init_features * 4)
        self.up3 = Up(in_channels=init_features * 4, out_channels=init_features * 2)
        self.up4 = Up(in_channels=init_features * 2, out_channels=init_features)
        self.conv2 = nn.Conv3d(in_channels=init_features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # Reshape input tensor to remove the extra dimension: for ex, [1, 64, 96, 128, 160, 1] now [64, 96, 128, 160, 1]
        # x = x.squeeze(0)
        print("input: x.shape = {}".format(x.shape))
        x1 = self.conv1(x)
        print("out conv1: x1.shape = {}".format(x1.shape))
        x2 = self.down1(x1)
        print("out down1: x2.shape = {}".format(x2.shape))
        x3 = self.down1(x2)
        print("out down1: x3.shape = {}".format(x3.shape))
        x4 = self.down1(x3)
        print("out down1: x4.shape = {}".format(x4.shape))
        x5 = self.down1(x4)
        print("out down1: x5.shape = {}".format(x5.shape))
        x = self.up1(x5, x4)
        print("out up1: x.shape = {}".format(x.shape))
        x = self.up1(x4, x3)
        print("out up1: x.shape = {}".format(x.shape))
        x = self.up1(x3, x2)
        print("out up1: x.shape = {}".format(x.shape))
        x = self.up1(x2, x1)
        print("out up1: x.shape = {}".format(x.shape))
        x = self.conv2(x)
        print("out up1: x.shape = {}".format(x1.shape))
        return x


# NOTE: SimpleUNet3D works now on some test data
# Referenced Aladdin's 2D UNet PyTorch tutorial video and adapted it for my 3D UNet:
    # "PyTorch Image Segmentation Tutorial with U-NET: everything from scratch baby"
    # https://youtu.be/IHq1t7NxS8k?si=OoExNDXHms8J3rB2
class SimpleUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_features=[64, 128, 256, 512]):
        super(SimpleUNet3D, self).__init__()

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
                print("x.shape = {}".format(x.shape))
                print("skip_connection.shape = {}".format(skip_connection.shape))
                print("skip_connection.shape[2:] = {}".format(skip_connection.shape[2:]))
                # x = TF.resize(x, size=skip_connection.shape[2:])
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="trilinear", align_corners=False)
                # monai_resize3d = monai.transforms.Resize(spatial_size=skip_connection.shape[2:])
                # x = monai_resize3d(x)

            print("Concatenating skip connections")
            # Concatenate skip connections and add them along channel dimensions (ex: we have batch, channel, h, w)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # Then run through DoubleConv
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

if __name__ == "__main__":
    test()
