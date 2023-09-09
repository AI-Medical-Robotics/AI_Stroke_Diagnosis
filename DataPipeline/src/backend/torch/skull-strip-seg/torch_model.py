import torch
import torch.nn as nn

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
            nn.MaxPool3d(2),
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

