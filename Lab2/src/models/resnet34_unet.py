import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """ Double (Conv + BN + ReLU) """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        # shortcut: stride!=1 或 channel 不同時用 1x1 conv 對齊
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


def _make_layer(in_channels, out_channels, num_blocks, stride=1):
    layers = [BasicBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class DecoderBlock(nn.Module):
    """ConvTranspose2d upsample + concat skip + DoubleConv"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels,
                                       kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # 處理尺寸不整除
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNet34UNet(nn.Module):

    def __init__(self, out_channels=1):
        super().__init__()

        # Encoder
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1  = _make_layer(64,  64,  num_blocks=3, stride=1)
        self.layer2  = _make_layer(64,  128, num_blocks=4, stride=2)
        self.layer3  = _make_layer(128, 256, num_blocks=6, stride=2)
        self.layer4  = _make_layer(256, 512, num_blocks=3, stride=2)

        # Decoder
        self.dec1 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec3 = DecoderBlock(128, 64,  64)
        self.dec4 = DecoderBlock(64,  64,  32)

        # Output
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s4 = self.relu(self.bn1(self.conv1(x)))
        x  = self.maxpool(s4)

        s3 = self.layer1(x)
        s2 = self.layer2(s3)
        s1 = self.layer3(s2)
        x  = self.layer4(s1)

        # Decoder
        x = self.dec1(x, s1)
        x = self.dec2(x, s2)
        x = self.dec3(x, s3)
        x = self.dec4(x, s4)

        # upsample 回原始大小
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.out_conv(x)
