import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


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
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

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


class Bottleneck(nn.Module):
    """concat layer3(256ch↓12²) + layer4(512ch@12²) → 32ch@12²"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(768, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, f3, f4):
        # f3: 24², f4: 12² → 將 f3 縮到 f4 的尺寸再 concat，輸出在 12²
        f3_down = F.interpolate(f3, size=f4.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([f3_down, f4], dim=1))


class DecoderBlock(nn.Module):
    """Upsample + concat skip + Conv+BN+ReLU + CBAM"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_channels, reduction=max(1, out_channels // 2))

    def forward(self, x, skip):
        x = self.up(x)
        # 處理尺寸不整除
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.cbam(self.conv(x))


class UpsampleBlock(nn.Module):
    """Upsample + Conv+BN+ReLU + CBAM（無 skip concat）"""

    def __init__(self, channels):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(channels, reduction=max(1, channels // 2))

    def forward(self, x):
        return self.cbam(self.conv(self.up(x)))


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

        # Bottleneck + Decoder
        self.bottleneck = Bottleneck()

        self.dec1 = DecoderBlock(32, 256, 32)  # skip=f3    (256ch, 24²)
        self.dec2 = DecoderBlock(32, 128, 32)  # skip=skip2 (128ch, 48²)
        self.dec3 = DecoderBlock(32, 64,  32)  # skip=skip1 (64ch,  96²)
        self.dec4 = UpsampleBlock(32)           # 96²→192², no skip
        self.dec5 = UpsampleBlock(32)           # 192²→384², no skip

        # Output
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x     = self.relu(self.bn1(self.conv1(x)))  # 192², 64ch
        x     = self.maxpool(x)                      # 96²,  64ch

        skip1 = self.layer1(x)      # 96²,  64ch  → dec3 skip
        skip2 = self.layer2(skip1)  # 48²,  128ch → dec2 skip
        f3    = self.layer3(skip2)  # 24²,  256ch → dec1 skip
        f4    = self.layer4(f3)     # 12²,  512ch → bottleneck

        # Bottleneck（輸出 12²）
        x = self.bottleneck(f3, f4)  # 12², 32ch

        # Decoder
        x = self.dec1(x, f3)    # 12²→24²,  skip=f3    (256ch)
        x = self.dec2(x, skip2)  # 24²→48²,  skip=skip2 (128ch)
        x = self.dec3(x, skip1)  # 48²→96²,  skip=skip1 (64ch)
        x = self.dec4(x)          # 96²→192², no skip
        x = self.dec5(x)          # 192²→384², no skip

        return self.out_conv(x)   # 384², 1ch
