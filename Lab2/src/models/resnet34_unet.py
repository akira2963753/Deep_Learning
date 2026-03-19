import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import DoubleConv


# ---------------------------------------------------------------------------
# ResNet34 基本單元
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """
    ResNet34 的基本殘差塊。
    兩層 Conv(3×3) + BN，含 shortcut connection。
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        # shortcut：若 stride≠1 或 channels 不同，用 Conv1×1 + BN 對齊
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
    """建立一個 ResNet layer（多個 BasicBlock 串接）。"""
    layers = [BasicBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Decoder Block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    UNet 風格的 Decoder Block。
    ConvTranspose2d upsample → concat skip → DoubleConv
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels,
                                       kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # 處理尺寸不整除造成的 off-by-one
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# ResNet34-UNet
# ---------------------------------------------------------------------------

class ResNet34UNet(nn.Module):
    """
    ResNet34 作為 Encoder，UNet 風格的 Decoder。
    從頭訓練，不使用任何預訓練權重。

    以 384×384 輸入為例：
        Encoder:
            conv1 (7×7, stride=2) → 64 × 192 × 192   (skip4)
            maxpool (stride=2)    → 64 × 96  × 96
            layer1 (×3)           → 64 × 96  × 96     (skip3)
            layer2 (×4, stride=2) → 128 × 48 × 48     (skip2)
            layer3 (×6, stride=2) → 256 × 24 × 24     (skip1)
            layer4 (×3, stride=2) → 512 × 12 × 12     (bottleneck)
        Decoder:
            512 + skip1(256) → 256 × 24  × 24
            256 + skip2(128) → 128 × 48  × 48
            128 + skip3(64)  →  64 × 96  × 96
             64 + skip4(64)  →  32 × 192 × 192
            bilinear ×2      →  32 × 384 × 384
            Conv1×1          →   1 × 384 × 384

    forward() 回傳 logits（不含 sigmoid）。
    """

    def __init__(self, out_channels=1):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1  = _make_layer(64,  64,  num_blocks=3, stride=1)
        self.layer2  = _make_layer(64,  128, num_blocks=4, stride=2)
        self.layer3  = _make_layer(128, 256, num_blocks=6, stride=2)
        self.layer4  = _make_layer(256, 512, num_blocks=3, stride=2)

        # ── Decoder ──────────────────────────────────────────
        self.dec1 = DecoderBlock(512, 256, 256)   # 12×12  → 24×24
        self.dec2 = DecoderBlock(256, 128, 128)   # 24×24  → 48×48
        self.dec3 = DecoderBlock(128, 64,  64)    # 48×48  → 96×96
        self.dec4 = DecoderBlock(64,  64,  32)    # 96×96  → 192×192

        # 最終 upsample ×2 + Conv1×1
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # ── Encoder ──
        skip4 = self.relu(self.bn1(self.conv1(x)))   # 64 × H/2 × W/2
        x     = self.maxpool(skip4)                   # 64 × H/4 × W/4

        skip3 = self.layer1(x)                        # 64  × H/4  × W/4
        skip2 = self.layer2(skip3)                    # 128 × H/8  × W/8
        skip1 = self.layer3(skip2)                    # 256 × H/16 × W/16
        x     = self.layer4(skip1)                    # 512 × H/32 × W/32

        # ── Decoder ──
        x = self.dec1(x, skip1)   # 256 × H/16 × W/16
        x = self.dec2(x, skip2)   # 128 × H/8  × W/8
        x = self.dec3(x, skip3)   #  64 × H/4  × W/4
        x = self.dec4(x, skip4)   #  32 × H/2  × W/2

        # 最終 upsample 回原始尺寸
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.out_conv(x)   # out_channels × H × W


if __name__ == "__main__":
    model = ResNet34UNet(out_channels=1)
    x = torch.randn(2, 3, 384, 384)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 1, 384, 384), f"Output shape error: {out.shape}"
    print("ResNet34UNet shape 驗證通過！")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
