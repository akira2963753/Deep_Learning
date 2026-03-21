import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """兩層 Conv2d(3×3) + ReLU（對應原始 UNet 論文架構）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d(2×2) + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """ConvTranspose2d(2×2) upsample + concat skip + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # 處理尺寸不整除造成的 off-by-one
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    標準 UNet 架構。

    Encoder: 3 → 64 → 128 → 256 → 512 → 1024（bottleneck）
    Decoder: 1024 → 512 → 256 → 128 → 64
    Output:  64 → out_channels（預設 1，binary segmentation）

    forward() 回傳 logits（shape: B × out_channels × H × W）
    不含 sigmoid，讓 BCEWithLogitsLoss 處理數值穩定性。
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = Down(64, 128)
        self.enc3 = Down(128, 256)
        self.enc4 = Down(256, 512)

        # Bottleneck
        self.bottleneck = Down(512, 1024)

        # Decoder
        self.dec4 = Up(1024, 512)
        self.dec3 = Up(512, 256)
        self.dec2 = Up(256, 128)
        self.dec1 = Up(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder（保留 skip connections）
        s1 = self.enc1(x)       # 64 × H × W
        s2 = self.enc2(s1)      # 128 × H/2 × W/2
        s3 = self.enc3(s2)      # 256 × H/4 × W/4
        s4 = self.enc4(s3)      # 512 × H/8 × W/8

        # Bottleneck
        b = self.bottleneck(s4) # 1024 × H/16 × W/16

        # Decoder
        x = self.dec4(b, s4)    # 512 × H/8 × W/8
        x = self.dec3(x, s3)    # 256 × H/4 × W/4
        x = self.dec2(x, s2)    # 128 × H/2 × W/2
        x = self.dec1(x, s1)    # 64 × H × W

        return self.out_conv(x) # out_channels × H × W


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 1, 256, 256), f"Output shape error: {out.shape}"
    print("UNet shape 驗證通過！")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
