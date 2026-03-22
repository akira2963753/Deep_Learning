import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """兩層 Conv2d(3×3, no padding) + ReLU（原始 UNet 論文架構）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d(2×2) + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """ConvTranspose2d(2×2) upsample + center crop skip + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def _center_crop(self, skip, target_h, target_w):
        """將 encoder 的 skip connection 裁切到與 decoder 相同的空間大小"""
        h, w = skip.shape[2], skip.shape[3]
        x1 = (h - target_h) // 2
        y1 = (w - target_w) // 2
        return skip[:, :, x1:x1 + target_h, y1:y1 + target_w]

    def forward(self, x, skip):
        x    = self.up(x)
        skip = self._center_crop(skip, x.shape[2], x.shape[3])
        x    = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    原始 UNet 架構（無 padding，copy and crop skip connections）。

    Encoder: 3 → 64 → 128 → 256 → 512 → 1024（bottleneck）
    Decoder: 1024 → 512 → 256 → 128 → 64
    Output:  64 → out_channels，最後 bilinear upsample 回原始輸入尺寸

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
        input_size = x.shape[2:]

        # Encoder（保留 skip connections）
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder（center crop skip connections）
        x = self.dec4(b, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        x = self.out_conv(x)

        # Bilinear upsample 回原始輸入尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 384, 384)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 1, 384, 384), f"Output shape error: {out.shape}"
    print("UNet shape 驗證通過！")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
