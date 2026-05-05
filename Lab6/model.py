import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int = 256) -> None:
        super().__init__()
        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        # pre-compute frequencies as a buffer so they aren't recreated every forward call
        half = emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half) / (half - 1)
        )
        self.register_buffer('freqs', freqs)  # (half,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        args = t[:, None].float() * self.freqs[None, :]                # (B, half)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, emb_dim)
        return self.mlp(emb)                                            # (B, emb_dim)


class LabelEmbedding(nn.Module):
    def __init__(self, num_classes: int = 24, emb_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, label: torch.Tensor) -> torch.Tensor:
        # label: (B, 24)
        return self.mlp(label)  # (B, emb_dim)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int = 256,
        dropout: float = 0.1,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # adaGN: projects condition to scale and shift for out_channels
        self.cond_proj = nn.Linear(emb_dim, out_channels * 2)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, in_C, H, W)  cond: (B, emb_dim)
        residual = self.skip_conv(x)

        h = F.silu(self.norm1(x))
        h = self.conv1(h)                                   # (B, out_C, H, W)

        scale_shift = self.cond_proj(cond)                  # (B, 2*out_C)
        scale, shift = scale_shift.chunk(2, dim=1)          # each (B, out_C)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h) * (1 + scale) + shift            # adaGN
        h = self.dropout(F.silu(h))
        h = self.conv2(h)

        return h + residual                                  # (B, out_C, H, W)


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        assert channels % num_heads == 0
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        residual = x

        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1).contiguous()  # (B, H*W, C)
        h, _ = self.attn(h, h, h)                                   # (B, H*W, C)
        h = h.permute(0, 2, 1).contiguous().reshape(B, C, H, W)    # (B, C, H, W)

        return h + residual


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int = 256,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResBlock(in_channels, out_channels, emb_dim, dropout)]
            + [
                ResBlock(out_channels, out_channels, emb_dim, dropout)
                for _ in range(num_res_blocks - 1)
            ]
        )
        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        h = x
        for res_block in self.res_blocks:
            h = res_block(h, cond)
            skips.append(h)
        return self.downsample(h), skips


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        emb_dim: int = 256,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        self.res_blocks = nn.ModuleList(
            [ResBlock(in_channels + skip_channels, out_channels, emb_dim, dropout)]
            + [
                ResBlock(out_channels, out_channels, emb_dim, dropout)
                for _ in range(num_res_blocks - 1)
            ]
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)                    # (B, in_C, H*2, W*2)
        x = torch.cat([x, skip], dim=1)         # (B, in_C + skip_C, H*2, W*2)
        for res_block in self.res_blocks:
            x = res_block(x, cond)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        emb_dim: int = 256,
        num_classes: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = [base_channels * m for m in channel_mults]  # [64,128,256,256]

        self.time_embed  = SinusoidalTimeEmbedding(emb_dim)
        self.label_embed = LabelEmbedding(num_classes, emb_dim)

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder: 64→128, 128→256, 256→256
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for out_ch in channels[1:]:
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, emb_dim, num_res_blocks, dropout)
            )
            in_ch = out_ch

        # Bottleneck at 8×8
        self.bottleneck = nn.ModuleList([
            ResBlock(channels[-1], channels[-1], emb_dim, dropout),
            SelfAttentionBlock(channels[-1], num_heads=4),
            ResBlock(channels[-1], channels[-1], emb_dim, dropout),
        ])

        # Decoder: (256,256→256), (256,256→128), (128,128→64)
        # skip_channels mirrors encoder output channels in reverse
        enc_channels = channels[1:]                    # [128, 256, 256]
        skip_ch_list = list(reversed(enc_channels))    # [256, 256, 128]
        out_ch_list  = list(reversed(channels[:-1]))   # [256, 128, 64]

        self.up_blocks = nn.ModuleList()
        in_ch = channels[-1]
        for skip_ch, out_ch in zip(skip_ch_list, out_ch_list):
            self.up_blocks.append(
                UpBlock(in_ch, skip_ch, out_ch, emb_dim, num_res_blocks, dropout)
            )
            in_ch = out_ch

        self.final_norm = nn.GroupNorm(num_groups=32, num_channels=base_channels)
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(
        self,
        x: torch.Tensor,          # (B, 3, 64, 64)
        t: torch.Tensor,          # (B,)
        condition: torch.Tensor,  # (B, 24)
    ) -> torch.Tensor:
        cond = self.time_embed(t) + self.label_embed(condition)  # (B, 256)

        h = self.init_conv(x)      # (B, 64, 64, 64)
        all_skips: List[torch.Tensor] = []

        for down_block in self.down_blocks:
            h, skips = down_block(h, cond)
            all_skips.append(skips[-1])  # one skip per level (closest to bottleneck)

        # Bottleneck
        h = self.bottleneck[0](h, cond)
        h = self.bottleneck[1](h)
        h = self.bottleneck[2](h, cond)

        for up_block in self.up_blocks:
            skip = all_skips.pop()
            h = up_block(h, skip, cond)

        h = F.silu(self.final_norm(h))
        return self.final_conv(h)   # (B, 3, 64, 64)


if __name__ == '__main__':
    model = UNet()
    model.eval()

    x    = torch.randn(2, 3, 64, 64)
    t    = torch.randint(1, 1001, (2,))
    cond = torch.zeros(2, 24)
    cond[0, 0] = 1.0
    cond[1, 1] = 1.0

    with torch.no_grad():
        out = model(x, t, cond)

    assert out.shape == (2, 3, 64, 64), f"Unexpected shape: {out.shape}"
    total = sum(p.numel() for p in model.parameters())
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {total:,}")
