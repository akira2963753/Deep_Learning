"""
inference.py — 對 test set 做推論並輸出 Kaggle 提交用 CSV。

執行方式：
    python src/inference.py \
        --model unet \
        --data_root dataset/oxford-iiit-pet \
        --checkpoint saved_models/unet_best.pth \
        --output unet_pred.csv

CSV 格式（Kaggle RLE）：
    id,predicted_mask
    Abyssinian_1,1 2 5 3 ...
    ...
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.oxford_pet import OxfordPetDataset
from src.utils import get_val_transform


# ---------------------------------------------------------------------------
# RLE encoding（Kaggle 常用格式：column-major / Fortran order）
# ---------------------------------------------------------------------------

def rle_encode(mask: np.ndarray) -> str:
    """
    將 binary mask (H×W, 值為 0/1) 轉成 RLE 字串。
    mask 以 column-major (Fortran) 順序展平後計算。
    回傳 "start1 length1 start2 length2 ..." 字串（1-indexed）。
    若 mask 全為 0，回傳空字串 ""。
    """
    mask = mask.flatten(order="F")  # column-major
    changes = np.diff(mask, prepend=0, append=0)
    starts  = np.where(changes == 1)[0] + 1   # 1-indexed
    ends    = np.where(changes == -1)[0] + 1   # exclusive end
    lengths = ends - starts

    if len(starts) == 0:
        return ""
    return " ".join(f"{s} {l}" for s, l in zip(starts, lengths))


# ---------------------------------------------------------------------------
# 推論主程式
# ---------------------------------------------------------------------------

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    # 建立模型
    if args.model == "unet":
        from src.models.unet import UNet
        model = UNet(in_channels=3, out_channels=1)
    elif args.model == "resnet34_unet":
        from src.models.resnet34_unet import ResNet34UNet
        model = ResNet34UNet(out_channels=1)
    else:
        raise ValueError(f"未知模型：{args.model}")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()
    print(f"已載入 checkpoint：{args.checkpoint}")

    # 測試集（只有 image，無 mask）
    img_tf, _ = get_val_transform()
    test_dataset = OxfordPetDataset(args.data_root, "test", transform=img_tf)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"測試集：{len(test_dataset)} 筆")

    rows = []
    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)
            preds  = model(images)                          # B×1×H×W logits
            preds  = torch.sigmoid(preds)
            preds  = (preds > args.threshold).cpu().numpy() # B×1×H×W bool

            for pred, name in zip(preds, names):
                mask_2d = pred[0].astype(np.uint8)          # H×W
                rle     = rle_encode(mask_2d)
                rows.append((name, rle))

    # 寫出 CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "predicted_mask"])
        writer.writerows(rows)

    print(f"✓ 已輸出 {len(rows)} 筆預測 → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on test set")
    parser.add_argument("--model",       type=str, default="unet",
                        choices=["unet", "resnet34_unet"])
    parser.add_argument("--data_root",   type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--output",      type=str, default="unet_pred.csv")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    run_inference(args)
