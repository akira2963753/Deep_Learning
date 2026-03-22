"""
infer_localtest.py — 在本機對 test set 計算 Dice Score，避免浪費 Kaggle 提交次數。

Oxford-IIIT Pet 是公開資料集，test set 的 GT mask 在本地即可取得。

執行方式：
    # 掃描最佳 threshold
    python src/infer_localtest.py \
        --model unet \
        --data_root dataset/oxford-iiit-pet \
        --checkpoint saved_models/unet_best.pth \
        --test_list nycu-2026-spring-dl-lab2-unet/test.txt \
        --tta

    # 指定固定 threshold
    python src/infer_localtest.py \
        --model resnet34_unet \
        --checkpoint saved_models/resnet34_unet_best.pth \
        --test_list nycu-2026-spring-dl-lab2-unet/test.txt \
        --threshold 0.45
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_val_transform, IMAGE_SIZE


# ---------------------------------------------------------------------------
# Test Dataset（含 GT mask）
# ---------------------------------------------------------------------------

class LocalTestDataset(Dataset):
    """讀取 test.txt 並載入圖片與 GT mask（trimap → binary）。"""

    def __init__(self, data_root, test_list_path, transform):
        self.images_dir = Path(data_root) / "images"
        self.masks_dir  = Path(data_root) / "annotations" / "trimaps"
        self.transform  = transform

        with open(test_list_path, "r") as f:
            self.names = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        image  = Image.open(self.images_dir / f"{name}.jpg").convert("RGB")
        trimap = np.array(Image.open(self.masks_dir / f"{name}.png"))
        binary = (trimap == 1).astype(np.uint8)   # 1=foreground, 2,3=background

        if self.transform:
            image = self.transform(image)

        # mask resize → tensor (1, H, W)
        mask_pil = Image.fromarray(binary)
        from torchvision import transforms
        mask_pil = mask_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask_pil)).unsqueeze(0).float()

        return image, mask


# ---------------------------------------------------------------------------
# Dice Score
# ---------------------------------------------------------------------------

def compute_dice(pred_binary: torch.Tensor, target: torch.Tensor, smooth=1e-6) -> float:
    """
    Dice = 2 × |Pred ∩ GT| / (|Pred| + |GT|)
    pred_binary: (B, 1, H, W) bool/float  已二值化
    target:      (B, 1, H, W) float
    """
    pred   = pred_binary.contiguous().view(-1).float()
    target = target.contiguous().view(-1).float()
    intersection = (pred * target).sum()
    return ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


# ---------------------------------------------------------------------------
# 推論主程式
# ---------------------------------------------------------------------------

def run(args):
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
    print(f"載入模型：{args.model}  checkpoint：{args.checkpoint}")

    # 資料集
    img_tf, _ = get_val_transform()
    dataset    = LocalTestDataset(args.data_root, args.test_list, img_tf)
    loader     = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"測試集：{len(dataset)} 筆\n")

    # 收集所有圖片的 sigmoid 輸出（在 threshold 掃描前先全部算完）
    all_probs  = []   # list of (1, H, W) tensors
    all_masks  = []   # list of (1, H, W) tensors

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)

            def _forward(imgs):
                logits = model(imgs)
                logits = F.interpolate(logits, size=(IMAGE_SIZE, IMAGE_SIZE),
                                       mode='bilinear', align_corners=False)
                return torch.sigmoid(logits)

            if args.tta:
                p0 = _forward(images)
                p1 = torch.flip(_forward(torch.flip(images, [3])), [3])
                p2 = torch.flip(_forward(torch.flip(images, [2])), [2])
                p3 = torch.flip(_forward(torch.flip(images, [2, 3])), [2, 3])
                probs = (p0 + p1 + p2 + p3) / 4.0
            else:
                probs = _forward(images)

            all_probs.append(probs.cpu())
            all_masks.append(masks)

    all_probs = torch.cat(all_probs, dim=0)   # (N, 1, H, W)
    all_masks = torch.cat(all_masks, dim=0)   # (N, 1, H, W)

    # threshold 掃描或固定 threshold
    if args.threshold is not None:
        thresholds = [args.threshold]
    else:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.71, 0.05)]

    print("掃描 Threshold...")
    best_dice, best_thr = 0.0, 0.5

    for thr in thresholds:
        preds_binary = (all_probs > thr).float()
        total_dice   = 0.0
        for i in range(len(all_probs)):
            total_dice += compute_dice(preds_binary[i:i+1], all_masks[i:i+1])
        avg_dice = total_dice / len(all_probs)

        marker = ""
        if avg_dice > best_dice:
            best_dice, best_thr = avg_dice, thr
            marker = "  ← best"
        print(f"  threshold={thr:.2f}  Dice={avg_dice:.4f}{marker}")

    print(f"\n最佳 threshold: {best_thr:.2f}  →  Dice Score: {best_dice:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local test-set Dice evaluation")
    parser.add_argument("--model",       type=str, default="unet",
                        choices=["unet", "resnet34_unet"])
    parser.add_argument("--data_root",   type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--test_list",   type=str, required=True,
                        help="test.txt 路徑（每行一個圖片名稱）")
    parser.add_argument("--threshold",   type=float, default=None,
                        help="固定 threshold（省略則掃描 0.30~0.70）")
    parser.add_argument("--tta",         action="store_true",
                        help="啟用 Test Time Augmentation")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    run(args)
