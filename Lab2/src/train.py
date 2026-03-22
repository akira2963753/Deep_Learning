import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# 將 src/ 加入 path，方便 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.oxford_pet import OxfordPetDataset, _get_splits
from src.utils import combined_loss, dice_score, get_train_transform, get_val_transform, JointTransform
from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# 支援 JointTransform 的 Dataset wrapper
# ---------------------------------------------------------------------------

class AugmentedPetDataset(torch.utils.data.Dataset):
    """
    訓練集專用：將 JointTransform 同時套用在 image 和 mask 上。
    """

    def __init__(self, root, split, joint_transform: JointTransform, splits_dir=None):
        from pathlib import Path
        if splits_dir is not None:
            from src.oxford_pet import _get_kaggle_splits
            splits = _get_kaggle_splits(splits_dir)
        else:
            splits = _get_splits(root)
        self.image_names    = splits[split]
        self.images_dir     = Path(root) / "images"
        self.masks_dir      = Path(root) / "annotations" / "trimaps"
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        image = Image.open(self.images_dir / f"{name}.jpg").convert("RGB")
        trimap = np.array(Image.open(self.masks_dir / f"{name}.png"))
        binary = (trimap == 1).astype(np.uint8)
        mask   = Image.fromarray(binary)
        return self.joint_transform(image, mask)


# ---------------------------------------------------------------------------
# 訓練主程式
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    # 建立模型
    if args.model == "unet":
        from src.models.unet import UNet
        model = UNet(in_channels=3, out_channels=1).to(device)
        save_path = os.path.join(args.save_dir, "unet_best.pth")
    elif args.model == "resnet34_unet":
        from src.models.resnet34_unet import ResNet34UNet
        model = ResNet34UNet(out_channels=1).to(device)
        save_path = os.path.join(args.save_dir, "resnet34_unet_best.pth")
    else:
        raise ValueError(f"未知模型：{args.model}")

    os.makedirs(args.save_dir, exist_ok=True)

    last_path = os.path.join(args.save_dir, f"{args.model}_last.pth")

    # 資料集
    train_dataset = AugmentedPetDataset(args.data_root, "train", get_train_transform(), splits_dir=args.splits_dir)
    img_tf, mask_tf = get_val_transform()
    val_dataset = OxfordPetDataset(args.data_root, "val", transform=img_tf, target_transform=mask_tf, splits_dir=args.splits_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"訓練集：{len(train_dataset)} 筆  驗證集：{len(val_dataset)} 筆")

    # 優化器 & 排程器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_dice  = 0.0
    start_epoch = 1

    # Resume
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt["best_dice"]
        print(f"Resume 從 epoch {start_epoch}，目前最佳 val_dice={best_dice:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        # ── 訓練 ──
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks  = masks.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(images)
                preds = F.interpolate(preds, size=masks.shape[2:], mode='bilinear', align_corners=False)
                loss  = combined_loss(preds, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # ── 驗證 ──
        model.eval()
        val_loss  = 0.0
        val_dice  = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks  = masks.to(device).float()

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    preds     = model(images)
                    preds     = F.interpolate(preds, size=masks.shape[2:], mode='bilinear', align_corners=False)
                    val_loss += combined_loss(preds, masks).item() * images.size(0)
                val_dice += dice_score(preds, masks.float()) * images.size(0)

        val_loss /= len(val_dataset)
        val_dice /= len(val_dataset)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_dice: {val_dice:.4f}"
        )

        # 儲存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ 已儲存最佳模型（val_dice={best_dice:.4f}）→ {save_path}")

        scheduler.step(val_dice)

        # 儲存 last checkpoint（供 resume 使用）
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict(),
            "best_dice": best_dice,
        }, last_path)

    print(f"\n訓練結束。最佳 val Dice Score: {best_dice:.4f}")
    print(f"模型已儲存至：{save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet or ResNet34-UNet")
    parser.add_argument("--model",       type=str, default="unet",
                        choices=["unet", "resnet34_unet"])
    parser.add_argument("--data_root",   type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--save_dir",    type=str, default="saved_models")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--splits_dir",  type=str, default=None,
                        help="Kaggle 提供的 split 目錄（含 train.txt / val.txt）")
    parser.add_argument("--resume",      action="store_true",
                        help="從 last checkpoint 繼續訓練")
    args = parser.parse_args()

    train(args)
