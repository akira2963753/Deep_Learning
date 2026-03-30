import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.oxford_pet import OxfordPetDataset, _get_splits
from src.utils import combined_loss, dice_score, get_train_transform, get_val_transform, JointTransform, IMAGE_SIZE
from PIL import Image
import numpy as np


def compute_pad_size(input_size, model):
    """算出 reflection padding 的大小，ResNet34-UNet 不需要所以回傳 0"""
    dummy = torch.zeros(1, 3, input_size, input_size)
    with torch.no_grad():
        out = model(dummy)
    out_size = out.shape[2]
    if out_size >= input_size:
        return 0
    return (input_size - out_size + 1) // 2


def pad_and_crop(images, model, pad, image_size):
    """pad -> forward -> crop 回原始大小，pad=0 時直接 forward"""
    if pad > 0:
        images = F.pad(images, [pad, pad, pad, pad], mode='reflect')
    preds = model(images)
    if pad > 0:
        oh, ow = preds.shape[2], preds.shape[3]
        y1 = (oh - image_size) // 2
        x1 = (ow - image_size) // 2
        preds = preds[:, :, y1:y1 + image_size, x1:x1 + image_size]
    return preds


class AugmentedPetDataset(torch.utils.data.Dataset):
    """訓練用，對 image 和 mask 同時做 augmentation"""

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


def train(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 建立模型
    if args.model == "unet":
        from src.models.unet import UNet
        model = UNet(in_channels=3, out_channels=1)
        save_path = os.path.join(args.save_dir, "unet_best.pth")
    elif args.model == "resnet34_unet":
        from src.models.resnet34_unet import ResNet34UNet
        model = ResNet34UNet(out_channels=1)
        save_path = os.path.join(args.save_dir, "resnet34_unet_best.pth")
    else:
        raise ValueError(f"Unknown model: {args.model}")

    pad = compute_pad_size(IMAGE_SIZE, model)
    print(f"Reflection padding: {pad} px per side")
    model.to(device)

    os.makedirs(args.save_dir, exist_ok=True)
    last_path = os.path.join(args.save_dir, f"{args.model}_last.pth")

    # 資料集
    elastic_p = 0.5 if args.model == "resnet34_unet" else 0.0
    train_dataset = AugmentedPetDataset(args.data_root, "train", get_train_transform(elastic_p=elastic_p), splits_dir=args.splits_dir)
    img_tf, mask_tf = get_val_transform()
    val_dataset = OxfordPetDataset(args.data_root, "val", transform=img_tf, target_transform=mask_tf, splits_dir=args.splits_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.model == "unet":
        warmup_epochs = 5
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5
        )
    else:  # resnet34_unet
        warmup_epochs = 5
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
        )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_dice  = 0.0
    start_epoch = 1

    # Resume (warmup_scheduler / main_scheduler 分開存)
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "warmup_scheduler" in ckpt:
            warmup_scheduler.load_state_dict(ckpt["warmup_scheduler"])
        main_scheduler.load_state_dict(ckpt["main_scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt["best_dice"]
        print(f"Resumed from epoch {start_epoch}, best val_dice={best_dice:.4f}")

    print(f"warmup_epochs: {warmup_epochs}, initial lr: {optimizer.param_groups[0]['lr']:.2e}")

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks  = masks.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = pad_and_crop(images, model, pad, IMAGE_SIZE)
                loss  = combined_loss(preds, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks  = masks.to(device).float()

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    preds = pad_and_crop(images, model, pad, IMAGE_SIZE)
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

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (val_dice={best_dice:.4f}) -> {save_path}")

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        elif args.model == "unet":
            main_scheduler.step(val_dice)
        else:
            main_scheduler.step()

        # last checkpoint
        torch.save({
            "epoch":            epoch,
            "model":            model.state_dict(),
            "optimizer":        optimizer.state_dict(),
            "warmup_scheduler": warmup_scheduler.state_dict(),
            "main_scheduler":   main_scheduler.state_dict(),
            "scaler":           scaler.state_dict(),
            "best_dice":        best_dice,
        }, last_path)

    print(f"\nDone. Best val Dice: {best_dice:.4f}")
    print(f"Model saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="unet",
                        choices=["unet", "resnet34_unet"])
    parser.add_argument("--data_root",   type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--save_dir",    type=str, default="saved_models")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--splits_dir",  type=str, default=None)
    parser.add_argument("--resume",      action="store_true")
    args = parser.parse_args()

    train(args)
