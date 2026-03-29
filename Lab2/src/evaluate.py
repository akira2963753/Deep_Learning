import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.oxford_pet import OxfordPetDataset, IMAGE_SIZE
from src.utils import get_val_transform, dice_score


def compute_pad_size(input_size, model):
    """算 reflection padding，ResNet34-UNet 不需要所以回傳 0"""
    dummy = torch.zeros(1, 3, input_size, input_size)
    with torch.no_grad():
        out = model.cpu()(dummy)
    out_size = out.shape[2]
    if out_size >= input_size:
        return 0
    return (input_size - out_size + 1) // 2


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model == "unet":
        from src.models.unet import UNet
        model = UNet(in_channels=3, out_channels=1)
    elif args.model == "resnet34_unet":
        from src.models.resnet34_unet import ResNet34UNet
        model = ResNet34UNet(out_channels=1)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    pad = compute_pad_size(IMAGE_SIZE, model)
    model.to(device).eval()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Reflection padding: {pad} px per side")

    img_tf, mask_tf = get_val_transform()
    val_dataset = OxfordPetDataset(args.data_root, "val", transform=img_tf, target_transform=mask_tf, splits_dir=args.splits_dir)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"Val set: {len(val_dataset)} samples")

    # 收集所有預測
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            if pad > 0:
                images = F.pad(images, [pad, pad, pad, pad], mode='reflect')
            preds = model(images)
            if pad > 0:
                oh, ow = preds.shape[2], preds.shape[3]
                y1 = (oh - IMAGE_SIZE) // 2
                x1 = (ow - IMAGE_SIZE) // 2
                preds = preds[:, :, y1:y1 + IMAGE_SIZE, x1:x1 + IMAGE_SIZE]
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu().float())

    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    if args.scan_threshold:
        print("\nThreshold scan:")
        print(f"{'Threshold':>12} | {'Dice Score':>12}")
        print("-" * 28)
        best_thresh, best_dice = 0.5, 0.0
        for t in [round(x * 0.05, 2) for x in range(6, 13)]:
            score = dice_score(all_preds, all_masks, threshold=t)
            marker = " <- best" if score > best_dice else ""
            if score > best_dice:
                best_dice, best_thresh = score, t
            print(f"{t:>12.2f} | {score:>12.4f}{marker}")
        print(f"\nBest threshold: {best_thresh}  Dice: {best_dice:.4f}")
    else:
        score = dice_score(all_preds, all_masks, threshold=args.threshold)
        print(f"\nVal Dice (threshold={args.threshold}): {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           type=str, default="unet",
                        choices=["unet", "resnet34_unet"])
    parser.add_argument("--data_root",       type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--num_workers",     type=int, default=2)
    parser.add_argument("--splits_dir",      type=str, default=None)
    parser.add_argument("--threshold",       type=float, default=0.5)
    parser.add_argument("--scan_threshold",  action="store_true")
    args = parser.parse_args()

    evaluate(args)
