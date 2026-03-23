import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.oxford_pet import OxfordPetDataset, IMAGE_SIZE
from src.utils import get_val_transform


def compute_pad_size(input_size, model):
    """計算需要多少 reflection padding 才能讓模型輸出 >= input_size。
    若模型輸出已等於輸入（如 ResNet34-UNet），回傳 0。"""
    dummy = torch.zeros(1, 3, input_size, input_size)
    with torch.no_grad():
        out = model.cpu()(dummy)
    out_size = out.shape[2]
    if out_size >= input_size:
        return 0
    pad_per_side = (input_size - out_size + 1) // 2 + 1
    return pad_per_side


def dice_score_with_threshold(pred, target, threshold=0.5, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1).float()
    intersection = (pred * target).sum()
    return ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def evaluate(args):
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
    pad = compute_pad_size(IMAGE_SIZE, model)
    model.to(device).eval()
    print(f"已載入 checkpoint：{args.checkpoint}")
    print(f"評估用 reflection padding：每邊 {pad} 像素")

    # 資料集
    img_tf, mask_tf = get_val_transform()
    val_dataset = OxfordPetDataset(args.data_root, "val", transform=img_tf, target_transform=mask_tf, splits_dir=args.splits_dir)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"驗證集：{len(val_dataset)} 筆")

    # 收集所有 logits 和 masks（避免重複推論）
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            # Reflection padding（UNet）或直接 forward（ResNet34-UNet）
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
        # 掃描 threshold，找最佳值
        print("\nThreshold 掃描結果：")
        print(f"{'Threshold':>12} | {'Dice Score':>12}")
        print("-" * 28)
        best_thresh, best_dice = 0.5, 0.0
        for t in [round(x * 0.05, 2) for x in range(6, 13)]:  # 0.30 ~ 0.60
            score = dice_score_with_threshold(all_preds, all_masks, threshold=t)
            marker = " ← best" if score > best_dice else ""
            if score > best_dice:
                best_dice, best_thresh = score, t
            print(f"{t:>12.2f} | {score:>12.4f}{marker}")
        print(f"\n最佳 Threshold：{best_thresh}  Dice Score：{best_dice:.4f}")
    else:
        score = dice_score_with_threshold(all_preds, all_masks, threshold=args.threshold)
        print(f"\n平均 Dice Score（val, threshold={args.threshold}）：{score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model on val set")
    parser.add_argument("--model",           type=str, default="unet",
                        choices=["unet", "resnet34_unet"])
    parser.add_argument("--data_root",       type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--num_workers",     type=int, default=2)
    parser.add_argument("--splits_dir",      type=str, default=None)
    parser.add_argument("--threshold",       type=float, default=0.5)
    parser.add_argument("--scan_threshold",  action="store_true",
                        help="掃描 0.30~0.60 找最佳 threshold")
    args = parser.parse_args()

    evaluate(args)
