import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.oxford_pet import OxfordPetDataset
from src.utils import dice_score, get_val_transform


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
    model.to(device).eval()
    print(f"已載入 checkpoint：{args.checkpoint}")

    # 資料集
    img_tf, mask_tf = get_val_transform()
    val_dataset = OxfordPetDataset(args.data_root, "val", transform=img_tf, target_transform=mask_tf)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"驗證集：{len(val_dataset)} 筆")

    total_dice = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks  = masks.to(device).float()
            preds  = model(images)
            total_dice += dice_score(preds, masks) * images.size(0)

    avg_dice = total_dice / len(val_dataset)
    print(f"\n平均 Dice Score（val）：{avg_dice:.4f}")
    return avg_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model on val set")
    parser.add_argument("--model",       type=str, default="unet",
                        choices=["unet", "resnet34_unet"])
    parser.add_argument("--data_root",   type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    evaluate(args)
