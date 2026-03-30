import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import label, binary_closing


def postprocess(mask: np.ndarray, closing_kernel: int = 5) -> np.ndarray:
    """
    1. Morphological Closing：填補體內破洞
    2. LCC：保留最大連通組件，清除背景雜訊
    mask: (H, W) uint8, 值為 0/1
    """
    struct = np.ones((closing_kernel, closing_kernel), dtype=bool)
    mask = binary_closing(mask.astype(bool), structure=struct).astype(np.uint8)

    labeled, num_features = label(mask)
    if num_features == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    return (labeled == largest).astype(np.uint8)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.oxford_pet import OxfordPetDataset
from src.utils import get_val_transform, IMAGE_SIZE


def compute_pad_size(input_size, model):
    """算 reflection padding，ResNet34-UNet 不需要所以回傳 0"""
    dummy = torch.zeros(1, 3, input_size, input_size)
    with torch.no_grad():
        out = model.cpu()(dummy)
    out_size = out.shape[2]
    if out_size >= input_size:
        return 0
    return (input_size - out_size + 1) // 2


def rle_encode(mask: np.ndarray) -> str:
    """把 binary mask 轉成 RLE 字串 (column-major, 1-indexed)"""
    mask = mask.flatten(order="F")
    changes = np.diff(mask, prepend=0, append=0)
    starts  = np.where(changes == 1)[0] + 1
    ends    = np.where(changes == -1)[0] + 1
    lengths = ends - starts
    if len(starts) == 0:
        return ""
    return " ".join(f"{s} {l}" for s, l in zip(starts, lengths))


def run_inference(args):
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

    img_tf, _ = get_val_transform()

    if args.test_list:
        from pathlib import Path
        import torch.utils.data as tud

        with open(args.test_list, "r") as f:
            image_names = [l.strip() for l in f if l.strip()]

        class KaggleTestDataset(tud.Dataset):
            def __init__(self, root, names, transform):
                self.images_dir = Path(root) / "images"
                self.names = names
                self.transform = transform
            def __len__(self): return len(self.names)
            def __getitem__(self, idx):
                name = self.names[idx]
                from PIL import Image as PILImage
                img = PILImage.open(self.images_dir / f"{name}.jpg").convert("RGB")
                orig_size = (img.height, img.width)
                if self.transform: img = self.transform(img)
                return img, name, orig_size

        test_dataset = KaggleTestDataset(args.data_root, image_names, img_tf)
    else:
        test_dataset = OxfordPetDataset(args.data_root, "test", transform=img_tf)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"Test set: {len(test_dataset)} samples")

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, names, orig_sizes = batch
            else:
                images, names = batch
                orig_sizes = None

            images = images.to(device)

            def _forward(imgs):
                if pad > 0:
                    imgs = F.pad(imgs, [pad, pad, pad, pad], mode='reflect')
                logits = model(imgs)
                if pad > 0:
                    oh, ow = logits.shape[2], logits.shape[3]
                    y1 = (oh - IMAGE_SIZE) // 2
                    x1 = (ow - IMAGE_SIZE) // 2
                    logits = logits[:, :, y1:y1 + IMAGE_SIZE, x1:x1 + IMAGE_SIZE]
                return torch.sigmoid(logits)

            if args.tta:
                # TTA: 原圖 + hflip + vflip + 兩者
                p0 = _forward(images)
                p1 = torch.flip(_forward(torch.flip(images, [3])), [3])
                p2 = torch.flip(_forward(torch.flip(images, [2])), [2])
                p3 = torch.flip(_forward(torch.flip(images, [2, 3])), [2, 3])
                preds = (p0 + p1 + p2 + p3) / 4.0
            else:
                preds = _forward(images)
            preds = (preds > args.threshold).cpu().numpy()

            for i, (pred, name) in enumerate(zip(preds, names)):
                mask_2d = pred[0].astype(np.uint8)
                if args.postprocess:
                    mask_2d = postprocess(mask_2d, closing_kernel=args.closing_kernel)

                # resize 回原始大小
                if orig_sizes is not None:
                    oh = int(orig_sizes[0][i]) if hasattr(orig_sizes[0], '__len__') else int(orig_sizes[0])
                    ow = int(orig_sizes[1][i]) if hasattr(orig_sizes[1], '__len__') else int(orig_sizes[1])
                    from PIL import Image as PILImage
                    mask_pil = PILImage.fromarray(mask_2d)
                    mask_pil = mask_pil.resize((ow, oh), PILImage.NEAREST)
                    mask_2d  = np.array(mask_pil)

                rle = rle_encode(mask_2d)
                rows.append((name, rle))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "encoded_mask"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} predictions -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="unet",
                        choices=["unet", "resnet34_unet"])
    parser.add_argument("--data_root",   type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--output",      type=str, default="unet_pred.csv")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--tta",             action="store_true")
    parser.add_argument("--num_workers",     type=int, default=2)
    parser.add_argument("--test_list",       type=str, default=None)
    parser.add_argument("--postprocess",     action="store_true")
    parser.add_argument("--closing_kernel",  type=int, default=5)
    args = parser.parse_args()

    run_inference(args)
