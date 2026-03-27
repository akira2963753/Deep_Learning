import os
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

URLS = {
    "images": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    "annotations": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
}

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 384

def _download_with_progress(url: str, dest: Path) -> None:
    try:
        from tqdm import tqdm

        class _ProgressHook:
            def __init__(self):
                self.pbar = None
            def __call__(self, block_num, block_size, total_size):
                if self.pbar is None:
                    self.pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name)
                downloaded = block_num * block_size
                self.pbar.update(min(block_size, total_size - downloaded + block_size))
            def close(self):
                if self.pbar:
                    self.pbar.close()

        hook = _ProgressHook()
        urllib.request.urlretrieve(url, dest, reporthook=hook)
        hook.close()

    except ImportError:
        print(f"Downloading {dest.name} ...")
        urllib.request.urlretrieve(url, dest)

def download_dataset(root: str) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    for name, url in URLS.items():
        tar_path = root / Path(url).name
        extracted_dir = root / name

        if extracted_dir.exists():
            print(f"[Skip] {name} already exists: {extracted_dir}")
            continue

        if not tar_path.exists():
            print(f"Downloading {name}...")
            _download_with_progress(url, tar_path)
        else:
            print(f"[Skip download] {tar_path.name} exists, extracting.")

        print(f"Extracting {tar_path.name}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(root)
        print(f"Done: {extracted_dir}")

def _read_name_list(path: Path) -> list:
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]

def _get_kaggle_splits(splits_dir: str) -> dict:
    """讀 Kaggle 提供的 train/val split"""
    splits_dir = Path(splits_dir)
    return {
        "train": _read_name_list(splits_dir / "train.txt"),
        "val":   _read_name_list(splits_dir / "val.txt"),
    }

def _parse_split_file(split_file: Path) -> list:
    """讀 Oxford split 檔，每行: <image_name> <class_id> <species> <breed_id>"""
    names = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line.split()[0])
    return names

def _get_splits(root: str, seed: int = 42):
    """從 trainval.txt 切 8:2 train/val，test.txt 當 test"""
    root = Path(root)
    trainval_file = root / "annotations" / "trainval.txt"
    test_file = root / "annotations" / "test.txt"

    trainval_names = _parse_split_file(trainval_file)
    test_names = _parse_split_file(test_file)

    rng = random.Random(seed)
    shuffled = trainval_names[:]
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * 0.8)
    train_names = shuffled[:split_idx]
    val_names = shuffled[split_idx:]

    return {"train": train_names, "val": val_names, "test": test_names}


def _trimap_to_binary(mask: np.ndarray) -> np.ndarray:
    """trimap -> binary: 1(前景)->1, 2(背景)->0, 3(邊界)->0"""
    return (mask == 1).astype(np.uint8)


class OxfordPetDataset(Dataset):

    def __init__(self, root, split="train", transform=None, target_transform=None, splits_dir=None):
        assert split in ("train", "val", "test")

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if splits_dir is not None and split != "test":
            splits = _get_kaggle_splits(splits_dir)
        else:
            splits = _get_splits(root)
        self.image_names = splits[split]

        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "annotations" / "trimaps"

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]

        img_path = self.images_dir / f"{name}.jpg"
        image = Image.open(img_path).convert("RGB")

        mask = None
        if self.split != "test":
            mask_path = self.masks_dir / f"{name}.png"
            trimap = np.array(Image.open(mask_path))
            binary = _trimap_to_binary(trimap)
            mask = Image.fromarray(binary)

        if self.transform is not None:
            image = self.transform(image)
        if mask is not None and self.target_transform is not None:
            mask = self.target_transform(mask)

        if mask is None:
            return image, name  # test 沒有 mask
        return image, mask


def get_default_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_default_target_transform():
    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.PILToTensor(),
    ])


def load_dataset(root, split, transform=None, target_transform=None, splits_dir=None):
    if transform is None:
        transform = get_default_transform()
    if target_transform is None and split != "test":
        target_transform = get_default_target_transform()
    return OxfordPetDataset(root, split, transform, target_transform, splits_dir=splits_dir)


if __name__ == "__main__":
    import torch

    DATASET_ROOT = "dataset/oxford-iiit-pet"

    print("=== Download dataset ===")
    download_dataset(DATASET_ROOT)

    print("\n=== Check dataset size ===")
    for split in ("train", "val", "test"):
        ds = load_dataset(DATASET_ROOT, split)
        print(f"  {split:5s}: {len(ds):5d}")

    print("\n=== Check sample ===")
    train_ds = load_dataset(DATASET_ROOT, "train")
    image, mask = train_ds[0]
    print(f"  image: {image.shape}  mask: {mask.shape}")
    unique_values = torch.unique(mask).tolist()
    print(f"  mask values: {unique_values}")
    assert set(unique_values).issubset({0, 1})

    test_ds = load_dataset(DATASET_ROOT, "test")
    img, name = test_ds[0]
    print(f"  test sample: {name}, shape: {img.shape}")
    print("\nAll OK!")
