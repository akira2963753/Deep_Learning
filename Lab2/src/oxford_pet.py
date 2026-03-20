import os
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Oxford-IIIT Pet Dataset 下載連結
URLS = {
    "images": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    "annotations": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
}

# ImageNet 標準化參數
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 512  # resize 目標尺寸


def _download_with_progress(url: str, dest: Path) -> None:
    """下載檔案並顯示進度。"""
    try:
        from tqdm import tqdm

        class _ProgressHook:
            def __init__(self):
                self.pbar = None

            def __call__(self, block_num, block_size, total_size):
                if self.pbar is None:
                    self.pbar = tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=dest.name,
                    )
                downloaded = block_num * block_size
                self.pbar.update(min(block_size, total_size - downloaded + block_size))

            def close(self):
                if self.pbar:
                    self.pbar.close()

        hook = _ProgressHook()
        urllib.request.urlretrieve(url, dest, reporthook=hook)
        hook.close()

    except ImportError:
        print(f"正在下載 {dest.name} ...")
        urllib.request.urlretrieve(url, dest)


def download_dataset(root: str) -> None:
    """
    下載並解壓縮 Oxford-IIIT Pet Dataset。
    若資料夾已存在則跳過下載。

    Args:
        root: 資料集根目錄，例如 'dataset/oxford-iiit-pet'
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    for name, url in URLS.items():
        tar_path = root / Path(url).name

        # 檢查是否已解壓縮完成
        extracted_dir = root / name  # 'images' or 'annotations'
        if extracted_dir.exists():
            print(f"[跳過] {name} 已存在：{extracted_dir}")
            continue

        # 下載
        if not tar_path.exists():
            print(f"正在下載 {name}...")
            _download_with_progress(url, tar_path)
        else:
            print(f"[跳過下載] {tar_path.name} 已存在，直接解壓縮。")

        # 解壓縮
        print(f"正在解壓縮 {tar_path.name}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(root)
        print(f"解壓縮完成：{extracted_dir}")

        # 解壓縮後刪除壓縮檔以節省空間（可選）
        # tar_path.unlink()


def _read_name_list(path: Path) -> list:
    """每行只有 image_name 的 txt 格式（Kaggle 格式）"""
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]


def _get_kaggle_splits(splits_dir: str) -> dict:
    """
    從 Kaggle 提供的 split txt 讀取 train/val 清單。
    splits_dir 應包含 train.txt 和 val.txt。
    """
    splits_dir = Path(splits_dir)
    return {
        "train": _read_name_list(splits_dir / "train.txt"),
        "val":   _read_name_list(splits_dir / "val.txt"),
    }


def _parse_split_file(split_file: Path) -> list:
    """
    解析 Oxford 提供的 split 文字檔（trainval.txt / test.txt）。
    每行格式：<image_name> <class_id> <species> <breed_id>
    回傳圖片名稱列表（不含副檔名）。
    """
    names = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line.split()[0])
    return names


def _get_splits(root: str, seed: int = 42):
    """
    從官方 trainval.txt 切出 8:2 的訓練/驗證集，
    test.txt 作為測試集。

    Returns:
        dict with keys 'train', 'val', 'test', each a list of image names.
    """
    root = Path(root)
    trainval_file = root / "annotations" / "trainval.txt"
    test_file = root / "annotations" / "test.txt"

    trainval_names = _parse_split_file(trainval_file)
    test_names = _parse_split_file(test_file)

    # 固定 seed 確保可重現
    rng = random.Random(seed)
    shuffled = trainval_names[:]
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * 0.8)
    train_names = shuffled[:split_idx]
    val_names = shuffled[split_idx:]

    return {"train": train_names, "val": val_names, "test": test_names}


def _trimap_to_binary(mask: np.ndarray) -> np.ndarray:
    """
    將 trimap 轉成 binary mask：
      1（前景）→ 1
      2（背景）→ 0
      3（邊界）→ 0  (依作業規定，邊界視為背景)
    """
    return (mask == 1).astype(np.uint8)


class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset PyTorch Dataset。

    Args:
        root:             資料集根目錄（包含 images/ 和 annotations/）
        split:            'train' | 'val' | 'test'
        transform:        對 PIL Image 進行的 transform（輸入圖片）
        target_transform: 對 binary mask（PIL Image）進行的 transform
        splits_dir:       Kaggle 提供的 split 目錄（含 train.txt / val.txt）。
                          若為 None，使用原本的 Oxford 8:2 切割。
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_transform=None,
        splits_dir: str = None,
    ):
        assert split in ("train", "val", "test"), \
            f"split 必須為 'train', 'val', 'test'，收到：{split}"

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

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        name = self.image_names[idx]

        # 載入圖片
        img_path = self.images_dir / f"{name}.jpg"
        image = Image.open(img_path).convert("RGB")

        # 載入 mask（test 集無 ground truth）
        mask = None
        if self.split != "test":
            mask_path = self.masks_dir / f"{name}.png"
            trimap = np.array(Image.open(mask_path))
            binary = _trimap_to_binary(trimap)
            mask = Image.fromarray(binary)

        # 套用 transform
        if self.transform is not None:
            image = self.transform(image)
        if mask is not None and self.target_transform is not None:
            mask = self.target_transform(mask)

        if mask is None:
            return image, name  # test 集回傳檔名供推論使用
        return image, mask


def get_default_transform():
    """回傳預設圖片 transform（resize → tensor → normalize）。"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_default_target_transform():
    """回傳預設 mask transform（resize → tensor，使用最近鄰插值避免產生新值）。"""
    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.PILToTensor(),  # shape: (1, H, W), dtype: uint8
    ])


def load_dataset(root: str, split: str, transform=None, target_transform=None, splits_dir: str = None) -> OxfordPetDataset:
    """
    建立並回傳 OxfordPetDataset。

    若 transform / target_transform 為 None，會自動套用預設前處理。

    Args:
        root:             資料集根目錄（如 'dataset/oxford-iiit-pet'）
        split:            'train' | 'val' | 'test'
        transform:        自訂圖片 transform（None 則用預設）
        target_transform: 自訂 mask transform（None 則用預設）

    Returns:
        OxfordPetDataset instance
    """
    if transform is None:
        transform = get_default_transform()
    if target_transform is None and split != "test":
        target_transform = get_default_target_transform()

    return OxfordPetDataset(root, split, transform, target_transform, splits_dir=splits_dir)


# ---------------------------------------------------------------------------
# 測試區塊：直接執行此檔案可驗證資料集是否正確載入
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    DATASET_ROOT = "dataset/oxford-iiit-pet"

    # 1. 下載資料集
    print("=" * 50)
    print("步驟 1：下載資料集")
    print("=" * 50)
    download_dataset(DATASET_ROOT)

    # 2. 載入各 split
    print("\n" + "=" * 50)
    print("步驟 2：載入資料集並確認筆數")
    print("=" * 50)
    for split in ("train", "val", "test"):
        ds = load_dataset(DATASET_ROOT, split)
        print(f"  {split:5s}: {len(ds):5d} 筆")

    # 3. 取一筆樣本確認
    print("\n" + "=" * 50)
    print("步驟 3：取一筆樣本確認 shape 與 mask 值")
    print("=" * 50)
    train_ds = load_dataset(DATASET_ROOT, "train")
    image, mask = train_ds[0]
    print(f"  image shape : {image.shape}  dtype: {image.dtype}")
    print(f"  mask  shape : {mask.shape}   dtype: {mask.dtype}")
    unique_values = torch.unique(mask).tolist()
    print(f"  mask unique values: {unique_values}")
    assert set(unique_values).issubset({0, 1}), "mask 值應只有 0 和 1！"
    print("  ✓ mask 值驗證通過")

    # 4. 確認 test split 回傳格式
    test_ds = load_dataset(DATASET_ROOT, "test")
    img, name = test_ds[0]
    print(f"\n  test 樣本 name: {name}, image shape: {img.shape}")
    print("\n全部驗證通過！")
