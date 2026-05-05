import os
import json
from typing import List, Dict, Optional, Callable, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def labels_to_onehot(labels: List[str], obj_dict: Dict[str, int]) -> torch.FloatTensor:
    vec = torch.zeros(len(obj_dict), dtype=torch.float32)
    for lbl in labels:
        if lbl not in obj_dict:
            raise KeyError(f"Unknown label: '{lbl}'. Valid: {list(obj_dict.keys())}")
        vec[obj_dict[lbl]] = 1.0
    return vec


def get_transform(image_size: int = 64) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class ICLEVRDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        json_path: str,
        obj_json_path: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir

        with open(obj_json_path, 'r') as f:
            self.obj_dict: Dict[str, int] = json.load(f)

        with open(json_path, 'r') as f:
            raw: Dict[str, List[str]] = json.load(f)

        self.data: List[Tuple[str, List[str]]] = list(raw.items())
        self.transform = transform if transform is not None else get_transform()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        filename, label_list = self.data[idx]
        img_path = os.path.join(self.image_dir, filename)

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Image not found: {img_path}. Check image_dir='{self.image_dir}'."
            )

        image_tensor = self.transform(image)
        label_onehot = labels_to_onehot(label_list, self.obj_dict)
        return image_tensor, label_onehot


class TestDataset(Dataset):
    def __init__(self, json_path: str, obj_json_path: str) -> None:
        super().__init__()

        with open(obj_json_path, 'r') as f:
            self.obj_dict: Dict[str, int] = json.load(f)

        with open(json_path, 'r') as f:
            self.labels: List[List[str]] = json.load(f)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> torch.FloatTensor:
        return labels_to_onehot(self.labels[idx], self.obj_dict)


if __name__ == '__main__':
    BASE = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR     = os.path.join(BASE, 'iclevr')
    TRAIN_JSON    = os.path.join(BASE, 'train.json')
    TEST_JSON     = os.path.join(BASE, 'test.json')
    NEW_TEST_JSON = os.path.join(BASE, 'new_test.json')
    OBJ_JSON      = os.path.join(BASE, 'objects.json')

    print("=== ICLEVRDataset ===")
    train_ds = ICLEVRDataset(IMAGE_DIR, TRAIN_JSON, OBJ_JSON)
    print(f"Train size      : {len(train_ds)}")

    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    imgs, lbls = next(iter(loader))
    print(f"Image shape     : {imgs.shape}")
    print(f"Label shape     : {lbls.shape}")
    print(f"Image min/max   : {imgs.min():.4f} / {imgs.max():.4f}")
    print(f"Label row sums  : {lbls.sum(dim=1).tolist()}")

    print("\n=== TestDataset ===")
    for name, path in [('test.json', TEST_JSON), ('new_test.json', NEW_TEST_JSON)]:
        ds = TestDataset(path, OBJ_JSON)
        ldr = DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=0)
        lbl = next(iter(ldr))
        print(f"{name}: size={len(ds)}, label shape={lbl.shape}, "
              f"row sums={lbl.sum(dim=1).tolist()}")
