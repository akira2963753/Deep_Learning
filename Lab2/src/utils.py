import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
import numpy as np

# ImageNet 標準化參數（與 oxford_pet.py 一致）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE = 384


# ---------------------------------------------------------------------------
# Loss & Metric
# ---------------------------------------------------------------------------

def dice_loss(pred, target, smooth=1e-6):
    """
    Dice Loss。
    pred:   logits，shape (B, 1, H, W)
    target: binary mask，shape (B, 1, H, W)，值為 0 或 1
    """
    pred = torch.sigmoid(pred)
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1).float()

    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Dice Score（評估用，不用梯度）。
    pred:   logits 或 sigmoid 輸出，shape (B, 1, H, W)
    target: binary mask，shape (B, 1, H, W)
    """
    with torch.no_grad():
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()
        pred   = pred.contiguous().view(-1)
        target = target.contiguous().view(-1).float()

        intersection = (pred * target).sum()
        return ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def combined_loss(pred, target):
    """
    0.5 × BCEWithLogitsLoss + 0.5 × DiceLoss
    """
    bce = F.binary_cross_entropy_with_logits(pred, target.float())
    d   = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * d


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------

class JointTransform:
    """
    對 (image PIL, mask PIL) 套用一致的幾何隨機變換，
    顏色變換只套用在 image 上。
    """

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        hflip_p=0.5,
        vflip_p=0.5,
        rotation_deg=15,
        color_jitter=True,
        elastic_p=0.5,
        elastic_alpha=80,
        elastic_sigma=10,
        gaussian_blur_p=0.0,
    ):
        self.image_size   = image_size
        self.hflip_p      = hflip_p
        self.vflip_p      = vflip_p
        self.rotation_deg = rotation_deg
        self.elastic_p     = elastic_p
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.gaussian_blur_p = gaussian_blur_p

        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ) if color_jitter else None

        self.gaussian_blur = transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))

        self.to_tensor_img  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.to_tensor_mask = transforms.PILToTensor()

    def __call__(self, image: Image.Image, mask: Image.Image):
        # 1. Resize（兩者一起）
        image = transforms.functional.resize(image, (self.image_size, self.image_size))
        mask  = transforms.functional.resize(
            mask, (self.image_size, self.image_size),
            interpolation=transforms.InterpolationMode.NEAREST
        )

        # 2. Random Horizontal Flip（同一個隨機決定）
        if random.random() < self.hflip_p:
            image = transforms.functional.hflip(image)
            mask  = transforms.functional.hflip(mask)

        # 3. Random Vertical Flip
        if random.random() < self.vflip_p:
            image = transforms.functional.vflip(image)
            mask  = transforms.functional.vflip(mask)

        # 4. Random Rotation（同一個角度）
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        image = transforms.functional.rotate(image, angle)
        mask  = transforms.functional.rotate(
            mask, angle,
            interpolation=transforms.InterpolationMode.NEAREST
        )

        # 5. Elastic Deformation（同一組位移場套用在 image 和 mask）
        if random.random() < self.elastic_p:
            image, mask = self._elastic_deform(image, mask)

        # 6. Color Jitter（只套用在 image）
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        # 7. Gaussian Blur（只套用在 image）
        if random.random() < self.gaussian_blur_p:
            image = self.gaussian_blur(image)

        # 8. ToTensor + Normalize
        image = self.to_tensor_img(image)
        mask  = self.to_tensor_mask(mask)  # shape (1, H, W), dtype uint8

        return image, mask

    def _elastic_deform(self, image: Image.Image, mask: Image.Image):
        """
        Elastic deformation（Simard et al., 2003 / UNet 原論文做法）。
        對 image 和 mask 施加相同的隨機位移場。
        """
        h, w = self.image_size, self.image_size
        # 產生隨機位移場，用高斯平滑使其連續
        dx = gaussian_filter(np.random.randn(h, w) * self.elastic_alpha, self.elastic_sigma)
        dy = gaussian_filter(np.random.randn(h, w) * self.elastic_alpha, self.elastic_sigma)

        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        indices_y = np.clip(y + dy, 0, h - 1)
        indices_x = np.clip(x + dx, 0, w - 1)

        # 對 image 的每個 channel 用三次插值
        img_arr = np.array(image)
        result_img = np.zeros_like(img_arr)
        for c in range(img_arr.shape[2]):
            result_img[:, :, c] = map_coordinates(
                img_arr[:, :, c], [indices_y, indices_x], order=1, mode='reflect'
            )
        # 對 mask 用最近鄰插值（避免產生非 0/1 的值）
        mask_arr = np.array(mask)
        result_mask = map_coordinates(
            mask_arr, [indices_y, indices_x], order=0, mode='reflect'
        )

        return Image.fromarray(result_img), Image.fromarray(result_mask)


def get_train_transform(elastic_p=0.0, gaussian_blur_p=0.0):
    """訓練集用（含 augmentation）"""
    return JointTransform(
        image_size=IMAGE_SIZE,
        hflip_p=0.5,
        vflip_p=0.5,
        rotation_deg=15,
        color_jitter=True,
        elastic_p=elastic_p,
        elastic_alpha=80,
        elastic_sigma=10,
        gaussian_blur_p=gaussian_blur_p,
    )


def get_val_transform():
    """驗證 / 測試集用（只做 resize + normalize）"""
    img_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.NEAREST
        ),
        transforms.PILToTensor(),
    ])
    return img_tf, mask_tf
