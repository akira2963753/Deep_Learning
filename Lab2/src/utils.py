import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
import numpy as np

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE = 384


def pad_and_crop(images, model, pad, image_size, apply_sigmoid=False):
    """pad -> forward -> crop 回原始大小，pad=0 時直接 forward"""
    if pad > 0:
        images = F.pad(images, [pad, pad, pad, pad], mode='reflect')
    out = model(images)
    if pad > 0:
        oh, ow = out.shape[2], out.shape[3]
        y1 = (oh - image_size) // 2
        x1 = (ow - image_size) // 2
        out = out[:, :, y1:y1 + image_size, x1:x1 + image_size]
    if apply_sigmoid:
        out = torch.sigmoid(out)
    return out


def compute_pad_size(input_size, model):
    """算 reflection padding，ResNet34-UNet 不需要所以回傳 0"""
    dummy = torch.zeros(1, 3, input_size, input_size)
    with torch.no_grad():
        out = model.cpu()(dummy)
    out_size = out.shape[2]
    if out_size >= input_size:
        return 0
    return (input_size - out_size + 1) // 2


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1).float()
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """評估用的 Dice Score"""
    with torch.no_grad():
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()
        pred   = pred.contiguous().view(-1)
        target = target.contiguous().view(-1).float()
        intersection = (pred * target).sum()
        return ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def combined_loss(pred, target):
    """0.5 * BCE + 0.5 * Dice"""
    bce = F.binary_cross_entropy_with_logits(pred, target.float())
    d   = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * d


class JointTransform:
    """對 image 和 mask 同時做 augmentation"""

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
    ):
        self.image_size   = image_size
        self.hflip_p      = hflip_p
        self.vflip_p      = vflip_p
        self.rotation_deg = rotation_deg
        self.elastic_p     = elastic_p
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ) if color_jitter else None

        self.to_tensor_img  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.to_tensor_mask = transforms.PILToTensor()

    def __call__(self, image: Image.Image, mask: Image.Image):
        # Resize
        image = transforms.functional.resize(image, (self.image_size, self.image_size))
        mask  = transforms.functional.resize(
            mask, (self.image_size, self.image_size),
            interpolation=transforms.InterpolationMode.NEAREST
        )

        # Random flip
        if random.random() < self.hflip_p:
            image = transforms.functional.hflip(image)
            mask  = transforms.functional.hflip(mask)
        if random.random() < self.vflip_p:
            image = transforms.functional.vflip(image)
            mask  = transforms.functional.vflip(mask)

        # Random rotation
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        image = transforms.functional.rotate(image, angle)
        mask  = transforms.functional.rotate(
            mask, angle,
            interpolation=transforms.InterpolationMode.NEAREST
        )

        # Elastic deformation (同一組位移場)
        if random.random() < self.elastic_p:
            image, mask = self._elastic_deform(image, mask)

        # Color jitter (只對 image)
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        image = self.to_tensor_img(image)
        mask  = self.to_tensor_mask(mask)
        return image, mask

    def _elastic_deform(self, image: Image.Image, mask: Image.Image):
        """Elastic deformation (Simard et al., 2003)"""
        h, w = self.image_size, self.image_size
        dx = gaussian_filter(np.random.randn(h, w) * self.elastic_alpha, self.elastic_sigma)
        dy = gaussian_filter(np.random.randn(h, w) * self.elastic_alpha, self.elastic_sigma)

        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        indices_y = np.clip(y + dy, 0, h - 1)
        indices_x = np.clip(x + dx, 0, w - 1)

        img_arr = np.array(image)
        result_img = np.zeros_like(img_arr)
        for c in range(img_arr.shape[2]):
            result_img[:, :, c] = map_coordinates(
                img_arr[:, :, c], [indices_y, indices_x], order=1, mode='reflect'
            )
        # mask 用 nearest 避免產生非 0/1 的值
        mask_arr = np.array(mask)
        result_mask = map_coordinates(
            mask_arr, [indices_y, indices_x], order=0, mode='reflect'
        )

        return Image.fromarray(result_img), Image.fromarray(result_mask)


def get_train_transform(elastic_p=0.0):
    return JointTransform(
        image_size=IMAGE_SIZE,
        hflip_p=0.5,
        vflip_p=0.5,
        rotation_deg=20,
        color_jitter=True,
        elastic_p=elastic_p,
        elastic_alpha=80,
        elastic_sigma=10,
    )


def get_val_transform():
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
