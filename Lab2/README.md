# DL Lab2 - Oxford-IIIT Pet Segmentation

NYCU 2026 Spring Deep Learning Lab2 - Binary Segmentation using UNet & ResNet34-UNet

## Results Summary

| Model | Val Dice | Test Dice (Local) | Kaggle Score | Kaggle Rank |
|-------|----------|-------------------|--------------|-------------|
| UNet (from scratch) | 0.9098 | ~0.9034 | ~0.9034 | - |
| ResNet34-UNet (from scratch) | 0.9336 | 0.9228 | 0.9228 | 7th |

## Training Configuration

### Common Settings

| Item | Value |
|------|-------|
| Input Size | 384 x 384 |
| Loss Function | 0.5 x BCEWithLogitsLoss + 0.5 x DiceLoss |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Scheduler | ReduceLROnPlateau (mode=max, patience=5, factor=0.5) |
| Mixed Precision (AMP) | Enabled |
| Gradient Clipping | max_norm=1.0 |
| Random Seed | 42 |

### UNet

| Item | Value |
|------|-------|
| Architecture | Vanilla UNet (2015 paper), valid convolutions (padding=0), no BatchNorm |
| Encoder Channels | 3 → 64 → 128 → 256 → 512 → 1024 |
| Skip Connections | Center crop (faithful to original paper) |
| Output Size | 196 x 196 (from 384 x 384 input) |
| Reflection Padding | 95 pixels per side (inference only) |
| Training Mask Handling | Center crop mask to 196 x 196 to match model output |
| Batch Size | 16 |
| Epochs | 70 |
| Data Augmentation | HFlip (p=0.5), VFlip (p=0.5), Rotation (±15°), ColorJitter |
| Elastic Deformation | Disabled (no BN, training unstable) |
| TTA | Original + HFlip + VFlip + HFlip+VFlip (4-fold average) |

### ResNet34-UNet

| Item | Value |
|------|-------|
| Architecture | ResNet34 encoder + UNet decoder, padding=1, BatchNorm |
| Encoder | ResNet34 BasicBlock (3/4/6/3 blocks), stride-based downsampling |
| Decoder | ConvTranspose2d upsample + DoubleConv (BN + ReLU) |
| Output Size | 384 x 384 (same as input, no padding needed) |
| Reflection Padding | Not needed (pad=0) |
| Batch Size | 16 |
| Epochs | 130 |
| Data Augmentation | HFlip (p=0.5), VFlip (p=0.5), Rotation (±15°), ColorJitter |
| Elastic Deformation | Enabled (p=0.3, alpha=80, sigma=10) |
| TTA | Original + HFlip + VFlip + HFlip+VFlip (4-fold average) |

## Key Techniques

### Reflection Padding (UNet only)
UNet uses valid convolutions (padding=0), so the output is smaller than input (384→196). During training, the mask is center-cropped to match. During inference, the input is padded with reflection padding (95px per side) so the model output covers the full image. The output is then center-cropped back to 384x384.

### Elastic Deformation (ResNet34-UNet only)
Following the original UNet paper (Simard et al., 2003), random elastic deformation is applied to both image and mask jointly. This improves generalization and reduces the val/test gap. Only used for ResNet34-UNet as UNet (without BN) was unstable with this augmentation.

### Test Time Augmentation (TTA)
During inference, predictions are averaged over 4 augmented views:
1. Original image
2. Horizontal flip
3. Vertical flip
4. Horizontal + Vertical flip

### Mixed Precision Training
`torch.amp.autocast` + `GradScaler` for faster training and lower VRAM usage.

## Project Structure

```
Lab2/
├── src/
│   ├── models/
│   │   ├── unet.py              # Vanilla UNet (valid conv, no BN)
│   │   └── resnet34_unet.py     # ResNet34 encoder + UNet decoder
│   ├── train.py                 # Training script (supports both models)
│   ├── evaluate.py              # Validation set evaluation
│   ├── inference.py             # Kaggle CSV generation
│   ├── infer_localtest.py       # Local test set evaluation
│   ├── oxford_pet.py            # Dataset loader
│   └── utils.py                 # Loss, metrics, augmentation
├── saved_models/                # Best & last checkpoints
├── dataset/                     # Oxford-IIIT Pet dataset
└── README.md
```

## Usage

### Training
```bash
# UNet
python src/train.py --model unet --batch_size 16 --epochs 50

# ResNet34-UNet
python src/train.py --model resnet34_unet --batch_size 16 --epochs 130

# Resume from last checkpoint
python src/train.py --model resnet34_unet --batch_size 16 --epochs 200 --resume
```

### Evaluation
```bash
# Validate on val set
python src/evaluate.py --model resnet34_unet --checkpoint saved_models/resnet34_unet_best.pth

# Local test evaluation with TTA
python src/infer_localtest.py --model resnet34_unet \
    --checkpoint saved_models/resnet34_unet_best.pth \
    --test_list nycu-2026-spring-dl-lab2-unet/test.txt --tta

# Generate Kaggle submission CSV
python src/inference.py --model resnet34_unet \
    --checkpoint saved_models/resnet34_unet_best.pth \
    --test_list nycu-2026-spring-dl-lab2-unet/test.txt \
    --output resnet34_unet_pred.csv --tta
```

## Training History Notes

### UNet
- Initial training with F.interpolate for mask resizing → val_dice plateaued at ~0.71
- **Key fix**: Replaced F.interpolate with center crop mask (training) + reflection padding (inference) → val_dice jumped to 0.9098
- Elastic deformation tested but disabled (training unstable without BN)

### ResNet34-UNet
- batch_size=8, 110 epochs, no elastic → Kaggle 0.9182
- batch_size=32, 110 epochs, no elastic → test 0.9133 (worse, batch too large)
- **Best**: batch_size=16, 130 epochs, elastic_p=0.3 → Kaggle 0.9228 (7th place)
