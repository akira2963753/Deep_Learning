# Lab2 — Binary Semantic Segmentation on Oxford-IIIT Pet Dataset

> **Course**: NYCU Deep Learning (Spring 2026)
> **Author**: 林明宏 (M11407439)
> **Task**: 寵物影像的前景/背景二元語義分割 (Binary Foreground Segmentation)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Project Structure](#2-project-structure)
3. [Architecture & Implementation](#3-architecture--implementation)
4. [Training Methodology](#4-training-methodology)
5. [Experiment & Evaluation](#5-experiment--evaluation)
6. [Key Takeaways & Learning Progress](#6-key-takeaways--learning-progress)
7. [Quick Start](#7-quick-start)

---

## 1. Introduction

### 任務描述

本專案針對 **Oxford-IIIT Pet Dataset** 進行二元語義分割（Binary Semantic Segmentation）。給定一張寵物圖片，模型需逐像素預測該像素是否屬於**寵物前景（Foreground）**。

這是一個看似簡單卻充滿挑戰的任務：

- 貓狗毛髮邊緣細節難以精確切割
- 圖片背景多樣（室內、戶外、複雜紋理）
- 遮擋（Occlusion）與姿態多變使模型泛化困難

### 應用場景

| 場景 | 說明 |
|------|------|
| 寵物 App 去背 | 自動將寵物從背景中分離 |
| 醫學影像分割 | 相同技術可遷移至器官輪廓標記 |
| 自動駕駛感知 | 行人/障礙物前景偵測 |
| AR/VR 合成 | 前景物件提取後疊加虛擬場景 |

### Dataset Overview

- **來源**: Oxford Visual Geometry Group (VGG)
- **總數**: ~7,349 張標記影像（37 個品種）
- **Annotation**: Trimap（前景 / 背景 / 邊界三分圖）→ 轉換為二元 Mask
- **解析度**: 統一 resize 至 **384 × 384**
- **Split**: Train 80% / Val 20% / Test (Kaggle 競賽集)

---

## 2. Project Structure

```
Lab2/
├── src/
│   ├── train.py              # 訓練主程式（含 warmup + scheduler + AMP）
│   ├── evaluate.py           # 驗證集評估（含閾值掃描 + TTA）
│   ├── inference.py          # 推論 + RLE 編碼輸出 CSV（Kaggle 格式）
│   ├── oxford_pet.py         # 資料集下載 & Dataset class
│   ├── utils.py              # Loss functions, transforms, pad/crop utilities
│   └── models/
│       ├── unet.py           # 原版 UNet（from scratch）
│       └── resnet34_unet.py  # ResNet34 Encoder + CBAM Decoder
│
├── dataset/
│   └── oxford-iiit-pet/
│       ├── images/           # RGB 輸入影像 (.jpg)
│       └── annotations/
│           ├── trimaps/      # 三分圖標記 (.png)
│           ├── trainval.txt
│           └── test.txt
│
├── saved_models/
│   ├── unet_best.pth
│   ├── unet_last.pth
│   ├── resnet34_unet_best.pth
│   └── resnet34_unet_last.pth
│
├── requirements.txt
└── README.md
```

---

## 3. Architecture & Implementation

本專案實作了兩個架構，分別代表兩種設計哲學：**從零設計的對稱編解碼器**與**借用預訓練特徵提取器的混合架構**。

---

### 3.1 UNet — 對稱式編解碼器

> **設計哲學**: 用 Skip Connection 橋接語意資訊與空間細節，讓深層特徵與淺層輪廓資訊同時服務於分割。

#### 架構圖

```
Input (3×384×384)
    │
    ├─ DoubleConv ──────────────────────────────────────→ Skip₁ (64ch)
    │   ↓ MaxPool
    ├─ DoubleConv ──────────────────────────────────────→ Skip₂ (128ch)
    │   ↓ MaxPool
    ├─ DoubleConv ──────────────────────────────────────→ Skip₃ (256ch)
    │   ↓ MaxPool
    ├─ DoubleConv ──────────────────────────────────────→ Skip₄ (512ch)
    │   ↓ MaxPool
    │
    └─ Bottleneck: DoubleConv (1024ch)
           ↓ ConvTranspose2d + concat(Skip₄) → DoubleConv (512ch)
           ↓ ConvTranspose2d + concat(Skip₃) → DoubleConv (256ch)
           ↓ ConvTranspose2d + concat(Skip₂) → DoubleConv (128ch)
           ↓ ConvTranspose2d + concat(Skip₁) → DoubleConv (64ch)
           ↓
        Conv1×1 → Output (1×384×384)
```

#### 設計細節

| 組件 | 規格 | 設計理由 |
|------|------|---------|
| DoubleConv | Conv3×3 + ReLU + Conv3×3 + ReLU | 兩層卷積增強局部感受野，不使用 BatchNorm（忠於原論文） |
| 上採樣 | ConvTranspose2d（學習式上採樣） | 相比 Bilinear，可學習到上採樣的最佳方式 |
| Skip Connection | Center-crop 對齊尺寸後 Concatenate | 無 Padding 卷積造成特徵圖縮小，需裁切對齊 |
| Reflection Padding | 前處理補邊，後處理裁切回原尺寸 | 解決無 Padding 導致的輸出尺寸不符問題 |

---

### 3.2 ResNet34-UNet — 預訓練編碼器 + CBAM 解碼器

> **設計哲學**: 利用在 ImageNet 上預訓練的 ResNet34 作為特徵提取骨幹，搭配輕量化解碼器，以更少的訓練資源達到更高的分割品質。

#### Encoder：ResNet34 Backbone

```
Input (3×384×384)
    │
    ↓ Conv7×7 (stride=2) + BN + ReLU + MaxPool(stride=2) → 64ch @ 96×96
    │
    ↓ Layer1: 3× BasicBlock → 64ch @ 96×96   (skip → dec3)
    ↓ Layer2: 4× BasicBlock → 128ch @ 48×48  (skip → dec2)
    ↓ Layer3: 6× BasicBlock → 256ch @ 24×24  (skip → dec1)
    ↓ Layer4: 3× BasicBlock → 512ch @ 12×12
```

#### Bottleneck：多尺度特徵融合

```
f3 (256ch @ 24×24) → 下採樣至 12×12
f4 (512ch @ 12×12)
    → Concat → 768ch
    → Conv3×3 → 32ch @ 12×12
```

> **設計亮點**: 在 Bottleneck 融合 Layer3 與 Layer4 的特徵，讓最深的語意資訊（Layer4）能與稍淺的細節（Layer3）互補，同時壓縮到 32 通道以減少後續計算量。

#### Decoder：輕量化 + CBAM 注意力

```
Bottleneck (32ch) → Upsample×2 + concat(skip Layer3) → Conv + BN + ReLU → CBAM → 32ch @ 24×24
                  → Upsample×2 + concat(skip Layer2) → Conv + BN + ReLU → CBAM → 32ch @ 48×48
                  → Upsample×2 + concat(skip Layer1) → Conv + BN + ReLU → CBAM → 32ch @ 96×96
                  → Upsample×2                        → Conv + BN + ReLU → CBAM → 32ch @ 192×192
                  → Upsample×2                        → Conv + BN + ReLU → CBAM → 32ch @ 384×384
                  → Conv1×1 → Output (1×384×384)
```

#### CBAM (Convolutional Block Attention Module)

每個 Decoder Block 後接一個 CBAM，分兩個階段精煉特徵：

```
1. Channel Attention（「哪些特徵通道重要？」）
   AvgPool + MaxPool → shared MLP → Sigmoid → element-wise multiply

2. Spatial Attention（「哪些空間位置重要？」）
   AvgPool & MaxPool along channel dim → Concat → Conv7×7 → Sigmoid → element-wise multiply
```

> **CBAM 的作用**: 讓模型學習在哪裡看（Spatial）以及看什麼（Channel），明確讓注意力聚焦在寵物邊緣與輪廓關鍵區域，有效提升邊界分割精度。

#### 兩種架構的核心差異

| 面向 | UNet | ResNet34-UNet |
|------|------|---------------|
| Encoder | From scratch | ResNet34（ImageNet 預訓練） |
| Bottleneck 通道數 | 1024 | 32 |
| 上採樣方式 | ConvTranspose2d | Bilinear Upsample |
| Normalization | 無 BatchNorm | 每層 BatchNorm |
| 注意力機制 | 無 | CBAM |
| 訓練穩定性 | 較難（無 BN，通道數龐大） | 較穩（BN + 預訓練初始化） |

---

## 4. Training Methodology

### 4.1 資料預處理 — Trimap 轉 Binary Mask

Oxford-IIIT Pet 的原始標記為 **Trimap**，數值含義：

```
1 → Foreground（寵物主體）→ 轉換為 1
2 → Background（背景）    → 轉換為 0
3 → Border/Uncertain（邊界）→ 轉換為 0
```

邊界像素標記為 0（背景）而非忽略，讓模型學習保守地預測邊界，避免過度膨脹預測。

### 4.2 Data Augmentation

增強策略的核心原則：**Image 與 Mask 必須同步變換**，避免幾何增強後影像與標記錯位。以下使用自訂的 `JointTransform` 類別實現。

#### 訓練集增強（Training）

| 增強方式 | 機率 / 參數 | 目的 |
|---------|-----------|------|
| Horizontal Flip | p=0.5 | 對稱不變性（貓臉左右皆存在） |
| Vertical Flip | p=0.5 | 提升姿態多樣性 |
| Rotation | ±20° | 旋轉不變性 |
| ColorJitter | brightness/contrast/saturation ±0.2 | 光照不變性，僅作用於影像 |
| **Elastic Distortion** | p=0.4（僅 ResNet）alpha=80, sigma=10 | 模擬毛髮、軟組織的自然形變 |
| ImageNet Normalize | mean=[0.485,0.456,0.406] | 對齊預訓練權重的輸入分布 |

> **為何 UNet 不用 Elastic Distortion？**
> UNet 無 BatchNorm，通道數龐大，對激進增強較敏感。實驗發現加入 Elastic Distortion 後 UNet 的訓練損失震盪加劇，反而降低最終分數，故關閉（p=0.0）。

#### 驗證 / 測試集增強（Validation & Test）

僅做 Resize（384×384）與 ImageNet Normalize，不施加任何隨機增強，確保評估結果的可重現性。

### 4.3 損失函數 (Loss Function)

```python
Loss = 0.5 × BCE + 0.5 × Dice
```

**為何使用組合損失？**

| Loss | 優點 | 缺點 |
|------|------|------|
| BCE (Binary Cross Entropy) | 逐像素監督，訓練穩定 | 對類別不平衡（背景遠多於邊緣）不敏感 |
| Dice Loss | 直接優化分割重疊率，對類別不平衡魯棒 | 梯度信號不夠精確，單獨使用易震盪 |
| **Combined (0.5+0.5)** | 兼得逐像素精度與形狀匹配 | — |

Dice Loss 實作使用 `smooth=1e-6` 避免分母為零，BCE 使用 `binary_cross_entropy_with_logits`（數值穩定版）。

### 4.4 優化器 (Optimizer)

```
AdamW — lr=1e-4, weight_decay=1e-4
```

選擇 AdamW 而非 Adam 的理由：AdamW 將 Weight Decay 從梯度更新中解耦，實際上等效於 L2 正則化，比 Adam 的 L2 Weight Decay 在實踐中有更好的泛化效果（尤其在有預訓練權重的場景）。

### 4.5 Learning Rate Schedule

#### Warmup（前 5 個 Epoch）

```
LinearLR: 0.1×lr → 1.0×lr（線性遞增）
```

**目的**：AdamW 在訓練初期的動量統計（一、二階矩）尚未穩定，若一開始就使用完整 Learning Rate，梯度更新方向不可靠，容易陷入次優解或震盪。Warmup 讓模型先「小步探索」，再以完整學習率加速收斂。

#### 主要 Schedule

| 模型 | Scheduler | 設定 |
|------|-----------|------|
| **UNet** | `ReduceLROnPlateau` | patience=5, factor=0.5, mode=max（監控 val_dice） |
| **ResNet34-UNet** | `MultiStepLR` | milestones=[50,100,125,150,175,200,215,230], gamma=0.5 |

**選擇邏輯**：
- UNet 訓練較不穩定，ReduceLROnPlateau 能自適應地在驗證指標停滯時降低 LR，避免震盪
- ResNet34-UNet 訓練穩定（有 BatchNorm + 預訓練初始化），MultiStepLR 的預設衰減點可以精確控制訓練節奏

### 4.6 其他訓練技巧

| 技巧 | 設定 | 目的 |
|------|------|------|
| Mixed Precision (AMP) | `torch.amp.autocast("cuda")` + `GradScaler` | 降低顯存占用、加速訓練 |
| Gradient Clipping | `max_norm=1.0` | 防止 Gradient Explosion，尤其在 UNet 深層通道數大時 |
| Model Checkpointing | 保存 best（最高 val_dice）+ last（可續訓） | 確保拿到最佳模型，並支援斷點續訓 |
| Pin Memory | `pin_memory=True` | 加速 CPU→GPU 資料搬運 |

---

## 5. Experiment & Evaluation

### 5.1 評估指標 — Dice Score

$$\text{Dice} = \frac{2 \times |P \cap G|}{|P| + |G|}$$

- $P$：預測前景像素集合（經閾值二值化）
- $G$：Ground Truth 前景像素集合
- 值域 [0, 1]，越高越好

相較於 Pixel Accuracy，Dice Score 對類別不平衡更魯棒（背景像素遠多於前景時，高 Accuracy 不代表好的分割）。

### 5.2 最終成績

| 模型 | Batch Size | Epochs | Val Dice | Test Dice (Kaggle) |
|------|-----------|--------|----------|--------------------|
| **UNet** | 16 | 130 | 0.9148 | 0.91053 |
| **ResNet34-UNet** | 16 | 210 | **0.9366** | **0.92667** |

- Val/Test Dice 差距極小（< 0.01），表示模型泛化良好，無明顯 Overfitting
- ResNet34-UNet 在驗證集與測試集均優於 UNet，驗證了預訓練 + 注意力機制的價值

### 5.3 Test-Time Augmentation (TTA)

推論時使用 4 路 TTA 平均預測結果：

```
Original image          → predict → p₀
Horizontal flip         → predict → flip back → p₁
Vertical flip           → predict → flip back → p₂
Both H+V flip           → predict → flip back → p₃

Final prediction = (p₀ + p₁ + p₂ + p₃) / 4.0
```

TTA 透過集成多種視角的預測，有效降低對單一角度的過擬合，通常可提升 Dice Score 約 0.003~0.008。

### 5.4 閾值掃描 (Threshold Scanning)

模型輸出為 Logit，轉換為 0/1 二值 Mask 時閾值影響分割結果：

```python
# 掃描範圍: 0.30 ~ 0.60（間距 0.05）
for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    dice = compute_dice(predictions > threshold, ground_truth)
```

通常預設閾值 0.5 已表現良好，但在邊界像素偏多的資料集上，0.45 有時可獲更高 Dice。

---

## 6. Key Takeaways & Learning Progress

### 挑戰一：UNet 的輸出尺寸不符問題

**問題**：原版 UNet 的卷積層不使用 Padding，每次 DoubleConv 後特徵圖尺寸縮小，導致輸出與輸入尺寸不同，同時 Skip Connection 也需要 Center-Crop 對齊。

**解法**：在前處理階段加入 **Reflection Padding**，使輸入尺寸能被模型整除；推論後再裁回原始尺寸。使用 Reflection 而非 Zero Padding 是因為邊緣像素的自然延伸減少了邊緣偽影。

```python
# utils.py — 自動計算所需 Padding 量
pad = compute_pad_size(IMAGE_SIZE, model)
output = pad_and_crop(images, model, pad, IMAGE_SIZE)
```

---

### 挑戰二：UNet 訓練震盪 (Oscillation)

**問題**：UNet 沒有 BatchNorm，且 Bottleneck 通道數高達 1024，梯度在反向傳播時容易出現震盪，Loss 難以平穩下降。

**解法組合**：
1. **LinearLR Warmup**：前 5 個 Epoch 小 LR 穩定動量統計
2. **Gradient Clipping (max_norm=1.0)**：截斷過大梯度
3. **ReduceLROnPlateau**：自適應降低 LR，在停滯時快速調整

---

### 挑戰三：ResNet34-UNet 通道壓縮設計

**問題**：如何在使用 ResNet34（512ch 深層特徵）的同時，避免解碼器因通道數過多而計算量暴增？

**解法**：Bottleneck 直接將 Layer3 + Layer4 融合後壓縮至 **32 通道**，整個解碼器均維持 32 通道。這個設計讓 GPU 記憶體佔用大幅降低，同時迫使模型學習更 Compact 的特徵表示。

---

### 挑戰四：Elastic Distortion 的雙面性

**發現**：Elastic Distortion 對 ResNet34-UNet 有效，但對 UNet 有害。

**分析**：
- ResNet34-UNet 有 BatchNorm 穩定訓練，有更多參數（ResNet 骨幹），能從更多樣的增強中學到更好的泛化
- UNet 無 BatchNorm 且本身訓練已較不穩定，激進的幾何增強進一步加劇困難

**教訓**：增強策略需針對具體架構設計，沒有放諸四海皆準的最佳增強組合。

---

### 挑戰五：不同模型選擇不同 LR Scheduler

**發現**：對 ResNet34-UNet 使用 ReduceLROnPlateau 效果反而不如 MultiStepLR。

**分析**：ResNet34-UNet 配備 BatchNorm，訓練非常穩定，Val Dice 幾乎單調遞增。ReduceLROnPlateau 的 patience 機制在此情境下反應過慢，而 MultiStepLR 的預定 Milestone 能讓 LR 在合適的訓練節奏下衰減，推動模型繼續精進。

**教訓**：LR Scheduler 的選擇應與模型穩定性匹配，穩定訓練適合預定節奏，震盪訓練適合自適應策略。

---

### 整體心得

```
UNet    → 簡潔的「學術原型」，理解 Encoder-Decoder + Skip Connection 的核心思想
ResNet  → 工程化的「實戰升級」，展示如何用預訓練 + 注意力機制系統性提升表現
```

這個 Lab 最大的收穫不是分數，而是**理解為什麼每個設計選擇會帶來什麼效果**。從 Warmup 的原理到 CBAM 的作用，每一個超參數背後都有清晰的邏輯可循。

---

## 7. Quick Start

### 環境安裝

```bash
pip install -r requirements.txt
```

### 資料集下載

```bash
python -m src.oxford_pet
```

### 訓練

```bash
# UNet（130 epochs）
python src/train.py --model unet --epochs 130 --batch_size 16 \
  --data_root dataset/oxford-iiit-pet

# ResNet34-UNet（210 epochs）
python src/train.py --model resnet34_unet --epochs 210 --batch_size 16 \
  --data_root dataset/oxford-iiit-pet
```

### 評估（含閾值掃描 + TTA）

```bash
python src/evaluate.py --model resnet34_unet \
  --checkpoint saved_models/resnet34_unet_best.pth \
  --data_root dataset/oxford-iiit-pet \
  --scan_threshold --tta
```

### 推論輸出（Kaggle CSV）

```bash
# UNet
python src/inference.py --model unet \
  --data_root dataset/oxford-iiit-pet \
  --checkpoint saved_models/unet_best.pth \
  --test_list <test_list> \
  --output unet_pred.csv --threshold 0.5 --tta

# ResNet34-UNet
python src/inference.py --model resnet34_unet \
  --data_root dataset/oxford-iiit-pet \
  --checkpoint saved_models/resnet34_unet_best.pth \
  --test_list <test_list> \
  --output resnet34_unet_pred.csv --threshold 0.5 --tta
```

---

*Generated with insights from hands-on implementation experience.*
