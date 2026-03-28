# ResNet34-UNet Project — Code Review 複習筆記

> 資料集：Oxford-IIIT Pet Dataset（貓狗語義分割）
> 任務：Binary Segmentation（前景 / 背景）
> 與 Plain UNet 的核心差異：Encoder 換成 ResNet34（手刻，非 pretrained）

---

## 一、模型架構：`resnet34_unet.py`

### 整體概念

ResNet34-UNet 的結構與 Plain UNet 相同，都是 Encoder-Bottleneck-Decoder 的 U 型架構，但 **Encoder 換成 ResNet34**，帶來更深的網路、殘差連接（Residual Connection）以及 BatchNorm。

---

### Building Block 1：`DoubleConv`（改良版）

與 Plain UNet 的 `DoubleConv` 相比多了兩個東西：

```python
Conv2d(3x3, padding=1, bias=False) → BatchNorm2d → ReLU →
Conv2d(3x3, padding=1, bias=False) → BatchNorm2d → ReLU
```

**`padding=1` 的作用：**
讓 3x3 卷積後的空間尺寸維持不變，不需要像 Plain UNet 那樣補 reflection padding，也不需要 center crop。

**`BatchNorm2d` 的作用：**
對每個 batch 的 feature map 做正規化，穩定訓練、加速收斂，並有輕微的正則化效果。

**為什麼 `bias=False`？**
BatchNorm 本身包含一個可學習的 shift 參數（β），功能等同於 bias。如果卷積層同時有 bias，兩者重複，bias 是多餘的，拿掉可以節省參數。

---

### Building Block 2：`BasicBlock`（ResNet 的核心）

這是 ResNet 的核心設計，結構如下：

```
input (x)
  ↓
Conv2d(3x3, stride) → BN → ReLU → Conv2d(3x3) → BN
  ↓
out += shortcut(x)   ← 殘差連接
  ↓
ReLU(out)
```

#### 為什麼需要 Residual Connection（殘差連接）？

深層網路訓練時，gradient 從後面層往前傳遞，每經過一層就可能衰減（gradient vanishing）。ResNet 的 shortcut 讓 gradient 可以「跳過」幾層直接流回去，大幅緩解這個問題，讓更深的網路得以訓練。

#### Shortcut 什麼時候需要 1x1 Conv？

```python
self.shortcut = nn.Sequential()
if stride != 1 or in_channels != out_channels:
    self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    )
```

當 `stride != 1`（解析度縮小）或 `in_channels != out_channels`（channel 數改變）時，input `x` 的 shape 和 `out` 的 shape 不一樣，無法直接相加。此時用 1x1 Conv 把 input 的 shape 對齊成跟 output 一樣，才能做殘差相加。

---

### Building Block 3：`_make_layer`（ResNet Layer 工廠）

```python
def _make_layer(in_channels, out_channels, num_blocks, stride=1):
    layers = [BasicBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)
```

每個 layer 由多個 BasicBlock 串成：
- **第一個 Block** 負責改變 channel 數和解析度（stride 可能 != 1）
- **後面的 Block** 保持 channel 和解析度不變（stride=1）

ResNet34 的 layer 配置固定為：

| Layer | in/out channels | num_blocks | stride |
|-------|----------------|------------|--------|
| layer1 | 64 → 64 | 3 | 1 |
| layer2 | 64 → 128 | 4 | 2 |
| layer3 | 128 → 256 | 6 | 2 |
| layer4 | 256 → 512 | 3 | 2 |

`[3, 4, 6, 3]` 是 ResNet34 論文原始的設計，總共 34 層（計算卷積層數）。

---

### Building Block 4：`DecoderBlock`（Decoder 的上採樣）

```python
ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
→ 尺寸不整除時用 bilinear interpolate 對齊
→ torch.cat([x, skip], dim=1)
→ DoubleConv(in_channels + skip_channels, out_channels)
```

與 Plain UNet 的 `Up` 相比有一個關鍵差異：

**Plain UNet：** 用 `_center_crop` 裁切 skip feature map 來對齊尺寸
**ResNet34-UNet：** 用 `F.interpolate(bilinear)` 把 x 調整到跟 skip 一樣大

```python
if x.shape[2:] != skip.shape[2:]:
    x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
```

為什麼要做這個處理？ResNet34 的 Encoder 有 7x7 conv（stride=2）和多次 stride，當輸入不是特定倍數時，上採樣後尺寸可能差 1 個 pixel，bilinear interpolate 可以彈性對齊到目標尺寸。

---

### Forward 流程總覽

```
Input (3ch, H×W)
  conv1(7x7, stride=2) → BN → ReLU    → s4 (64ch, H/2×W/2)
  maxpool(stride=2)                    → (64ch, H/4×W/4)
  layer1(3 blocks, stride=1)           → s3 (64ch,  H/4×W/4)
  layer2(4 blocks, stride=2)           → s2 (128ch, H/8×W/8)
  layer3(6 blocks, stride=2)           → s1 (256ch, H/16×W/16)
  layer4(3 blocks, stride=2)           →  x (512ch, H/32×W/32)

  dec1: DecoderBlock(512, 256, 256) ← skip: s1 (256ch)
  dec2: DecoderBlock(256, 128, 128) ← skip: s2 (128ch)
  dec3: DecoderBlock(128,  64,  64) ← skip: s3 (64ch)
  dec4: DecoderBlock( 64,  64,  32) ← skip: s4 (64ch)

  F.interpolate(scale_factor=2, bilinear)  ← 補回 conv1 的 stride=2
  out_conv: Conv2d(32 → 1, kernel=1)       → Output Mask (1ch)
```

#### 為什麼最後要再 upsample 一次？

Encoder 入口 conv1 有 stride=2，加上 maxpool 又縮了一半，等於一開始就縮了 4 倍。Decoder 的 4 個 DecoderBlock 逐步放大，但只恢復了 layer1～layer4 的縮小倍率，conv1 的那一次縮小沒有被恢復。最後這個 `interpolate(scale_factor=2)` 就是把那個缺的 2 倍補回來，讓輸出解析度和輸入一致。

---

### Plain UNet vs ResNet34-UNet 架構對比

| | Plain UNet | ResNet34-UNet |
|--|------------|---------------|
| Encoder 設計 | MaxPool + DoubleConv | ResNet34 BasicBlock |
| Residual Connection | ❌ | ✅ |
| BatchNorm | ❌ | ✅ |
| Conv padding | ❌（需要補邊） | ✅ padding=1 |
| Skip 對齊方式 | center crop | bilinear interpolate |
| 最後 upsample | ❌ | ✅（補回 conv1 stride） |
| 深度（大致） | 淺 | 深（34 層卷積） |

---

### Code Review 常見問題

| 問題 | 回答 |
|------|------|
| 為什麼要有 Residual Connection？ | 解決深層網路 gradient vanishing，讓更深的網路得以訓練 |
| Shortcut 什麼時候需要 1x1 Conv？ | stride!=1 或 channel 數改變，shape 不同無法直接相加 |
| 為什麼 bias=False？ | BatchNorm 的 shift 參數已有 bias 效果，重複設定是多餘的 |
| 為什麼用 bilinear 而非 center crop 對齊 skip？ | bilinear 更彈性，不受固定尺寸限制 |
| 最後為什麼要多一次 upsample？ | 補回 conv1(stride=2) 造成的縮小，讓輸出和輸入解析度一致 |
| ResNet34 的 [3,4,6,3] 是什麼？ | 各 layer 的 BasicBlock 數量，來自 ResNet34 論文原始設計 |
| 這樣算 pretrained 嗎？ | **不算**，ResNet34 是手刻的，所有權重都是從頭隨機初始化訓練 |

---

## 二、資料集處理：`oxford_pet.py`

ResNet34-UNet 使用完全相同的資料集處理邏輯，以下列出重點，詳細說明參考 Plain UNet 筆記。

### Trimap → Binary Mask

Oxford Pet 的 mask 是 trimap（三值）：

| 值 | 意義 |
|----|------|
| 1 | 前景（寵物本體） |
| 2 | 背景 |
| 3 | 邊界（不確定區域） |

`_trimap_to_binary()` 只保留 `mask == 1` 的地方為 1，其他全部變 0，邊界直接視為背景。

### 資料集切分

- 固定 `seed=42` 從 `trainval.txt` 切 8:2 成 train / val
- `test.txt` 用官方提供的

### Transform 設計

**Image：** `Resize → ToTensor → Normalize(ImageNet mean/std)`
**Mask：** `Resize(NEAREST) → PILToTensor`

Mask 用 `NEAREST` 插值，避免產生非 0/1 的中間值。

### Code Review 常見問題

| 問題 | 回答 |
|------|------|
| 為什麼 seed=42？ | 確保每次資料切分一致，實驗可重現 |
| 為什麼 mask Resize 用 NEAREST？ | 避免插值破壞 0/1 整數 label |
| 用 ImageNet mean/std 算 pretrained 嗎？ | **不算**，只是數值縮放，沒有載入外部權重 |

---

## 三、訓練流程：`train.py`（ResNet34 專屬差異）

大部分訓練邏輯與 Plain UNet 共用，以下只說明 ResNet34-UNet 特有的差異。

### Elastic Augmentation 只對 ResNet34 開啟

```python
elastic_p = 0.3 if args.model == "resnet34_unet" else 0.0
```

ResNet34 模型較深、參數量更多，容量大，不加強 augmentation 容易 overfit。Plain UNet 較淺，augmentation 太強反而讓模型學不到東西。

### Padding 自動偵測，ResNet34 回傳 0

```python
pad = compute_pad_size(IMAGE_SIZE, model)
```

Plain UNet 沒有 padding，需要先補 reflection padding 再 forward。ResNet34-UNet 所有卷積都有 `padding=1`，輸入輸出尺寸一致，`compute_pad_size` 回傳 0，直接 forward 不需要任何補丁。

### Checkpoint 命名

| 類型 | 檔名 |
|------|------|
| Best model | `resnet34_unet_best.pth` |
| Last checkpoint（resume 用） | `resnet34_unet_last.pth` |

### 共用的訓練設定（與 Plain UNet 相同）

- **Optimizer**：AdamW（lr=1e-4, weight_decay=1e-4）
- **Scheduler**：ReduceLROnPlateau（patience=5, factor=0.5, mode="max"）
- **AMP**：autocast + GradScaler（F16 forward/backward，F32 weight update）
- **Gradient Clipping**：max_norm=1.0
- **Seed**：random / numpy / torch / cuda 全部固定為 42

### Code Review 常見問題

| 問題 | 回答 |
|------|------|
| 為什麼 ResNet34 不需要 reflection padding？ | 所有卷積都有 padding=1，尺寸不縮小 |
| 為什麼 elastic augmentation 只對 ResNet34 開？ | 模型越強越需要更強的正則化避免 overfit |
| 為什麼用 AdamW 不用 Adam？ | AdamW 的 weight decay 不被 adaptive lr 縮放，正則化效果更穩定 |
| 為什麼要 GradScaler + unscale_？ | F16 gradient 可能 underflow；unscale_ 讓 clip_grad_norm_ 看到真實梯度大小 |
| 為什麼存兩個 checkpoint？ | best 給 inference 用，last 給 --resume 中斷續訓用 |

---

## 四、Loss 與 Augmentation：`utils.py`

Loss Function 邏輯與 Plain UNet 完全相同，Augmentation 的差別只有 elastic deformation 的開關。

### Loss Function

#### Dice Loss

```python
pred = torch.sigmoid(pred)
intersection = (pred * target).sum()
return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
```

衡量預測區域和真實區域的重疊比例，對類別不平衡友善。`smooth=1e-6` 防止分母為 0。

#### Combined Loss

```python
0.5 * BCE + 0.5 * Dice
```

BCE 提供穩定的 pixel-level gradient，Dice 確保整體形狀的重疊好，兩者互補。

#### `dice_loss` vs `dice_score` 的差別

| | `dice_loss` | `dice_score` |
|--|-------------|--------------|
| 用途 | 訓練 | 評估 |
| Threshold | ❌ sigmoid 後連續值 | ✅ >0.5 轉 binary |
| 原因 | 需要連續 gradient backprop | 反映真實推論時的行為 |

---

### Data Augmentation：`JointTransform`

Image 和 Mask 必須套用**完全相同的幾何變換**，`JointTransform` 先抽 random 參數再同時套到兩者。

#### 各種 Augmentation 說明

**Random Flip / Rotation**
- Mask 一律用 `NEAREST` interpolation 保持整數 label

**Color Jitter**
- 只對 image 做，不對 mask 做
- 顏色與 mask 的空間標記無關

**Elastic Deformation（ResNet34 專屬，`elastic_p=0.3`）**

模擬毛皮、組織的自然形變，步驟：
1. 產生兩個隨機 noise field（dx, dy）
2. 用 Gaussian filter 平滑化，讓形變看起來自然
3. 用位移場重新 sample 每個 pixel 的來源（`map_coordinates`）

| | interpolation order | 原因 |
|--|---------------------|------|
| image | `order=1`（雙線性） | 圖片視覺平滑 |
| mask | `order=0`（nearest） | 保持 0/1，不產生中間值 |

Plain UNet 的 `elastic_p=0.0`，等於關閉。ResNet34 的 backbone 較強，需要更強的 augmentation 抑制 overfitting。

---

### Code Review 常見問題

| 問題 | 回答 |
|------|------|
| 為什麼用 combined loss？ | BCE 逐 pixel 穩定，Dice 抗類別不平衡，兩者互補 |
| dice_loss 為什麼不做 threshold？ | 連續值才有 gradient，才能 backprop |
| Elastic deformation image 和 mask 的 order 為什麼不同？ | image 要視覺平滑，mask 要保持整數 label |
| 為什麼 Color Jitter 不對 mask 做？ | 顏色不影響空間位置，mask 是空間標記 |
| Elastic deformation 對 ResNet34 開 0.3 的意義？ | 30% 機率觸發，不是每張圖都做，避免 augmentation 過強 |

---

## 五、評估：`evaluate.py`

### 整體功能

在 **val set** 上載入訓練好的 checkpoint，計算 Dice Score，確認模型效果。兩個模型的評估邏輯完全相同，只差在載入哪個 checkpoint。

---

### Threshold Scan（`--scan_threshold`）

```python
for t in [round(x * 0.05, 2) for x in range(6, 13)]:
    # 掃描 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
```

預設 threshold 是 0.5，但不一定是最佳值。做法：

1. 先把整個 val set 的預測全部收集起來（`all_preds`）
2. 再一次掃描多個 threshold，分別算 Dice
3. 找出 val Dice 最高的 threshold

**為什麼先收集全部預測再掃？** 每換一個 threshold 不需要重新 forward，只需重新算 Dice，效率高很多。

---

### Padding 邏輯

`evaluate.py` 也有 `compute_pad_size`，與 `train.py` 一樣的邏輯：ResNet34-UNet 回傳 0，直接 forward 不補邊。

---

### Code Review 常見問題

| 問題 | 回答 |
|------|------|
| 為什麼先收集所有預測再掃 threshold？ | 避免重複 forward，節省計算資源 |
| 為什麼掃 0.30～0.60 這個範圍？ | 0.5 附近合理，太極端的 threshold 通常效果差 |
| evaluate 用 val set 而不是 test set？ | test set 沒有 annotation，只能用 val set 評估 |
| ResNet34 和 UNet 用一樣的評估邏輯嗎？ | 是，差別只在 checkpoint 和是否需要 reflection padding |

---

## 六、推論：`inference.py`

### 整體功能

對 **test set** 做推論，輸出 CSV 供 Kaggle 提交。兩個模型的推論邏輯完全相同。

---

### RLE Encoding（`rle_encode`）

Kaggle 的 mask 提交格式是 **Run-Length Encoding（RLE）**，用「起始位置 + 長度」表示連續的 1，大幅壓縮資料量。

```python
mask = mask.flatten(order="F")   # column-major 展開
```

兩個重要細節：
- `flatten(order="F")`：**column-major**（先掃第一欄再第二欄），Kaggle 規定的格式
- **1-indexed**：起始位置從 1 開始算，不是 0

---

### TTA（Test-Time Augmentation，`--tta`）

```python
p0 = _forward(images)                               # 原圖
p1 = flip(_forward(flip(images, [3])), [3])         # 水平翻轉後預測，再翻回來
p2 = flip(_forward(flip(images, [2])), [2])         # 垂直翻轉後預測，再翻回來
p3 = flip(_forward(flip(images, [2, 3])), [2, 3])  # 水平 + 垂直都翻
preds = (p0 + p1 + p2 + p3) / 4.0
```

同一張圖用 4 種翻轉方式各跑一次，把預測機率平均。模型可能對某個方向有偏差，平均多個角度可以降低這個偏差，是競賽中常見的技巧。

---

### Resize 回原始大小

```python
mask_pil.resize((ow, oh), PILImage.NEAREST)
```

輸入統一 resize 到 384x384，inference 後要把 mask 還原回原始尺寸再做 RLE encode。用 `NEAREST` 保持 0/1 整數值。

---

### Code Review 常見問題

| 問題 | 回答 |
|------|------|
| RLE 為什麼要 column-major？ | Kaggle 規定的格式 |
| RLE 為什麼 1-indexed？ | Kaggle 規定，從 1 開始計數 |
| TTA 為什麼能提升效果？ | 平均多角度的預測，降低模型的方向性偏差 |
| Resize 回原始大小為什麼用 NEAREST？ | 保持 mask 的 0/1 整數值不被插值破壞 |
| 為什麼輸出 CSV 而不是圖片？ | RLE 格式體積小，是 Kaggle 的標準提交格式 |
| ResNet34 和 UNet 的推論流程一樣嗎？ | 是，差別只在 checkpoint 和是否需要 reflection padding |
