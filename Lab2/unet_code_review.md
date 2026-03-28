# UNet Project — Code Review 複習筆記

> 資料集：Oxford-IIIT Pet Dataset（貓狗語義分割）
> 任務：Binary Segmentation（前景 / 背景）

---

## 一、模型架構：`unet.py`

### 整體概念

UNet 是語義分割最經典的架構，結構形如一個 **U 字型**：

```
Input
  → Encoder（解析度縮小、channel 增加）
  → Bottleneck（最深處）
  → Decoder（解析度恢復）
  → Output Mask
```

---

### Building Block 1：`DoubleConv`

```python
Conv2d(3x3) → ReLU → Conv2d(3x3) → ReLU
```

- 每次 3x3 卷積（無 padding）會讓空間尺寸縮小 2 pixel
- 因此每過一個 DoubleConv，feature map 會比輸入稍微小一點
- 這也是後面 Skip Connection 需要做 center crop 的根本原因

---

### Building Block 2：`Down`（Encoder 下採樣）

```python
MaxPool2d(2x2) → DoubleConv
```

- MaxPool 讓解析度減半
- channel 數同時翻倍：`64 → 128 → 256 → 512 → 1024`
- 解析度下降但 channel 增加，模型在空間細節減少的同時保留更豐富的語意資訊

---

### Building Block 3：`Up`（Decoder 上採樣 + Skip Connection）

三個步驟：

1. `ConvTranspose2d` — 解析度放大 2 倍，channel 砍半
2. `_center_crop` — 把 Encoder 對應層的 feature map 從中心裁切，對齊 Decoder 的尺寸
3. `torch.cat` — 在 channel 維度把兩者 concat 在一起

#### 為什麼需要 Skip Connection？

Encoder 下採樣的過程中會損失空間細節（例如物體邊緣）。Decoder 雖然能恢復解析度，但細節已經丟失。Skip Connection 把 Encoder 各層的 feature map 「直接接回」對應的 Decoder 層，讓 Decoder 同時擁有：

- **高層語意**（來自 Bottleneck）
- **低層細節**（來自 Encoder 的 skip）

這樣分割的邊界才會準確。

#### 為什麼用 center crop 而非 padding？

因為沒有 padding 的卷積會讓 feature map 略縮小，Encoder 和 Decoder 的尺寸會差幾個 pixel。Center crop 從中間裁切讓兩者對齊，比補零更自然。

---

### Forward 流程總覽

```
Input (3ch)
  enc1: DoubleConv  →  s1 (64ch)
  enc2: Down        →  s2 (128ch)
  enc3: Down        →  s3 (256ch)
  enc4: Down        →  s4 (512ch)
  bottleneck: Down  →  b  (1024ch)

  dec4: Up(b,  s4)  →  (512ch)
  dec3: Up(x,  s3)  →  (256ch)
  dec2: Up(x,  s2)  →  (128ch)
  dec1: Up(x,  s1)  →  (64ch)

  out_conv: Conv2d(1x1) → Output Mask (1ch)
```

最後的 `1x1 Conv` 把 64 個 channel 壓成 1 個 channel，每個 pixel 輸出一個 logit，代表該 pixel 屬於前景的預測值。

---

### Code Review 常見問題

| 問題 | 回答 |
|------|------|
| 為什麼 channel 數要翻倍？ | 解析度降低時用更多 channel 保留語意資訊 |
| Skip Connection 的作用？ | 把 Encoder 的低層細節傳給 Decoder，補回下採樣損失的空間資訊 |
| 為什麼用 center crop？ | 無 padding 的卷積讓 feature map 略縮，center crop 對齊尺寸 |
| 為什麼輸出用 1x1 Conv？ | 把多個 channel 的特徵整合成最終的 1 個 channel mask |

---

## 二、資料集處理：`oxford_pet.py`

### Trimap 是什麼？

Oxford Pet 的 mask 不是一般 binary，而是 **trimap**（三值）：

| 值 | 意義 |
|----|------|
| 1 | 前景（寵物本體） |
| 2 | 背景 |
| 3 | 邊界（不確定區域） |

`_trimap_to_binary()` 做的事：只有 `mask == 1` 的地方保留為 1，其他（包含邊界）全部變 0，簡化成 binary 分割問題。

---

### 資料集切分

`_get_splits()` 的做法：

- 從官方 `trainval.txt` 拿全部名單
- **固定 seed=42** 做 shuffle 後切 8:2 成 train / val
- `test.txt` 直接用官方提供的

固定 seed 的目的：確保每次執行切出來的資料完全相同，讓實驗結果具有可重現性與可比性。

---

### Transform 設計重點

Image 和 Mask 使用**不同的 transform**：

#### Image Transform

```
Resize → ToTensor → Normalize(ImageNet mean/std)
```

- **ToTensor**：PIL Image 轉 Tensor，shape 從 `(H, W, C)` 變 `(C, H, W)`，值從 `0~255` 縮放到 `0.0~1.0`
- **Normalize**：套用公式 `(pixel - mean) / std`，輸出大約落在 `-2 ~ +2` 的範圍

為什麼用 ImageNet 的 mean/std？若 Encoder 使用 ImageNet pretrained 的 backbone，該模型訓練時的輸入就是用這組數字正規化的。使用相同的正規化方式，讓模型看到的輸入分布與 pretrained 時一致，能更有效利用預訓練權重。即使不用 pretrained，這也是一個合理的正規化範圍選擇。

#### Mask Transform

```
Resize(NEAREST) → PILToTensor
```

關鍵在 `InterpolationMode.NEAREST`。若用雙線性插值（Bilinear），0 和 1 之間會出現 0.3、0.7 等中間值，破壞 mask 的整數 label。NEAREST 直接取最近鄰，確保 mask 的值只有 0 和 1。

---

### Test Set 的特殊處理

Oxford 官方 test 不提供 annotation，所以 `__getitem__` 在 test split 時回傳：

```python
return image, name   # 而非 image, mask
```

`name` 是圖片檔名，inference 時用來對應輸出結果。

---

### Code Review 常見問題

| 問題 | 回答 |
|------|------|
| 為什麼 seed=42？ | 確保每次資料切分一致，實驗可重現 |
| 邊界（trimap=3）為什麼丟掉？ | 邊界模糊不確定，直接視為背景，簡化問題 |
| 為什麼 mask 用 NEAREST？ | 避免插值產生非整數值，破壞 label |
| 為什麼用 ImageNet normalization？ | 配合 pretrained backbone 的輸入分布 |
| 單純用 ImageNet mean/std 算 pretrained 嗎？ | **不算**，pretrained 是指載入別人的權重，normalization 只是數值縮放 |

---

## 三、訓練流程：`train.py`

### 固定所有 Random Seed

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

Python、NumPy、PyTorch CPU、PyTorch GPU 各自有獨立的亂數系統，四個都要設才能完整固定隨機性。

---

### Padding 策略：`compute_pad_size` / `pad_and_crop`

原始 UNet 沒有 padding，輸出解析度會小於輸入。解法：

1. 丟一個 dummy input 進去，量出輸出比輸入少多少
2. 訓練時先用 **reflection padding** 把輸入補大
3. 模型跑完後 crop 回原始大小

用 `reflect` 而非補零的原因：reflection 是鏡像邊緣，視覺上自然，不會引入人工黑邊雜訊。

ResNet34-UNet 有 padding 的卷積，`compute_pad_size` 回傳 0，這段邏輯直接跳過。

---

### Optimizer：AdamW

```python
optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

AdamW 與 Adam 的差別：Adam 的 weight decay 會被 adaptive learning rate 縮放掉，效果不一致；AdamW 把 weight decay 獨立出來直接作用在參數上，正則化效果更穩定，是目前的主流選擇。

---

### Scheduler：ReduceLROnPlateau

```python
optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
```

- 監控 `val_dice`，連續 5 個 epoch 沒進步就把 lr 乘以 0.5
- `mode="max"` 因為 Dice 越高越好
- 自適應的學習率策略，比固定 StepLR 更智慧，不需要手動猜何時降 lr

---

### Mixed Precision Training（AMP）

```python
scaler = torch.amp.GradScaler("cuda")

with torch.amp.autocast("cuda"):
    preds = pad_and_crop(images, model, pad, IMAGE_SIZE)
    loss  = combined_loss(preds, masks)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

對應 NVIDIA Mixed Precision Training 論文的流程：

| 論文概念 | 程式碼對應 |
|---------|-----------|
| Master-Weights 存 F32 | PyTorch 預設 model 參數為 F32 |
| FWD / BWD 用 F16 | `autocast` 自動 cast |
| Loss Scaling 防 underflow | `scaler.scale(loss)` 放大 loss |
| Weight Update 用 F32 | `scaler.step(optimizer)` unscale 後更新 |

`unscale_` 的目的：讓 `clip_grad_norm_` 看到真實的 gradient 大小，否則 clip 的 threshold 會失準。

---

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

把所有參數的 gradient L2 norm 限制在 1.0 以內，防止某 batch 的 loss 特別大導致 gradient 爆炸，讓訓練崩掉。

---

### Checkpoint 儲存策略

儲存兩個檔案，目的不同：

| 檔案 | 內容 | 用途 |
|------|------|------|
| `unet_best.pth` | 只有 `model.state_dict()` | Inference 使用 |
| `unet_last.pth` | model + optimizer + scheduler + scaler + epoch + best_dice | `--resume` 中斷續訓 |

---

### Code Review 常見問題

| 問題 | 回答 |
|------|------|
| 為什麼用 AdamW 不用 Adam？ | weight decay 實作方式不同，AdamW 更穩定 |
| 為什麼 reflection padding？ | 比補零更自然，不引入人工邊緣 |
| 為什麼要 GradScaler + unscale_？ | float16 gradient 可能 underflow，unscale_ 讓 clip 準確 |
| 為什麼存兩個 checkpoint？ | best 給 inference，last 給 resume |
| ReduceLROnPlateau 的 patience 是什麼？ | 幾個 epoch 沒進步才降 lr，避免過早降 lr |

---

## 四、Loss 與 Augmentation：`utils.py`

### Loss Function

#### Dice Loss

```python
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
```

衡量預測區域與真實區域的重疊程度，對**類別不平衡**友善。前景 pixel 少時，BCE 傾向預測全背景仍有高 accuracy，Dice 不會有這個問題。

`smooth=1e-6`：防止分母為 0，讓 gradient 更穩定。

#### BCE Loss

對每個 pixel 獨立計算預測機率與 label 的差距，提供穩定且 pixel-level 的 gradient。

#### Combined Loss（`combined_loss`）

```python
0.5 * BCE + 0.5 * Dice
```

兩者各取 0.5 結合：BCE 提供穩定的逐 pixel gradient，Dice 確保整體形狀的重疊好，互補兩者優點，是分割任務的常見做法。

---

### `dice_loss` vs `dice_score` 的差別

| | `dice_loss` | `dice_score` |
|--|-------------|--------------|
| 用途 | 訓練 | 評估 |
| Threshold | ❌ 不做，用 sigmoid 後的連續值 | ✅ 先 >0.5 轉 binary |
| 原因 | 需要連續 gradient 讓 backprop 流通 | 要反映真實推論時的行為 |

---

### Data Augmentation：`JointTransform`

Image 和 Mask 必須套用**完全相同的幾何變換**，否則 mask 對不上圖。`JointTransform` 先抽 random 參數，再同時套到兩者。

#### 各種 Augmentation 的設計細節

**Random Flip / Rotation**
- Mask 一律用 `NEAREST` interpolation，保持整數 label

**Color Jitter**
- **只對 image 做，不對 mask 做**
- 顏色變化不影響空間位置，mask 是空間標記，與顏色無關

**Elastic Deformation（來自 Simard et al., 2003，UNet 原論文也有提）**

模擬毛皮、組織的自然形變，步驟：

1. 產生兩個隨機 noise field（dx, dy）
2. 用 **Gaussian filter 平滑化**，讓形變看起來自然而非雜亂
3. 用位移場重新 sample 每個 pixel 的來源位置（`map_coordinates`）

| | interpolation order | 原因 |
|--|---------------------|------|
| image | `order=1`（雙線性） | 讓圖片平滑、視覺自然 |
| mask | `order=0`（nearest） | 保持 0/1，不產生中間值 |

Elastic Deformation 只在 `resnet34_unet` 訓練時啟用（`elastic_p=0.3`），plain UNet 為 `elastic_p=0.0`。ResNet34 backbone 較強，搭配更強的 augmentation 可避免 overfitting；plain UNet 較弱，過強的 augmentation 反而讓模型學不好。

---

### Code Review 常見問題

| 問題 | 回答 |
|------|------|
| 為什麼用 combined loss？ | BCE 逐 pixel 穩定，Dice 抗類別不平衡，兩者互補 |
| dice_loss 為什麼不做 threshold？ | 連續值才有 gradient，才能 backprop |
| dice_score 為什麼要做 threshold？ | 反映真實推論行為 |
| Elastic deformation 的 image 和 mask order 為什麼不同？ | image 要視覺平滑，mask 要保持整數 label |
| 為什麼 Color Jitter 不對 mask 做？ | 顏色與 mask 的空間標記無關 |
| 為什麼 Elastic deformation 只對 ResNet34 開啟？ | backbone 較強，需要更強的正則化避免 overfitting |

---

## 五、評估：`evaluate.py`

### 整體功能

在 **val set** 上載入訓練好的 checkpoint，計算 Dice Score，確認模型效果。

---

### Threshold Scan（`--scan_threshold`）

```python
for t in [round(x * 0.05, 2) for x in range(6, 13)]:
    # 掃描 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
```

預設 threshold 是 0.5，但不一定是最佳值。`--scan_threshold` 的做法：

1. 先把整個 val set 的預測結果全部收集起來（`all_preds`）
2. 再一次掃描多個 threshold，分別計算 Dice Score
3. 找出 val Dice 最高的 threshold

**為什麼先收集全部預測再掃？** 這樣每換一個 threshold 不需要重新跑 forward，只需重新算 Dice，效率高很多。

---

### Code Review 常見問題

| 問題 | 回答 |
|------|------|
| 為什麼先收集所有預測再掃 threshold？ | 避免重複 forward，節省計算資源 |
| 為什麼掃 0.30～0.60 這個範圍？ | 0.5 附近合理，太極端的 threshold 通常效果差 |
| evaluate 用 val set 而不是 test set？ | test set 沒有 annotation，只能用 val set 評估 |

---

## 六、推論：`inference.py`

### 整體功能

對 **test set** 做推論，把預測結果輸出成 CSV 檔供 Kaggle 提交。

---

### RLE Encoding（`rle_encode`）

Kaggle 的 mask 提交格式是 **Run-Length Encoding（RLE）**，不是直接上傳圖片，原因是圖片太大、數量太多。

RLE 概念：把連續的 1 用「起始位置 + 長度」來表示。例如從第 10 個 pixel 開始有 5 個 1，就記成 `10 5`，大幅壓縮資料量。

兩個重要細節：

- `flatten(order="F")`：按 **column-major**（先掃第一欄再第二欄）展開，這是 Kaggle 規定的格式
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

同一張圖用 4 種翻轉方式各跑一次，把預測機率平均起來。模型對某個方向可能有偏差，平均多個角度可以降低這個偏差，提升預測穩定性，是競賽中常見的技巧。

---

### Resize 回原始大小

```python
mask_pil.resize((ow, oh), PILImage.NEAREST)
```

輸入時把圖統一 resize 到 384x384，inference 結束後要把 mask 還原回原始圖片的尺寸再做 RLE encode。同樣用 `NEAREST`，保持 mask 的 0/1 整數值。

---

### Code Review 常見問題

| 問題 | 回答 |
|------|------|
| RLE 為什麼要 column-major？ | Kaggle 規定的格式 |
| RLE 為什麼 1-indexed？ | Kaggle 規定，從 1 開始計數 |
| TTA 為什麼能提升效果？ | 平均多個角度的預測，降低模型的方向性偏差 |
| Resize 回原始大小為什麼用 NEAREST？ | 保持 mask 的 0/1 整數值不被插值破壞 |
| 為什麼輸出 CSV 而不是圖片？ | RLE 格式體積小，是 Kaggle 的標準提交格式 |
