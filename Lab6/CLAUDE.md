# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NYCU Deep Learning Lab6 — Conditional DDPM for i-CLEVR image generation.
Goal: train a model that generates 64×64 RGB images conditioned on multi-label object descriptors (e.g., "red sphere", "cyan cylinder"), achieving classification accuracy ≥ 0.8 on the provided evaluator.

Deadline: **2026-05-12 23:59**. No demo required.

## Planned File Architecture

```
Lab6/
├── dataset.py      # ICLEVRDataset (train), TestDataset (test/new_test), helpers
├── model.py        # Conditional U-Net (TimeEmbedding, LabelEmbedding, ResBlock, UNet)
├── ddpm.py         # DDPM class: noise schedule, q_sample, p_losses, p_sample, sample
├── train.py        # Training loop entry point
├── inference.py    # Sampling + saving images/grids/denoising visualization
├── evaluator.py    # TA-provided ResNet18 evaluator — DO NOT MODIFY
├── checkpoint.pth  # Evaluator weights — DO NOT MODIFY
├── objects.json    # 24-class label dict {"gray cube": 0, ..., "yellow cylinder": 23}
├── train.json      # Dict: {filename: [labels]}  — 18009 training samples
├── test.json       # List: [[labels], ...]        — 32 test samples
├── new_test.json   # List: [[labels], ...]        — 32 test samples
├── iclevr/         # Training images (DO NOT commit)
└── images/
    ├── test/       # 0.png ~ 31.png (must match test.json order)
    └── new_test/   # 0.png ~ 31.png (must match new_test.json order)
```

## Running the Code

```bash
# Verify dataset pipeline
python dataset.py

# Train
python train.py

# Generate test images and evaluate
python inference.py
```

## Critical Constraints

**Evaluator interface** (`evaluator.py`):
- Input images: `(batch, 3, 64, 64)`, normalized with `Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))`
- Input labels: `(batch, 24)` float one-hot vectors
- Call: `evaluator.eval(images, labels)` → returns scalar accuracy
- `evaluator.py` and `checkpoint.pth` must not be modified (inheriting the class is allowed)

**Image normalization convention**:
- Training/evaluation: `[-1, 1]` range via `Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))`
- Saving to disk: denormalize first → `img = (img * 0.5 + 0.5).clamp(0, 1)`

**Output format** (graded):
- `images/test/` and `images/new_test/`: individual `.png` files, index matches JSON order
- Grid images: `make_grid(nrow=8)`, 4 rows × 8 cols per test JSON
- Denoising visualization: label `["red sphere", "cyan cylinder", "cyan cube"]`, ≥8 frames in one row

## Key Design Decisions (from planning)

**Condition injection**: 24-dim one-hot → `nn.Linear(24, emb_dim)`, combined with sinusoidal time embedding via addition or concat, injected into each U-Net ResBlock.

**Noise schedule**: Cosine schedule recommended over linear for 64×64 images (more stable gradients at low noise levels).

**Label encoding**: `labels_to_onehot(labels: List[str], obj_dict: Dict[str, int]) -> FloatTensor(24,)` — same object never appears twice per image per spec.

**Sampling**: DDPM or DDIM both acceptable; classifier guidance using the pretrained evaluator is allowed and can boost accuracy.

## Submission

Zip as `DL_LAB6_M11407439_林明宏.zip` containing:
- Report (`.pdf`)
- Source code (`.py` files)
- `images/` folder

Do **not** include `iclevr/` dataset in the zip.
