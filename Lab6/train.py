import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset import ICLEVRDataset, TestDataset
from ddpm import DDPM
from model import UNet

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
IMAGE_DIR     = BASE_DIR / 'iclevr'
TRAIN_JSON    = BASE_DIR / 'train.json'
TEST_JSON     = BASE_DIR / 'test.json'
NEW_TEST_JSON = BASE_DIR / 'new_test.json'
OBJ_JSON      = BASE_DIR / 'objects.json'
CKPT_DIR      = BASE_DIR / 'checkpoints'
CKPT_LATEST   = CKPT_DIR / 'latest.pth'
CKPT_BEST     = CKPT_DIR / 'best.pth'

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────

BATCH_SIZE     = 64
LR             = 2e-4
WEIGHT_DECAY   = 1e-4
NUM_EPOCHS     = 300
WARMUP_STEPS   = 500
GRAD_CLIP      = 1.0
EMA_DECAY      = 0.9999
VAL_EVERY      = 10    # epochs
SAVE_EVERY     = 10    # epochs
DDIM_STEPS     = 200
GUIDANCE_SCALE = 0.0
LOG_EVERY      = 50    # steps
SEED           = 42


# ─────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay  = decay
        self.shadow = {
            name: param.data.clone().float()
            for name, param in model.named_parameters()
        }
        self._backup: dict = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                param.data.float(), alpha=1.0 - self.decay
            )

    def apply_shadow(self, model: nn.Module) -> None:
        self._backup = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name].to(param.dtype))

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            param.data.copy_(self._backup[name])


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    unet: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_acc: float,
    acc_test,
    acc_new,
    is_best: bool,
) -> None:
    ckpt = {
        'model':        unet.state_dict(),
        'ema_shadow':   {k: v.cpu() for k, v in ema.shadow.items()},
        'optimizer':    optimizer.state_dict(),
        'scheduler':    scheduler.state_dict(),
        'scaler':       scaler.state_dict(),
        'epoch':        epoch,
        'global_step':  global_step,
        'best_acc':     best_acc,
        'acc_test':     acc_test,
        'acc_new_test': acc_new,
    }
    torch.save(ckpt, path)
    tag = '[BEST]' if is_best else '[latest]'
    print(f"  {tag} Checkpoint saved → {path}")


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def run_validation(
    ddpm: DDPM,
    unet: nn.Module,
    ema: EMA,
    evaluator,
    test_cond: torch.Tensor,
    new_test_cond: torch.Tensor,
) -> tuple:
    unet.eval()
    ema.apply_shadow(unet)
    try:
        gen_test = ddpm.ddim_sample(test_cond,     n_steps=DDIM_STEPS, guidance_scale=GUIDANCE_SCALE)
        gen_new  = ddpm.ddim_sample(new_test_cond, n_steps=DDIM_STEPS, guidance_scale=GUIDANCE_SCALE)
        acc_test = evaluator.eval(gen_test, test_cond)
        acc_new  = evaluator.eval(gen_new,  new_test_cond)
    finally:
        ema.restore(unet)
        unet.train()
    return acc_test, acc_new


# ─────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Device: {device}  AMP: {use_amp}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────
    train_ds = ICLEVRDataset(
        image_dir=str(IMAGE_DIR),
        json_path=str(TRAIN_JSON),
        obj_json_path=str(OBJ_JSON),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    test_ds     = TestDataset(str(TEST_JSON),     str(OBJ_JSON))
    new_test_ds = TestDataset(str(NEW_TEST_JSON), str(OBJ_JSON))
    test_cond     = torch.stack([test_ds[i]     for i in range(len(test_ds))]).to(device)
    new_test_cond = torch.stack([new_test_ds[i] for i in range(len(new_test_ds))]).to(device)

    # ── Model ───────────────────────────────
    unet = UNet(
        in_channels=3,
        base_channels=64,
        channel_mults=[1, 2, 4, 4],
        num_res_blocks=2,
        emb_dim=256,
        num_classes=24,
        dropout=0.1,
    ).to(device)

    ddpm = DDPM(unet, T=1000).to(device)
    ema  = EMA(unet, decay=EMA_DECAY)

    # ── Optimizer / Scheduler / Scaler ──────
    optimizer    = torch.optim.AdamW(
        unet.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999)
    )
    total_steps  = NUM_EPOCHS * len(train_loader)

    def lr_lambda(current_step: int) -> float:
        if current_step < WARMUP_STEPS:
            return current_step / max(1, WARMUP_STEPS)
        progress = (current_step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler('cuda', enabled=use_amp)

    # ── Evaluator (CUDA only) ────────────────
    evaluator = None
    if device.type == 'cuda':
        from evaluator import evaluation_model
        evaluator = evaluation_model()
        print("Evaluator loaded.")

    # ── Resume ──────────────────────────────
    start_epoch = 0
    global_step = 0
    best_acc    = 0.0
    acc_test    = None
    acc_new     = None

    if CKPT_LATEST.exists():
        print(f"Resuming from {CKPT_LATEST}")
        ckpt = torch.load(CKPT_LATEST, map_location=device, weights_only=True)
        unet.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        ema.shadow  = {k: v.to(device) for k, v in ckpt['ema_shadow'].items()}
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        best_acc    = ckpt['best_acc']
        acc_test    = ckpt.get('acc_test')
        acc_new     = ckpt.get('acc_new_test')
        print(f"  Resumed: epoch={start_epoch}, step={global_step}, best_acc={best_acc:.4f}")

    total_params = sum(p.numel() for p in unet.parameters())
    print(f"UNet params: {total_params:,}")
    print(f"Train batches/epoch: {len(train_loader)}")

    # ── Training ────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS):
        unet.train()
        epoch_loss  = 0.0
        num_batches = 0

        for images, conditions in train_loader:
            images     = images.to(device, non_blocking=True)
            conditions = conditions.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=use_amp):
                loss = ddpm.p_losses(images, conditions)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(unet.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(unet)

            epoch_loss  += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % LOG_EVERY == 0:
                avg_loss = epoch_loss / num_batches
                cur_lr   = scheduler.get_last_lr()[0]
                print(f"[Epoch {epoch:03d} | Step {global_step:6d}] "
                      f"loss={avg_loss:.4f}  lr={cur_lr:.2e}")

        avg_epoch_loss = epoch_loss / num_batches
        print(f"=== Epoch {epoch:03d} done | avg_loss={avg_epoch_loss:.4f} ===")

        # ── Validation ──────────────────────
        if evaluator is not None and ((epoch + 1) % VAL_EVERY == 0 or epoch == NUM_EPOCHS - 1):
            acc_test, acc_new = run_validation(
                ddpm, unet, ema, evaluator, test_cond, new_test_cond
            )
            print(f"  [Val] test_acc={acc_test:.4f}  new_test_acc={acc_new:.4f}")

            avg_acc = (acc_test + acc_new) / 2.0
            if avg_acc > best_acc:
                best_acc = avg_acc
                save_checkpoint(
                    CKPT_BEST, unet, ema, optimizer, scheduler, scaler,
                    epoch, global_step, best_acc, acc_test, acc_new, is_best=True,
                )
                print(f"  [Best] New best! avg_acc={best_acc:.4f}")

        # ── Save latest ─────────────────────
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            save_checkpoint(
                CKPT_LATEST, unet, ema, optimizer, scheduler, scaler,
                epoch, global_step, best_acc, acc_test, acc_new, is_best=False,
            )

    print(f"\nTraining complete. Best avg_acc: {best_acc:.4f}")


if __name__ == '__main__':
    main()
