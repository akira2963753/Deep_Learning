import argparse
import json
from pathlib import Path

import torch
import torchvision.utils as vutils
from PIL import Image

from dataset import TestDataset, labels_to_onehot
from ddpm import DDPM
from evaluator import evaluation_model
from model import UNet
from train import EMA

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
TEST_JSON     = BASE_DIR / 'test.json'
NEW_TEST_JSON = BASE_DIR / 'new_test.json'
OBJ_JSON      = BASE_DIR / 'objects.json'
CKPT_BEST     = BASE_DIR / 'checkpoints' / 'best.pth'
OUT_TEST      = BASE_DIR / 'images' / 'test'
OUT_NEW_TEST  = BASE_DIR / 'images' / 'new_test'

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DDIM_STEPS     = 200
GUIDANCE_SCALE = 0.0   # set > 0 (e.g. 2.0) to enable classifier guidance
ETA            = 0.0

# Labels for denoising visualization
VIZ_LABELS = ['red sphere', 'cyan cylinder', 'cyan cube']
VIZ_FRAMES = 16   # number of denoising intermediate frames (≥ 8 required)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    # t: (3, H, W) in [-1, 1]
    img = (t * 0.5 + 0.5).clamp(0, 1)
    return Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))


def save_images(images: torch.Tensor, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        tensor_to_pil(img).save(out_dir / f'{i}.png')


def save_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = vutils.make_grid(
        (images * 0.5 + 0.5).clamp(0, 1),
        nrow=nrow, padding=2, pad_value=1.0
    )
    Image.fromarray(
        (grid.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    ).save(path)


def load_model(device: torch.device, use_ema: bool = True) -> tuple:
    unet = UNet(
        in_channels=3, base_channels=64, channel_mults=[1, 2, 4, 4],
        num_res_blocks=2, emb_dim=256, num_classes=24, dropout=0.1,
    ).to(device)
    ddpm = DDPM(unet, T=1000).to(device)
    ema  = EMA(unet)

    ckpt = torch.load(CKPT_BEST, map_location=device, weights_only=True)
    unet.load_state_dict(ckpt['model'])
    if use_ema:
        ema.shadow = {k: v.to(device) for k, v in ckpt['ema_shadow'].items()}
        ema.apply_shadow(unet)
        print("Weights: EMA")
    else:
        print("Weights: raw (no EMA)")
    unet.eval()

    epoch = ckpt.get('epoch', '?')
    best  = ckpt.get('best_acc', float('nan'))
    print(f"Loaded checkpoint: epoch={epoch}  best_avg_acc={best:.4f}")
    return ddpm, unet, ema


def load_conditions(json_path: Path, obj_json_path: Path, device: torch.device) -> torch.Tensor:
    ds = TestDataset(str(json_path), str(obj_json_path))
    return torch.stack([ds[i] for i in range(len(ds))]).to(device)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run inference for conditional DDPM')
    parser.add_argument('--no_ema', action='store_true',
                        help='Use raw checkpoint weights instead of EMA weights')
    parser.add_argument('--ddim_steps', type=int, default=DDIM_STEPS,
                        help=f'Number of DDIM sampling steps (default: {DDIM_STEPS})')
    parser.add_argument('--guidance_scale', type=float, default=GUIDANCE_SCALE,
                        help='Classifier guidance scale (default: 0.0 = disabled)')
    return parser.parse_args()


def main() -> None:
    args      = parse_args()
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = evaluation_model()
    print(f"Device: {device}")

    ddpm, unet, ema = load_model(device, use_ema=not args.no_ema)

    test_cond     = load_conditions(TEST_JSON,     OBJ_JSON, device)
    new_test_cond = load_conditions(NEW_TEST_JSON, OBJ_JSON, device)

    # ── Generate test images ─────────────────
    print("\nGenerating test.json images...")
    gen_test = ddpm.ddim_sample(
        test_cond,
        evaluator=evaluator if args.guidance_scale > 0 else None,
        guidance_scale=args.guidance_scale,
        n_steps=args.ddim_steps,
        eta=ETA,
    )
    save_images(gen_test, OUT_TEST)
    save_grid(gen_test, BASE_DIR / 'images' / 'test_grid.png', nrow=8)
    acc_test = evaluator.eval(gen_test, test_cond)
    print(f"  test_acc = {acc_test:.4f}   → saved to {OUT_TEST}/")

    # ── Generate new_test images ─────────────
    print("\nGenerating new_test.json images...")
    gen_new = ddpm.ddim_sample(
        new_test_cond,
        evaluator=evaluator if args.guidance_scale > 0 else None,
        guidance_scale=args.guidance_scale,
        n_steps=args.ddim_steps,
        eta=ETA,
    )
    save_images(gen_new, OUT_NEW_TEST)
    save_grid(gen_new, BASE_DIR / 'images' / 'new_test_grid.png', nrow=8)
    acc_new = evaluator.eval(gen_new, new_test_cond)
    print(f"  new_test_acc = {acc_new:.4f}   → saved to {OUT_NEW_TEST}/")

    # ── Denoising visualization ──────────────
    print(f"\nGenerating denoising visualization: {VIZ_LABELS}")
    with open(OBJ_JSON) as f:
        obj_dict = json.load(f)
    viz_cond = labels_to_onehot(VIZ_LABELS, obj_dict).unsqueeze(0).to(device)  # (1, 24)

    _, frames = ddpm.ddim_sample(
        viz_cond,
        n_steps=args.ddim_steps,
        eta=ETA,
        return_intermediates=True,
        n_intermediates=VIZ_FRAMES,
    )
    # frames: list of (1, 3, 64, 64), concat into single-row grid
    strip = torch.cat(frames, dim=0)           # (VIZ_FRAMES, 3, 64, 64)
    save_grid(strip, BASE_DIR / 'images' / 'denoising_viz.png', nrow=VIZ_FRAMES)
    print(f"  Denoising strip saved ({VIZ_FRAMES} frames) → images/denoising_viz.png")

    print(f"\n{'='*40}")
    print(f"  test accuracy    : {acc_test:.4f}")
    print(f"  new_test accuracy: {acc_new:.4f}")
    print(f"  avg accuracy     : {(acc_test + acc_new) / 2:.4f}")
    print(f"{'='*40}")


if __name__ == '__main__':
    main()
