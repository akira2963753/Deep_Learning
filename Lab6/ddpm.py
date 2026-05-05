import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import UNet


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    # a: (T,)  t: (B,) 0-based index  →  (B, 1, 1, 1)
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))


def _build_cosine_schedule(T: int, s: float = 0.008) -> dict:
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1.0 + s) * math.pi / 2.0) ** 2

    alphas_cumprod = (f[1:] / f[0]).float()                          # (T,)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # (T,)

    betas  = (1.0 - f[1:] / f[:-1]).clamp(0, 0.999).float()         # (T,)
    alphas = 1.0 - betas                                              # (T,)

    sqrt_alphas_cumprod           = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    ).clamp(min=1e-20)
    posterior_log_variance_clipped = posterior_variance.log()

    posterior_mean_coef1 = alphas_cumprod_prev.sqrt() * betas / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = alphas.sqrt() * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        'betas':                          betas,
        'alphas':                         alphas,
        'alphas_cumprod':                 alphas_cumprod,
        'alphas_cumprod_prev':            alphas_cumprod_prev,
        'sqrt_alphas_cumprod':            sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod':  sqrt_one_minus_alphas_cumprod,
        'posterior_variance':             posterior_variance,
        'posterior_log_variance_clipped': posterior_log_variance_clipped,
        'posterior_mean_coef1':           posterior_mean_coef1,
        'posterior_mean_coef2':           posterior_mean_coef2,
    }


# ─────────────────────────────────────────────
# DDPM
# ─────────────────────────────────────────────

class DDPM(nn.Module):
    def __init__(self, model: UNet, T: int = 1000) -> None:
        super().__init__()
        self.model = model
        self.T     = T

        schedule = _build_cosine_schedule(T)
        for name in (
            'betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
            'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
            'posterior_variance', 'posterior_log_variance_clipped',
        ):
            self.register_buffer(name, schedule[name])

    # ── Forward Process ──────────────────────

    def q_sample(
        self,
        x0:    torch.Tensor,            # (B, 3, 64, 64)
        t:     torch.Tensor,            # (B,) 0-based
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_abar   = extract(self.sqrt_alphas_cumprod,           t, x0.shape)
        sqrt_1mabar = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_abar * x0 + sqrt_1mabar * noise

    # ── Training Loss ────────────────────────

    def p_losses(
        self,
        x0:        torch.Tensor,        # (B, 3, 64, 64)
        condition: torch.Tensor,        # (B, 24)
        noise:     Optional[torch.Tensor] = None,
        loss_type: str = 'huber',
    ) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(x0)

        x_t      = self.q_sample(x0, t, noise)
        eps_pred = self.model(x_t, t, condition)

        if loss_type == 'huber':
            return F.smooth_l1_loss(eps_pred, noise)
        elif loss_type == 'l2':
            return F.mse_loss(eps_pred, noise)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    # ── DDPM Single-step Reverse ─────────────

    @torch.no_grad()
    def p_sample(
        self,
        x:         torch.Tensor,        # (B, 3, 64, 64)
        t:         int,                 # scalar 0-based
        condition: torch.Tensor,        # (B, 24)
    ) -> torch.Tensor:
        B       = x.shape[0]
        t_batch = torch.full((B,), t, device=x.device, dtype=torch.long)

        betas_t    = extract(self.betas,                          t_batch, x.shape)
        sqrt_1mab  = extract(self.sqrt_one_minus_alphas_cumprod,  t_batch, x.shape)
        sqrt_rec_a = extract(self.alphas,                         t_batch, x.shape).sqrt().reciprocal()

        eps_pred = self.model(x, t_batch, condition)
        mean     = sqrt_rec_a * (x - betas_t / sqrt_1mab * eps_pred)

        if t == 0:
            return mean

        log_var = extract(self.posterior_log_variance_clipped, t_batch, x.shape)
        return mean + (0.5 * log_var).exp() * torch.randn_like(x)

    # ── DDPM Full Sampling Loop ──────────────

    @torch.no_grad()
    def p_sample_loop(
        self,
        condition:            torch.Tensor,   # (B, 24)
        return_intermediates: bool = False,
        n_intermediates:      int  = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        B      = condition.shape[0]
        device = condition.device
        x      = torch.randn(B, 3, 64, 64, device=device)

        intermediates: List[torch.Tensor] = []
        if return_intermediates:
            save_steps = set(
                torch.linspace(self.T - 1, 0, n_intermediates).long().tolist()
            )

        for t in reversed(range(self.T)):
            x = self.p_sample(x, t, condition)
            if return_intermediates and t in save_steps:
                intermediates.append(x.clone())

        x = x.clamp(-1.0, 1.0)
        return (x, intermediates) if return_intermediates else x

    # ── DDIM Sampling (with optional Classifier Guidance) ──

    @torch.no_grad()
    def ddim_sample(
        self,
        condition:            torch.Tensor,   # (B, 24)
        evaluator=None,                       # evaluation_model instance（可選）
        guidance_scale:       float = 1.0,
        n_steps:              int   = 200,
        eta:                  float = 0.0,
        return_intermediates: bool  = False,
        n_intermediates:      int   = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        B      = condition.shape[0]
        device = condition.device
        x      = torch.randn(B, 3, 64, 64, device=device)

        ddim_ts      = torch.linspace(self.T - 1, 0, n_steps).long().tolist()
        ddim_ts_prev = ddim_ts[1:] + [-1]

        intermediates: List[torch.Tensor] = []
        save_interval = max(n_steps // n_intermediates, 1)

        for i, (t, t_prev) in enumerate(zip(ddim_ts, ddim_ts_prev)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            abar_t    = extract(self.alphas_cumprod, t_batch, x.shape)
            sqrt_1mab = extract(self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
            sqrt_abar = extract(self.sqrt_alphas_cumprod, t_batch, x.shape)

            if t_prev >= 0:
                t_prev_batch = torch.full((B,), t_prev, device=device, dtype=torch.long)
                abar_prev    = extract(self.alphas_cumprod, t_prev_batch, x.shape)
            else:
                abar_prev = torch.ones_like(abar_t)

            # Step 1: U-Net forward（no_grad，只做一次）
            with torch.no_grad():
                eps_pred = self.model(x, t_batch, condition)

            # Step 2: Classifier Guidance（梯度只穿過 x0_hat 與 ResNet18）
            if evaluator is not None and guidance_scale > 0.0:
                with torch.enable_grad():
                    x_in   = x.detach().requires_grad_(True)
                    # eps_pred 已 detach，切斷 U-Net 計算圖
                    x0_hat = (x_in - sqrt_1mab * eps_pred) / sqrt_abar
                    x0_hat = x0_hat.clamp(-1.0, 1.0)
                    probs  = evaluator.resnet18(x0_hat)
                    log_p  = -F.binary_cross_entropy(probs, condition, reduction='sum')
                    grad   = torch.autograd.grad(log_p, x_in)[0]
                eps_pred = eps_pred - sqrt_1mab * guidance_scale * grad

            # Step 3: DDIM update
            x0_pred = (x - sqrt_1mab * eps_pred) / sqrt_abar
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            sigma_t = eta * (
                (1.0 - abar_prev) / (1.0 - abar_t) * (1.0 - abar_t / abar_prev)
            ).clamp(min=0.0).sqrt()

            dir_xt = (1.0 - abar_prev - sigma_t ** 2).clamp(min=0.0).sqrt() * eps_pred
            noise  = sigma_t * torch.randn_like(x) if eta > 0.0 else 0.0
            x      = abar_prev.sqrt() * x0_pred + dir_xt + noise

            if return_intermediates and (i % save_interval == 0):
                intermediates.append(x.detach().clone())

        x = x.clamp(-1.0, 1.0)
        if return_intermediates:
            return x, intermediates[:n_intermediates]
        return x


# ─────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = UNet().to(device)
    ddpm  = DDPM(model, T=1000).to(device)

    cond = torch.zeros(2, 24, device=device)
    cond[0, 0] = 1.0
    cond[1, 1] = 1.0
    x0 = torch.randn(2, 3, 64, 64, device=device)
    t  = torch.randint(0, 1000, (2,), device=device)

    # Forward process
    xt = ddpm.q_sample(x0, t)
    assert xt.shape == (2, 3, 64, 64), f"q_sample shape: {xt.shape}"
    print(f"q_sample OK  : {xt.shape}")

    # Training loss
    loss = ddpm.p_losses(x0, cond)
    assert loss.ndim == 0
    print(f"p_losses OK  : {loss.item():.4f}")

    # DDIM sample (fast smoke test, n_steps=10)
    gen = ddpm.ddim_sample(cond, n_steps=10)
    assert gen.shape == (2, 3, 64, 64)
    assert gen.min() >= -1.01 and gen.max() <= 1.01
    print(f"ddim_sample OK : {gen.shape}, range [{gen.min():.3f}, {gen.max():.3f}]")

    # Intermediates
    gen, frames = ddpm.ddim_sample(
        cond[:1], n_steps=20, return_intermediates=True, n_intermediates=8
    )
    assert len(frames) == 8
    print(f"intermediates OK : {len(frames)} frames")

    print("\nAll checks passed.")
