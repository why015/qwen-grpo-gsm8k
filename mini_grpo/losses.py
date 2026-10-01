"""MiniGRPO losses: GRPO, DAPO, Dr.GRPO policy gradient losses.

Aligned with TRL 1.14 GRPOTrainer. Pure math, no I/O.
"""
from typing import Dict, Tuple

import torch


# ============================================================
# 1. Group-relative advantage
# ============================================================

def compute_advantages(
    rewards: torch.Tensor,
    group_size: int,
    scale_by_std: bool = True,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute group-relative advantage from per-completion rewards.

    Args:
        rewards: (B,) scalar rewards. B must be divisible by group_size.
        group_size: number of completions per group.
        scale_by_std: True -> (r - mean) / std; False -> (r - mean).
        eps: small value to avoid division by zero.

    Returns:
        (B,) advantages.
    """
    B = rewards.shape[0]
    assert B % group_size == 0, f"B={B} not divisible by group_size={group_size}"

    rg = rewards.view(-1, group_size)                # (B/G, G)
    mean = rg.mean(dim=1, keepdim=True)              # (B/G, 1)
    centered = rg - mean                             # (B/G, G)

    if scale_by_std:
        std = rg.std(dim=1, unbiased=False, keepdim=True)
        adv = centered / (std + eps)
    else:
        adv = centered

    return adv.view(-1)                              # (B,)


# ============================================================
# 2. Shared helpers
# ============================================================

def _compute_ratio(logp: torch.Tensor, old_logp: torch.Tensor) -> torch.Tensor:
    """rho = exp(logp - old_logp). Numerically stabilized by subtracting max."""
    diff = logp - old_logp
    diff = diff - diff.max().detach()
    return torch.exp(diff)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


# ============================================================
# 3. GRPO loss (sequence-level, symmetric clipping)
# ============================================================

def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.04,
    epsilon: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """GRPO original loss: per-sequence mean then group mean."""
    advantages = advantages.unsqueeze(-1)            # (B, 1)

    ratio = _compute_ratio(logp, old_logp)           # (B, T)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    surr = torch.minimum(unclipped, clipped)         # (B, T)

    surr_per_seq = (surr * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)   # (B,)
    policy_loss = -surr_per_seq.mean()

    kl_per_token = logp - ref_logp                   # (B, T)
    kl = _masked_mean(kl_per_token, mask)

    total = policy_loss + beta * kl

    metrics = {
        "policy_loss": policy_loss.item(),
        "kl": kl.item(),
        "ratio_mean": _masked_mean(ratio, mask).item(),
        "clip_frac": _masked_mean((unclipped != surr).float(), mask).item(),
    }
    return total, metrics


# ============================================================
# 4. DAPO loss (token-level, asymmetric clipping)
# ============================================================

def dapo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.04,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """DAPO loss: asymmetric clipping + token-level aggregation."""
    advantages = advantages.unsqueeze(-1)

    ratio = _compute_ratio(logp, old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high) * advantages
    surr = torch.minimum(unclipped, clipped)

    policy_loss = -(surr * mask).sum() / (mask.sum() + 1e-8)

    kl_per_token = logp - ref_logp
    kl = _masked_mean(kl_per_token, mask)

    total = policy_loss + beta * kl

    metrics = {
        "policy_loss": policy_loss.item(),
        "kl": kl.item(),
        "ratio_mean": _masked_mean(ratio, mask).item(),
        "clip_frac": _masked_mean((unclipped != surr).float(), mask).item(),
    }
    return total, metrics


# ============================================================
# 5. Dr.GRPO loss (same as GRPO but advantage not scaled by std)
# ============================================================

def dr_grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.04,
    epsilon: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Dr.GRPO loss: same as GRPO, but caller must pass unscaled advantages."""
    return grpo_loss(logp, old_logp, ref_logp, advantages, mask, beta, epsilon)
