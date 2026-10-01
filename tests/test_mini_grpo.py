"""Unit tests for mini_grpo/losses.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from mini_grpo.losses import (
    compute_advantages,
    grpo_loss,
    dapo_loss,
    dr_grpo_loss,
)


# ============================================================
# compute_advantages
# ============================================================

def test_advantages_basic():
    """4 completions, one group of 4: rewards [0, 1, 0, 1]."""
    r = torch.tensor([0.0, 1.0, 0.0, 1.0])
    adv = compute_advantages(r, group_size=4, scale_by_std=True)
    # mean = 0.5, std = 0.5
    expected = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    assert torch.allclose(adv, expected, atol=1e-3), f"got {adv}"


def test_advantages_no_std():
    """Dr.GRPO style: no scaling by std."""
    r = torch.tensor([0.0, 1.0, 0.0, 1.0])
    adv = compute_advantages(r, group_size=4, scale_by_std=False)
    expected = torch.tensor([-0.5, 0.5, -0.5, 0.5])
    assert torch.allclose(adv, expected, atol=1e-6), f"got {adv}"


def test_advantages_multiple_groups():
    """2 groups of 4, different reward distributions."""
    r = torch.tensor([0.0, 0.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0])
    adv = compute_advantages(r, group_size=4, scale_by_std=True)
    # group 1: mean=0.5, std=0.5 -> [-1, -1, 1, 1]
    # group 2: mean=1.0, std=0    -> [0, 0, 0, 0] (eps prevents div by zero)
    assert abs(adv[4].item()) < 1e-2
    assert abs(adv[7].item()) < 1e-2


def test_advantages_zero_variance():
    """All rewards equal -> advantages should be near 0 (eps handling)."""
    r = torch.tensor([0.5, 0.5, 0.5, 0.5])
    adv = compute_advantages(r, group_size=4, scale_by_std=True)
    assert adv.abs().max().item() < 1e-2


# ============================================================
# grpo_loss / dapo_loss / dr_grpo_loss
# ============================================================

def _make_fake_batch(B=2, T=4):
    """Build a tiny batch where logp == old_logp (ratio = 1)."""
    torch.manual_seed(0)
    logp = torch.randn(B, T) * 0.1
    old_logp = logp.clone()                     # ratio = 1
    ref_logp = logp - 0.01                      # small KL
    mask = torch.ones(B, T)
    advantages = torch.tensor([1.0, -1.0])      # B = 2
    return logp, old_logp, ref_logp, advantages, mask


def test_grpo_loss_ratio_one():
    """When ratio = 1, surr = advantages (no clipping)."""
    logp, old_logp, ref_logp, adv, mask = _make_fake_batch()
    loss, metrics = grpo_loss(logp, old_logp, ref_logp, adv, mask, beta=0.0)
    # policy_loss = -mean(advantages) = -(1 + (-1))/2 = 0
    assert abs(metrics["policy_loss"]) < 1e-5, f"got {metrics['policy_loss']}"
    # ratio should be all 1
    assert abs(metrics["ratio_mean"] - 1.0) < 1e-5


def test_grpo_loss_with_beta():
    """With beta > 0, loss should include KL term."""
    logp, old_logp, ref_logp, adv, mask = _make_fake_batch()
    loss0, _ = grpo_loss(logp, old_logp, ref_logp, adv, mask, beta=0.0)
    loss1, metrics1 = grpo_loss(logp, old_logp, ref_logp, adv, mask, beta=1.0)
    # KL = logp - ref_logp = 0.01 everywhere -> mean = 0.01
    assert abs(metrics1["kl"] - 0.01) < 1e-4, f"kl={metrics1['kl']}"
    diff = (loss1 - loss0).item()
    assert abs(diff - 0.01) < 1e-4, f"diff={diff}"


def test_dapo_loss_matches_grpo_when_symmetric():
    """DAPO with eps_low == eps_high == 0.2 should equal GRPO (ratio=1)."""
    logp, old_logp, ref_logp, adv, mask = _make_fake_batch()
    g, _ = grpo_loss(logp, old_logp, ref_logp, adv, mask, beta=0.0, epsilon=0.2)
    d, _ = dapo_loss(logp, old_logp, ref_logp, adv, mask, beta=0.0,
                     epsilon_low=0.2, epsilon_high=0.2)
    # Different aggregation (seq-mean vs token-mean) but same mask -> same value
    assert abs(g.item() - d.item()) < 1e-5


def test_dr_grpo_matches_grpo():
    """Dr.GRPO uses same loss as GRPO; only advantage differs (external)."""
    logp, old_logp, ref_logp, adv, mask = _make_fake_batch()
    g, _ = grpo_loss(logp, old_logp, ref_logp, adv, mask, beta=0.0)
    d, _ = dr_grpo_loss(logp, old_logp, ref_logp, adv, mask, beta=0.0)
    assert abs(g.item() - d.item()) < 1e-6


# ============================================================
# Numerical stability
# ============================================================

def test_no_nan_extreme_logp():
    """Extreme logp differences should not produce NaN."""
    logp = torch.tensor([[100.0, -100.0]])
    old_logp = torch.tensor([[-100.0, 100.0]])   # huge ratio
    ref_logp = logp.clone()
    adv = torch.tensor([1.0])
    mask = torch.ones(1, 2)
    loss, metrics = grpo_loss(logp, old_logp, ref_logp, adv, mask, beta=0.0)
    assert not torch.isnan(loss), "loss is NaN"
    assert not torch.isinf(loss), "loss is Inf"


def test_mask_zero_handled():
    """All-zero mask should not divide by zero."""
    logp = torch.randn(1, 4)
    old_logp = logp.clone()
    ref_logp = logp.clone()
    adv = torch.tensor([1.0])
    mask = torch.zeros(1, 4)
    loss, _ = grpo_loss(logp, old_logp, ref_logp, adv, mask, beta=0.0)
    assert not torch.isnan(loss)


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    import traceback
    passed = failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
                passed += 1
            except AssertionError:
                print(f"[FAIL] {name}")
                traceback.print_exc()
                failed += 1
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
