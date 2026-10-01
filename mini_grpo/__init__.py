"""MiniGRPO: 从零实现 GRPO / DAPO / Dr.GRPO 的损失函数。

目标：与 TRL 1.14 GRPOTrainer 做数值对齐验证。
"""
from .losses import (
    compute_advantages,
    grpo_loss,
    dapo_loss,
    dr_grpo_loss,
)

__all__ = [
    "compute_advantages",
    "grpo_loss",
    "dapo_loss",
    "dr_grpo_loss",
]
