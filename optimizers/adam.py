"""Minimal stateless Adam step, matching the style of muon.py."""

from __future__ import annotations

import math
import torch
from torch import Tensor


def adam_step(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply one Adam step with decoupled weight decay (AdamW).

    Returns (new_param, new_exp_avg, new_exp_avg_sq).
    """
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    step_size = lr / bias_correction1
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

    new_param = param * (1.0 - lr * weight_decay) - step_size * exp_avg / denom
    return new_param, exp_avg, exp_avg_sq
