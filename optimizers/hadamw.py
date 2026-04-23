"""Adam-Hyperball (AdamH): Adam update projected onto the Frobenius hyperball.

Analogous to hMuon but using Adam's adaptive update direction instead of
the Newton-Schulz orthogonalized Muon direction.

W_next = Normalize_R(W - lr * adam_direction)
"""

from __future__ import annotations

import math
import torch
from torch import Tensor

from optimizers.hmuon import normalize_R


def hadamw_step(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step: int,
    lr: float,
    R: Tensor,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply one AdamH step.

    Returns (new_param, new_exp_avg, new_exp_avg_sq).
    Weight decay is implicit — the hyperball projection fixes the norm to R,
    so explicit L2 decay is redundant.
    """
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    step_size = lr / bias_correction1
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

    adam_dir = exp_avg / denom
    new_param = normalize_R(param - step_size * adam_dir, R)
    return new_param, exp_avg, exp_avg_sq
