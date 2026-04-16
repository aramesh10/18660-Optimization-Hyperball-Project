"""Muon-Hyperball implementation based on the blog equation."""

from __future__ import annotations

import torch
from torch import Tensor

from optimizers.muon import muon_direction


def normalize_R(x: Tensor, R: Tensor, eps: float = 1e-12) -> Tensor:
    """Project x onto the Frobenius sphere with radius R."""
    norm = x.norm()
    if norm <= eps:
        return torch.zeros_like(x)
    return R * x / norm

def hmuon_step(
    param: Tensor,
    grad: Tensor,
    momentum_buffer: Tensor,
    lr: float,
    R: Tensor,
    momentum: float = 0.95,
    nesterov: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply one Muon-Hyperball step.

    W_next = Normalize_R(W - lr * Normalize_R(u_t))

    R should be the fixed Frobenius norm of this matrix at initialization.
    """
    update, new_buffer = muon_direction(
        grad=grad,
        momentum_buffer=momentum_buffer,
        momentum=momentum,
        nesterov=nesterov,
    )
    new_param = normalize_R(param - lr * normalize_R(update, R), R)
    return new_param, new_buffer, update
