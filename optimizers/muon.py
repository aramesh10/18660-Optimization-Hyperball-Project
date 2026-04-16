"""Minimal Muon implementation used by the comparison script."""

from __future__ import annotations

import math

import torch
from torch import Tensor


EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def zeropower_via_newtonschulz(
    grad: Tensor,
    ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
) -> Tensor:
    """Newton-Schulz orthogonalization from the Muon implementation."""
    if grad.ndim != 2:
        raise ValueError("Muon expects a 2D matrix.")

    a, b, c = ns_coefficients
    X = grad.bfloat16()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T

    X.div_(X.norm().clamp_min(eps))

    for _ in range(ns_steps):
        A = X @ X.T
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a)

    if transposed:
        X = X.T
    return X


def adjust_lr(lr: float, shape: torch.Size, mode: str | None = "original") -> float:
    """Learning-rate adjustment used by Muon."""
    rows, cols = shape[:2]

    if mode is None or mode == "original":
        return lr * math.sqrt(max(1.0, rows / cols))
    if mode == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(rows, cols))
    if mode == "none":
        return lr

    raise ValueError(f"Unknown adjust_lr mode: {mode}")


def muon_direction(
    grad: Tensor,
    momentum_buffer: Tensor,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
) -> tuple[Tensor, Tensor]:
    """Return Muon's orthogonalized update direction and new momentum buffer."""
    new_buffer = momentum * momentum_buffer + (1.0 - momentum) * grad
    update = grad.lerp(new_buffer, momentum) if nesterov else new_buffer
    update = zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)
    return update, new_buffer


def muon_step(
    param: Tensor,
    grad: Tensor,
    momentum_buffer: Tensor,
    lr: float,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
    nesterov: bool = True,
    adjust_lr_mode: str | None = "original",
    update_radius: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply one Muon step and return parameter, momentum buffer, update direction."""
    update, new_buffer = muon_direction(
        grad=grad,
        momentum_buffer=momentum_buffer,
        momentum=momentum,
        nesterov=nesterov,
    )
    step_update = update
    if update_radius is not None:
        step_update = update_radius * update / update.norm().clamp_min(EPS)

    adjusted_lr = adjust_lr(lr, param.shape, adjust_lr_mode)
    new_param = param * (1.0 - lr * weight_decay) - adjusted_lr * step_update
    return new_param, new_buffer, update
