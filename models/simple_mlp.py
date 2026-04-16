"""Simple functional MLP used by the optimizer comparisons."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def init_mlp_weights(
    d_in: int,
    hidden: int,
    d_out: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> list[Tensor]:
    """Initialize matrix-only MLP weights."""
    return [
        torch.randn(hidden, d_in, generator=generator, device=device) / d_in**0.5,
        torch.randn(hidden, hidden, generator=generator, device=device) / hidden**0.5,
        torch.randn(d_out, hidden, generator=generator, device=device) / hidden**0.5,
    ]


def mlp_forward(weights: list[Tensor], x: Tensor) -> Tensor:
    """Forward pass for a ReLU MLP represented by a list of weight matrices."""
    x = F.relu(x @ weights[0].T)
    x = F.relu(x @ weights[1].T)
    return x @ weights[2].T


def make_teacher_dataset(
    n_samples: int,
    d_in: int,
    teacher_weights: list[Tensor],
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Create synthetic classification data from a fixed teacher MLP."""
    x = torch.randn(n_samples, d_in, generator=generator, device=device)
    with torch.no_grad():
        y = mlp_forward(teacher_weights, x).argmax(dim=1)
    return x, y
