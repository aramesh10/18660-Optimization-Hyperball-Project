"""Simple Muon vs. Muon-Hyperball comparison in PyTorch.

Blog equation used for hMuon:

    W_next = Normalize_R(W - lr * Normalize_R(u))
    Normalize_R(x) = R * x / ||x||_F

Here u is the standard Muon update direction.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor

from models.simple_mlp import init_mlp_weights, make_teacher_dataset, mlp_forward
from optimizers.hmuon import hmuon_step, normalize_R
from optimizers.muon import adjust_lr, muon_step


def train(
    method: str,
    x_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    y_val: Tensor,
    initial_weights: list[Tensor],
    lr: float,
    steps: int,
    batch_size: int,
    beta: float,
    adjust_lr_mode: str | None,
):
    weights = [w.detach().clone().requires_grad_(True) for w in initial_weights]
    # Hyperball uses the initial Frobenius norm. These radii stay fixed.
    radii = [w.detach().norm() for w in weights]
    buffers = [torch.zeros_like(w) for w in weights]

    history = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "weight_norm": [],
        "update_norm": [],
        "sec_per_step": [],
    }

    n_train = x_train.shape[0]
    start_time = time.perf_counter()
    for step in range(1, steps + 1):
        idx = torch.randint(n_train, (batch_size,))
        loss = F.cross_entropy(mlp_forward(weights, x_train[idx]), y_train[idx])
        loss.backward()

        with torch.no_grad():
            update_norm_sum = 0.0
            for i, weight in enumerate(weights):
                effective_lr = adjust_lr(lr, weight.shape, adjust_lr_mode)
                if method == "muon":
                    new_weight, buffers[i], update = muon_step(
                        weight,
                        weight.grad,
                        buffers[i],
                        lr=effective_lr,
                        momentum=beta,
                        adjust_lr_mode="none",
                        update_radius=radii[i],
                    )
                elif method == "hmuon":
                    new_weight, buffers[i], update = hmuon_step(
                        weight,
                        weight.grad,
                        buffers[i],
                        lr=effective_lr,
                        R=radii[i],
                        momentum=beta,
                    )
                else:
                    raise ValueError(f"unknown method: {method}")

                update_norm_sum += normalize_R(update, radii[i]).norm().item()
                weight.copy_(new_weight)
                weight.grad = None

        if step == 1 or step % 10 == 0 or step == steps:
            with torch.no_grad():
                val_logits = mlp_forward(weights, x_val)
                val_loss = F.cross_entropy(val_logits, y_val).item()
                val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["weight_norm"].append(sum(w.detach().norm().item() for w in weights))
            history["update_norm"].append(update_norm_sum)
            history["sec_per_step"].append((time.perf_counter() - start_time) / step)

    return history


def plot_comparison(muon, hmuon, lr: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plots = [
        ("train_loss", "Training loss"),
        ("val_acc", "Validation accuracy"),
        ("weight_norm", "Weight norm"),
    ]

    for ax, (key, title) in zip(axes, plots):
        ax.plot(muon["step"], muon[key], label="Muon", linewidth=2)
        ax.plot(hmuon["step"], hmuon[key], label="hMuon", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("step")
        if key == "update_norm":
            update_max = max(max(muon[key]), max(hmuon[key]))
            ax.set_ylim(0.0, update_max * 1.1)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(f"Muon vs hMuon, lr={lr:g}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_experiment(
    lr: float,
    steps: int,
    batch_size: int,
    beta: float,
    seed: int,
    output_dir: Path,
    adjust_lr_mode: str | None,
) -> None:
    d_in = 64
    hidden = 128
    d_out = 10
    teacher_gen = torch.Generator().manual_seed(seed + 10)
    teacher_weights = init_mlp_weights(d_in, hidden, d_out, teacher_gen)
    data_gen = torch.Generator().manual_seed(seed)
    x_train, y_train = make_teacher_dataset(8192, d_in, teacher_weights, data_gen)
    x_val, y_val = make_teacher_dataset(2048, d_in, teacher_weights, data_gen)
    init_gen = torch.Generator().manual_seed(seed + 1)
    initial_weights = init_mlp_weights(d_in, hidden, d_out, init_gen)

    train_kwargs = dict(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        initial_weights=initial_weights,
        lr=lr,
        steps=steps,
        batch_size=batch_size,
        beta=beta,
        adjust_lr_mode=adjust_lr_mode,
    )
    muon = train("muon", **train_kwargs)
    hmuon = train("hmuon", **train_kwargs)

    plot_comparison(muon, hmuon, lr, output_dir / f"muon_vs_hmuon_lr_{lr:g}.png")

    print(f"\nlr = {lr:g}")
    print(
        f"Muon : train_loss={muon['train_loss'][-1]:.6f}, "
        f"val_acc={muon['val_acc'][-1]:.4f}, final ||W||_F={muon['weight_norm'][-1]:.4f}"
    )
    print(
        f"hMuon: train_loss={hmuon['train_loss'][-1]:.6f}, "
        f"val_acc={hmuon['val_acc'][-1]:.4f}, final ||W||_F={hmuon['weight_norm'][-1]:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results_hmuon"))
    parser.add_argument(
        "--adjust-lr-mode",
        choices=["none", "original", "match_rms_adamw"],
        default="none",
        help="Use 'none' for the literal blog equation. Other modes reproduce Muon's LR convention.",
    )
    args = parser.parse_args()
    adjust_lr_mode = None if args.adjust_lr_mode == "none" else args.adjust_lr_mode

    for lr in args.lrs:
        run_experiment(lr, args.steps, args.batch_size, args.beta, args.seed, args.output_dir, adjust_lr_mode)

    print(f"\nSaved plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
