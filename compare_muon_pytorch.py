"""Compare this repo's Muon step with torch.optim.Muon.

The script applies the same sequence of fixed gradients to both optimizers and
prints the maximum absolute parameter difference after each step.
"""

from __future__ import annotations

import argparse
import sys

import torch

from optimizers.muon import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_C,
    DEFAULT_NS_STEPS,
    EPS,
    muon_step,
)


def get_torch_muon():
    try:
        return torch.optim.Muon
    except AttributeError as exc:
        raise RuntimeError(
            "This PyTorch install does not provide torch.optim.Muon. "
            "Upgrade to a PyTorch version that includes Muon, then rerun this script."
        ) from exc


def run_comparison(
    shape: tuple[int, int],
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    steps: int,
    seed: int,
    adjust_lr_mode: str | None,
) -> None:
    TorchMuon = get_torch_muon()

    generator = torch.Generator().manual_seed(seed)
    initial_param = torch.randn(shape, generator=generator)
    fixed_grads = [torch.randn(shape, generator=generator) for _ in range(steps)]

    ours = initial_param.clone()
    our_momentum_buffer = torch.zeros_like(ours)

    torch_param = torch.nn.Parameter(initial_param.clone())
    torch_optimizer = TorchMuon(
        [torch_param],
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_coefficients=(DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps=EPS,
        ns_steps=DEFAULT_NS_STEPS,
        adjust_lr_fn=adjust_lr_mode,
    )

    print("step  max_abs_diff  mean_abs_diff")
    print("-----------------------------------")

    for step, grad in enumerate(fixed_grads, start=1):
        with torch.no_grad():
            ours, our_momentum_buffer, _ = muon_step(
                param=ours,
                grad=grad,
                momentum_buffer=our_momentum_buffer,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
                adjust_lr_mode=adjust_lr_mode,
            )

        torch_param.grad = grad.clone()
        torch_optimizer.step()
        torch_optimizer.zero_grad(set_to_none=True)

        diff = (ours - torch_param.detach()).abs()
        print(f"{step:4d}  {diff.max().item():12.6g}  {diff.mean().item():13.6g}")

    print("\nFinal check")
    print(f"ours  ||W||_F = {ours.norm().item():.6f}")
    print(f"torch ||W||_F = {torch_param.detach().norm().item():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--no-nesterov", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--adjust-lr-mode",
        choices=["none", "original", "match_rms_adamw"],
        default="original",
    )
    args = parser.parse_args()

    adjust_lr_mode = None if args.adjust_lr_mode == "none" else args.adjust_lr_mode
    run_comparison(
        shape=(args.rows, args.cols),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=not args.no_nesterov,
        steps=args.steps,
        seed=args.seed,
        adjust_lr_mode=adjust_lr_mode,
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)
