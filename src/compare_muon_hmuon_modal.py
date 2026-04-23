"""Run a larger Muon vs hMuon experiment on Modal H100.

This uses a simple 3-layer MLP on a larger synthetic classification dataset.
The remote H100 job trains two models from the same initialization:

    Muon : W <- W - lr * Normalize_R(u_t)
    hMuon: W <- Normalize_R(W - lr * Normalize_R(u_t))

The only difference is the final Hyperball projection.
"""

from __future__ import annotations

from pathlib import Path

import modal

from utils import PROJECT_ROOT, timestamped_output_dir, write_histories_csv, write_text_log


app = modal.App("compare-muon-hmuon-h100")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "matplotlib")
    .add_local_dir(str(PROJECT_ROOT / "models"), "/root/models")
    .add_local_dir(str(PROJECT_ROOT / "optimizers"), "/root/optimizers")
)


@app.function(gpu="H100", image=image, timeout=60 * 60)
def train_on_h100(
    steps: int = 1000,
    batch_size: int = 2048,
    lr: float = 0.01,
    seed: int = 0,
) -> dict:
    import sys
    import time

    sys.path.insert(0, "/root")

    import torch
    import torch.nn.functional as F

    from models.simple_mlp import init_mlp_weights, make_teacher_dataset, mlp_forward
    from optimizers.hmuon import hmuon_step, normalize_R
    from optimizers.muon import muon_step

    device = torch.device("cuda")
    torch.manual_seed(seed)

    n_train = 131_072
    n_val = 16_384
    d_in = 128
    hidden = 512
    d_out = 10
    beta = 0.95

    teacher_gen = torch.Generator(device=device).manual_seed(seed + 10)
    teacher_weights = init_mlp_weights(d_in, hidden, d_out, teacher_gen, device=device)

    data_gen = torch.Generator(device=device).manual_seed(seed)
    x_train, y_train = make_teacher_dataset(n_train, d_in, teacher_weights, data_gen, device=device)
    x_val, y_val = make_teacher_dataset(n_val, d_in, teacher_weights, data_gen, device=device)

    init_gen = torch.Generator(device=device).manual_seed(seed + 1)
    initial_weights = init_mlp_weights(d_in, hidden, d_out, init_gen, device=device)

    def evaluate(weights):
        with torch.no_grad():
            logits = mlp_forward(weights, x_val)
            loss = F.cross_entropy(logits, y_val).item()
            acc = (logits.argmax(dim=1) == y_val).float().mean().item()
        return loss, acc

    def train_one(method: str):
        weights = [w.detach().clone().requires_grad_(True) for w in initial_weights]
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

        start_time = time.perf_counter()
        for step in range(1, steps + 1):
            idx = torch.randint(n_train, (batch_size,), device=device)
            xb = x_train[idx]
            yb = y_train[idx]

            step_start = time.perf_counter()
            loss = F.cross_entropy(mlp_forward(weights, xb), yb)
            loss.backward()

            with torch.no_grad():
                update_norm_sum = 0.0
                for i, weight in enumerate(weights):
                    if method == "muon":
                        new_weight, buffers[i], update = muon_step(
                            param=weight,
                            grad=weight.grad,
                            momentum_buffer=buffers[i],
                            lr=lr,
                            momentum=beta,
                            adjust_lr_mode="none",
                            update_radius=radii[i],
                        )
                    elif method == "hmuon":
                        new_weight, buffers[i], update = hmuon_step(
                            param=weight,
                            grad=weight.grad,
                            momentum_buffer=buffers[i],
                            lr=lr,
                            R=radii[i],
                            momentum=beta,
                        )
                    else:
                        raise ValueError(method)

                    update_norm_sum += normalize_R(update, radii[i]).norm().item()
                    weight.copy_(new_weight)
                    weight.grad = None

            if step == 1 or step % 50 == 0 or step == steps:
                val_loss, val_acc = evaluate(weights)
                total_norm = sum(w.detach().norm().item() for w in weights)
                history["step"].append(step)
                history["train_loss"].append(loss.item())
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["weight_norm"].append(total_norm)
                history["update_norm"].append(update_norm_sum)
                history["sec_per_step"].append((time.perf_counter() - start_time) / step)

            torch.cuda.synchronize()

        return history

    return {
        "device": torch.cuda.get_device_name(0),
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "muon": train_one("muon"),
        "hmuon": train_one("hmuon"),
    }


def save_plots(result: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    muon = result["muon"]
    hmuon = result["hmuon"]

    metrics = [
        ("train_loss", "Train loss"),
        ("val_loss", "Validation loss"),
        ("val_acc", "Validation accuracy"),
        ("weight_norm", "Sum of weight Frobenius norms"),
        ("update_norm", "Sum of normalized update norms"),
        ("sec_per_step", "Seconds per step"),
    ]

    for key, title in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(muon["step"], muon[key], label="Muon", linewidth=2)
        ax.plot(hmuon["step"], hmuon[key], label="hMuon", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("step")
        if key == "update_norm":
            ymax = max(max(muon[key]), max(hmuon[key]))
            ax.set_ylim(0.0, ymax * 1.1)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{key}.png", dpi=160)
        plt.close(fig)


@app.local_entrypoint()
def main(
    steps: int = 1000,
    batch_size: int = 2048,
    lr: float = 0.01,
    seed: int = 0,
    output_dir: str = "results",
) -> None:
    output_path = timestamped_output_dir("compare_muon_hmuon_modal", output_dir)
    result = train_on_h100.remote(
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )
    save_plots(result, output_path)
    write_histories_csv(
        output_path / "history.csv",
        {"muon": result["muon"], "hmuon": result["hmuon"]},
        metadata={
            "steps": result["steps"],
            "batch_size": result["batch_size"],
            "lr": result["lr"],
            "seed": seed,
            "device": result["device"],
        },
    )

    summary_lines = [
        f"Device: {result['device']}",
        f"steps={result['steps']} batch_size={result['batch_size']} lr={result['lr']}",
    ]
    for name in ("muon", "hmuon"):
        hist = result[name]
        summary_lines.append(
            f"{name:5s}: "
            f"train_loss={hist['train_loss'][-1]:.4f}, "
            f"val_loss={hist['val_loss'][-1]:.4f}, "
            f"val_acc={hist['val_acc'][-1]:.4f}, "
            f"sec_per_step={hist['sec_per_step'][-1]:.4f}"
        )

    write_text_log(output_path / "terminal.log", summary_lines)
    print("\n".join(summary_lines))
    print(f"Saved plots, terminal log, and CSV values to {output_path.resolve()}")
