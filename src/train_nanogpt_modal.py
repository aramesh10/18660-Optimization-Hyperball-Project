"""Train NanoGPT (GPT-2 124M) on text data with Muon vs hMuon on Modal H100.

Both runs start from identical random initialization and train on the same data.
Results: train loss, val loss, and MFU curves saved to timestamped folders under results/.

Datasets
--------
  fineweb      – FineWeb-Edu 10BT sample, streams 100M train tokens (default)
  wikitext103  – WikiText-103 (~103M tokens, tiktoken BPE)
  shakespeare  – TinyShakespeare (~1 MB, char-level, fast to load)

Usage:
    modal run src/train_nanogpt_modal.py
    modal run src/train_nanogpt_modal.py --dataset wikitext103
    modal run src/train_nanogpt_modal.py --dataset shakespeare
    modal run src/train_nanogpt_modal.py --steps 5000 --lr-muon 0.02
"""

from __future__ import annotations
from pathlib import Path
import traceback
import modal

try:
    from utils import PROJECT_ROOT, timestamped_output_dir, write_histories_csv, write_text_log
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def timestamped_output_dir(run_name: str, output_root: str | Path) -> Path:
        raise RuntimeError("timestamped_output_dir is only available from the local entrypoint.")

    def write_histories_csv(path: Path, histories: dict[str, dict], metadata: dict | None = None) -> None:
        raise RuntimeError("write_histories_csv is only available from the local entrypoint.")

    def write_text_log(path: Path, lines) -> None:
        raise RuntimeError("write_text_log is only available from the local entrypoint.")

app = modal.App("nanogpt-muon-vs-hmuon")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "matplotlib", "requests", "tiktoken", "datasets")
    .add_local_dir(str(PROJECT_ROOT / "src"), "/root/src")
    .add_local_dir(str(PROJECT_ROOT / "models"), "/root/models")
    .add_local_dir(str(PROJECT_ROOT / "optimizers"), "/root/optimizers")
)


# ---------------------------------------------------------------------------
# Dataset loaders  (called inside the remote function)
# ---------------------------------------------------------------------------

def load_shakespeare(device):
    """Character-level TinyShakespeare. Returns (train, val, vocab_size)."""
    import requests, torch
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    text = response.text
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def load_wikitext103(device):
    """WikiText-103 tokenised with GPT-2 BPE (tiktoken). Returns (train, val, vocab_size)."""
    import torch, tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    def encode_split(split_name):
        tokens = []
        for row in ds[split_name]["text"]:
            if row.strip():
                tokens.extend(enc.encode_ordinary(row))
        return torch.tensor(tokens, dtype=torch.long, device=device)

    train_data = encode_split("train")
    val_data   = encode_split("validation")
    vocab_size = enc.n_vocab  # 50257
    print(f"WikiText-103: {len(train_data):,} train tokens, {len(val_data):,} val tokens")
    return train_data, val_data, vocab_size


def load_fineweb(device):
    """FineWeb-Edu (10B token sample) tokenised with GPT-2 BPE. Returns (train, val, vocab_size).

    Uses the HuggingFace streaming API so only the tokens needed are downloaded.
    Train: first 100M tokens; val: next 1M tokens.
    """
    import torch, tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    TRAIN_TOKENS = 2_500_000_000
    VAL_TOKENS   =     5_000_000

    def stream_encode(n_tokens):
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-100BT",
            split="train",
            streaming=True,
        )
        buf = []
        for row in ds:
            buf.extend(enc.encode_ordinary(row["text"]))
            if len(buf) >= n_tokens:
                break
        return torch.tensor(buf[:n_tokens], dtype=torch.long, device=device)

    train_data = stream_encode(TRAIN_TOKENS)
    val_data   = stream_encode(VAL_TOKENS)
    vocab_size = enc.n_vocab  # 50257
    print(f"FineWeb-Edu: {len(train_data):,} train tokens, {len(val_data):,} val tokens")
    return train_data, val_data, vocab_size


LOADERS = {
    "shakespeare": load_shakespeare,
    "wikitext103": load_wikitext103,
    "fineweb":     load_fineweb,
}


# ---------------------------------------------------------------------------
# Remote training function
# ---------------------------------------------------------------------------

@app.function(gpu="B200", image=image, timeout=12 * 60 * 60, secrets=[modal.Secret.from_name("huggingface")])
def train(
    dataset: str = "fineweb",
    model_size: str = "small",
    steps: int = 38147,
    batch_size: int = 4,
    block_size: int = 16384,
    lr_muon: float = 0.0025,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eval_interval: int = 100,
    seed: int = 42,
    compile_model: bool = True,
) -> dict:
    import sys, time
    import torch
    import numpy as np

    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root")
    from models.nanogpt import GPT, GPTConfig
    from optimizers.muon import muon_step
    from optimizers.hmuon import hmuon_step

    if dataset not in LOADERS:
        raise ValueError(f"dataset must be one of {list(LOADERS)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Modal container. Check the Modal GPU request/image.")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    if eval_interval < 1:
        raise ValueError("eval_interval must be at least 1")

    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    train_data, val_data, vocab_size = LOADERS[dataset](device)
    if len(train_data) <= block_size or len(val_data) <= block_size:
        raise ValueError(
            f"dataset is too small for block_size={block_size}: "
            f"train tokens={len(train_data)}, val tokens={len(val_data)}"
        )

    def get_batch(split):
        src = train_data if split == "train" else val_data
        ix = torch.randint(len(src) - block_size, (batch_size,), device=device)
        offsets = torch.arange(block_size, device=device)
        positions = ix[:, None] + offsets[None, :]
        x = src[positions]
        y = src[positions + 1]
        return x, y

    MODEL_SIZES = {
        "small":  dict(n_layer=12, n_head=12, n_embd=768),   # 124M
        "medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
        "large":  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
        "xl":     dict(n_layer=48, n_head=25, n_embd=1600),  # 1.6B
    }
    if model_size not in MODEL_SIZES:
        raise ValueError(f"model_size must be one of {list(MODEL_SIZES)}")
    arch = MODEL_SIZES[model_size]

    cfg = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=arch["n_layer"], n_head=arch["n_head"], n_embd=arch["n_embd"],
        dropout=0.0, bias=True,
    )

    def make_model():
        torch.manual_seed(seed)
        m = GPT(cfg).to(device)
        if compile_model:
            return torch.compile(m)
        return m

    def unwrap_model(model):
        return getattr(model, "_orig_mod", model)

    def make_muon_state(model):
        buffers, radii = {}, {}
        for name, p in unwrap_model(model).named_parameters():
            if p.requires_grad and p.dim() >= 2:
                buffers[name] = torch.zeros_like(p)
                radii[name] = p.detach().norm()
        return buffers, radii

    # -----------------------------------------------------------------------
    # Training loop — strategy is "muon" or "hmuon"
    # -----------------------------------------------------------------------
    def train_run(strategy: str) -> dict:
        model = make_model()
        scaler = torch.amp.GradScaler("cuda")
        base_model = unwrap_model(model)

        # 1-D params (biases, LayerNorm) use AdamW — standard Muon recipe
        onedim_params = [
            p for p in base_model.parameters()
            if p.requires_grad and p.dim() < 2
        ]
        adamw_1d = torch.optim.AdamW(
            onedim_params, lr=lr_muon, betas=(beta1, beta2), weight_decay=0.0
        )
        buffers, radii = make_muon_state(model)

        history: dict[str, list] = {
            "step": [], "train_loss": [], "val_loss": [], "mfu": []
        }
        t0 = time.perf_counter()

        for step in range(1, steps + 1):
            model.train()
            x, y = get_batch("train")

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(adamw_1d)

            with torch.no_grad():
                for name, p in base_model.named_parameters():
                    if not p.requires_grad or p.grad is None or p.dim() < 2:
                        continue
                    g = p.grad.float()
                    if strategy == "muon":
                        new_p, buffers[name], _ = muon_step(
                            param=p.float(),
                            grad=g,
                            momentum_buffer=buffers[name],
                            lr=lr_muon,
                            momentum=beta1,
                            weight_decay=weight_decay,
                            adjust_lr_mode="original",
                        )
                    else:  # hmuon
                        new_p, buffers[name], _ = hmuon_step(
                            param=p.float(),
                            grad=g,
                            momentum_buffer=buffers[name],
                            lr=lr_muon,
                            R=radii[name],
                            momentum=beta1,
                        )
                    p.copy_(new_p)
                    p.grad = None

            scaler.step(adamw_1d)
            scaler.update()
            adamw_1d.zero_grad(set_to_none=True)

            if step % eval_interval == 0 or step == steps:
                model.eval()
                with torch.no_grad():
                    val_losses = [model(*get_batch("val"))[1].item() for _ in range(10)]
                dt = (time.perf_counter() - t0) / step
                mfu = base_model.estimate_mfu(batch_size, dt)
                history["step"].append(step)
                history["train_loss"].append(loss.item())
                history["val_loss"].append(float(np.mean(val_losses)))
                history["mfu"].append(mfu)
                print(
                    f"[{strategy}] step {step:4d} | "
                    f"train {loss.item():.4f} | "
                    f"val {history['val_loss'][-1]:.4f} | "
                    f"mfu {mfu * 100:.2f}%"
                )

        return history

    return {
        "device": torch.cuda.get_device_name(0),
        "dataset": dataset,
        "model_size": model_size,
        "steps": steps,
        "batch_size": batch_size,
        "block_size": block_size,
        "lr_muon": lr_muon,
        "compile_model": compile_model,
        "muon": train_run("muon"),
        "hmuon": train_run("hmuon"),
    }


# ---------------------------------------------------------------------------
# Local: save comparison plots
# ---------------------------------------------------------------------------

def save_plots(result: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("train_loss", "Train loss"),
        ("val_loss",   "Val loss"),
        ("mfu",        "Model FLOPs utilization"),
    ]
    for key, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, color in [("muon", "C1"), ("hmuon", "C2")]:
            h = result[name]
            ax.plot(h["step"], h[key], label=name, color=color, linewidth=2)
        ax.set_title(f"GPT-2 {result['model_size']} ({result['dataset']}) — {title}")
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{key}.png", dpi=160)
        plt.close(fig)
        print(f"  saved {out_dir / key}.png")


@app.local_entrypoint()
def main(
    dataset: str = "fineweb",
    model_size: str = "small",
    steps: int = 38147,
    batch_size: int = 4,
    block_size: int = 16384,
    lr_muon: float = 0.0025,
    seed: int = 42,
    compile_model: bool = True,
    output_dir: str = "results",
) -> None:
    out = timestamped_output_dir("train_nanogpt_modal", output_dir)
    try:
        result = train.remote(
            dataset=dataset,
            model_size=model_size,
            steps=steps,
            batch_size=batch_size,
            block_size=block_size,
            lr_muon=lr_muon,
            seed=seed,
            compile_model=compile_model,
        )
    except Exception:
        error_path = out / "error.log"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Remote run failed. Saved local traceback to {error_path.resolve()}")
        raise

    save_plots(result, out)
    write_histories_csv(
        out / "history.csv",
        {"muon": result["muon"], "hmuon": result["hmuon"]},
        metadata={
            "dataset": result["dataset"],
            "model_size": model_size,
            "steps": result["steps"],
            "batch_size": batch_size,
            "block_size": block_size,
            "lr_muon": lr_muon,
            "seed": seed,
            "device": result["device"],
            "compile_model": result["compile_model"],
        },
    )

    summary_lines = [
        f"Device  : {result['device']}",
        f"Dataset : {result['dataset']}",
        f"Model   : {model_size}",
        f"Steps   : {result['steps']}",
        "",
        f"{'Optimizer':<8} {'Final train':>12} {'Final val':>10}",
        "-" * 35,
    ]
    for name in ("muon", "hmuon"):
        h = result[name]
        summary_lines.append(f"{name:<8} {h['train_loss'][-1]:>12.4f} {h['val_loss'][-1]:>10.4f}")

    write_text_log(out / "terminal.log", summary_lines)
    print("\n".join(["", *summary_lines]))
    print(f"\nPlots, terminal log, and CSV values saved to {out.resolve()}/")


# ---------------------------------------------------------------------------
# Adam vs AdamH training function
# ---------------------------------------------------------------------------

@app.function(gpu="B200", image=image, timeout=12 * 60 * 60, secrets=[modal.Secret.from_name("huggingface")])
def train_adam(
    dataset: str = "fineweb",
    model_size: str = "small",
    steps: int = 38147,
    batch_size: int = 4,
    block_size: int = 16384,
    lr: float = 0.0025,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    eval_interval: int = 100,
    seed: int = 42,
    compile_model: bool = True,
) -> dict:
    import sys, time
    import torch
    import numpy as np

    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root")
    from models.nanogpt import GPT, GPTConfig
    from optimizers.adam import adam_step
    from optimizers.hadamw import hadamw_step

    if dataset not in LOADERS:
        raise ValueError(f"dataset must be one of {list(LOADERS)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    train_data, val_data, vocab_size = LOADERS[dataset](device)

    def get_batch(split):
        src = train_data if split == "train" else val_data
        ix = torch.randint(len(src) - block_size, (batch_size,), device=device)
        offsets = torch.arange(block_size, device=device)
        positions = ix[:, None] + offsets[None, :]
        x = src[positions]
        y = src[positions + 1]
        return x, y

    MODEL_SIZES = {
        "small":  dict(n_layer=12, n_head=12, n_embd=768),
        "medium": dict(n_layer=24, n_head=16, n_embd=1024),
        "large":  dict(n_layer=36, n_head=20, n_embd=1280),
        "xl":     dict(n_layer=48, n_head=25, n_embd=1600),
    }
    if model_size not in MODEL_SIZES:
        raise ValueError(f"model_size must be one of {list(MODEL_SIZES)}")
    arch = MODEL_SIZES[model_size]

    cfg = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=arch["n_layer"], n_head=arch["n_head"], n_embd=arch["n_embd"],
        dropout=0.0, bias=True,
    )

    def make_model():
        torch.manual_seed(seed)
        m = GPT(cfg).to(device)
        return torch.compile(m) if compile_model else m

    def unwrap_model(model):
        return getattr(model, "_orig_mod", model)

    def train_run(strategy: str) -> dict:
        model = make_model()
        base_model = unwrap_model(model)

        # Adam state: first and second moment for every parameter
        exp_avgs, exp_avg_sqs, radii = {}, {}, {}
        for name, p in base_model.named_parameters():
            if p.requires_grad:
                exp_avgs[name] = torch.zeros_like(p, dtype=torch.float32)
                exp_avg_sqs[name] = torch.zeros_like(p, dtype=torch.float32)
                if strategy == "hadamw" and p.dim() >= 2:
                    radii[name] = p.detach().norm()

        history: dict[str, list] = {"step": [], "train_loss": [], "val_loss": [], "mfu": []}
        t0 = time.perf_counter()

        for step in range(1, steps + 1):
            model.train()
            x, y = get_batch("train")

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)

            loss.backward()

            with torch.no_grad():
                for name, p in base_model.named_parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    g = p.grad.float()
                    if strategy == "adam":
                        new_p, exp_avgs[name], exp_avg_sqs[name] = adam_step(
                            param=p.float(),
                            grad=g,
                            exp_avg=exp_avgs[name],
                            exp_avg_sq=exp_avg_sqs[name],
                            step=step,
                            lr=lr,
                            beta1=beta1,
                            beta2=beta2,
                            eps=eps,
                            weight_decay=weight_decay if p.dim() >= 2 else 0.0,
                        )
                    else:  # hadamw — only project 2D params, use adam for 1D
                        if p.dim() >= 2:
                            new_p, exp_avgs[name], exp_avg_sqs[name] = hadamw_step(
                                param=p.float(),
                                grad=g,
                                exp_avg=exp_avgs[name],
                                exp_avg_sq=exp_avg_sqs[name],
                                step=step,
                                lr=lr,
                                R=radii[name],
                                beta1=beta1,
                                beta2=beta2,
                                eps=eps,
                            )
                        else:
                            new_p, exp_avgs[name], exp_avg_sqs[name] = adam_step(
                                param=p.float(),
                                grad=g,
                                exp_avg=exp_avgs[name],
                                exp_avg_sq=exp_avg_sqs[name],
                                step=step,
                                lr=lr,
                                beta1=beta1,
                                beta2=beta2,
                                eps=eps,
                                weight_decay=0.0,
                            )
                    p.copy_(new_p)
                    p.grad = None

            if step % eval_interval == 0 or step == steps:
                model.eval()
                with torch.no_grad():
                    val_losses = [model(*get_batch("val"))[1].item() for _ in range(10)]
                dt = (time.perf_counter() - t0) / step
                mfu = base_model.estimate_mfu(batch_size, dt)
                history["step"].append(step)
                history["train_loss"].append(loss.item())
                history["val_loss"].append(float(np.mean(val_losses)))
                history["mfu"].append(mfu)
                print(
                    f"[{strategy}] step {step:4d} | "
                    f"train {loss.item():.4f} | "
                    f"val {history['val_loss'][-1]:.4f} | "
                    f"mfu {mfu * 100:.2f}%"
                )

        return history

    return {
        "device": torch.cuda.get_device_name(0),
        "dataset": dataset,
        "model_size": model_size,
        "steps": steps,
        "batch_size": batch_size,
        "block_size": block_size,
        "lr": lr,
        "compile_model": compile_model,
        "adam": train_run("adam"),
        "hadamw": train_run("hadamw"),
    }


def save_plots_adam(result: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("train_loss", "Train loss"),
        ("val_loss",   "Val loss"),
        ("mfu",        "Model FLOPs utilization"),
    ]
    for key, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, color in [("adam", "C0"), ("hadamw", "C3")]:
            h = result[name]
            ax.plot(h["step"], h[key], label=name, color=color, linewidth=2)
        ax.set_title(f"GPT-2 {result['model_size']} ({result['dataset']}) — {title}")
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{key}.png", dpi=160)
        plt.close(fig)
        print(f"  saved {out_dir / key}.png")


@app.local_entrypoint()
def main_adam(
    dataset: str = "fineweb",
    model_size: str = "small",
    steps: int = 38147,
    batch_size: int = 4,
    block_size: int = 16384,
    lr: float = 0.0025,
    weight_decay: float = 0.1,
    seed: int = 42,
    compile_model: bool = True,
    output_dir: str = "results",
) -> None:
    out = timestamped_output_dir("train_adam_modal", output_dir)
    try:
        result = train_adam.remote(
            dataset=dataset,
            model_size=model_size,
            steps=steps,
            batch_size=batch_size,
            block_size=block_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            compile_model=compile_model,
        )
    except Exception:
        error_path = out / "error.log"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Remote run failed. Saved local traceback to {error_path.resolve()}")
        raise

    save_plots_adam(result, out)
    write_histories_csv(
        out / "history.csv",
        {"adam": result["adam"], "hadamw": result["hadamw"]},
        metadata={
            "dataset": result["dataset"],
            "model_size": model_size,
            "steps": result["steps"],
            "batch_size": batch_size,
            "block_size": block_size,
            "lr": lr,
            "seed": seed,
            "device": result["device"],
            "compile_model": result["compile_model"],
        },
    )

    summary_lines = [
        f"Device  : {result['device']}",
        f"Dataset : {result['dataset']}",
        f"Model   : {model_size}",
        f"Steps   : {result['steps']}",
        "",
        f"{'Optimizer':<8} {'Final train':>12} {'Final val':>10}",
        "-" * 35,
    ]
    for name in ("adam", "hadamw"):
        h = result[name]
        summary_lines.append(f"{name:<8} {h['train_loss'][-1]:>12.4f} {h['val_loss'][-1]:>10.4f}")

    write_text_log(out / "terminal.log", summary_lines)
    print("\n".join(["", *summary_lines]))
    print(f"\nPlots, terminal log, and CSV values saved to {out.resolve()}/")
