"""Train NanoGPT (GPT-2 124M) on text data with Muon vs hMuon on Modal H100.

Both runs start from identical random initialization and train on the same data.
Results: train loss, val loss, and MFU curves saved to results_nanogpt/.

Datasets
--------
  fineweb      – FineWeb-Edu 10BT sample, streams 100M train tokens (default)
  wikitext103  – WikiText-103 (~103M tokens, tiktoken BPE)
  shakespeare  – TinyShakespeare (~1 MB, char-level, fast to load)

Usage:
    modal run train_nanogpt_modal.py
    modal run train_nanogpt_modal.py --dataset wikitext103
    modal run train_nanogpt_modal.py --dataset shakespeare
    modal run train_nanogpt_modal.py --steps 5000 --lr-muon 0.02
"""

from __future__ import annotations
from pathlib import Path
import modal

app = modal.App("nanogpt-muon-vs-hmuon")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "matplotlib", "requests", "tiktoken", "datasets")
    .add_local_dir("models", "/root/models")
    .add_local_dir("optimizers", "/root/optimizers")
)


# ---------------------------------------------------------------------------
# Dataset loaders  (called inside the remote function)
# ---------------------------------------------------------------------------

def load_shakespeare(device):
    """Character-level TinyShakespeare. Returns (train, val, vocab_size)."""
    import requests, torch
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
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
    TRAIN_TOKENS = 100_000_000
    VAL_TOKENS   =   1_000_000

    def stream_encode(n_tokens):
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
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

@app.function(gpu="H100", image=image, timeout=3 * 60 * 60)
def train(
    dataset: str = "fineweb",
    steps: int = 2000,
    batch_size: int = 16,
    block_size: int = 1024,
    lr_muon: float = 0.02,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eval_interval: int = 25,
    seed: int = 42,
) -> dict:
    import sys, time
    import torch
    import numpy as np

    sys.path.insert(0, "/root")
    from models.nanogpt import GPT, GPTConfig
    from optimizers.muon import muon_step
    from optimizers.hmuon import hmuon_step

    assert dataset in LOADERS, f"dataset must be one of {list(LOADERS)}"

    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    train_data, val_data, vocab_size = LOADERS[dataset](device)

    def get_batch(split):
        src = train_data if split == "train" else val_data
        ix = torch.randint(len(src) - block_size, (batch_size,))
        x = torch.stack([src[i : i + block_size] for i in ix])
        y = torch.stack([src[i + 1 : i + block_size + 1] for i in ix])
        return x, y

    cfg = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=12, n_head=12, n_embd=768,
        dropout=0.0, bias=True,
    )

    def make_model():
        torch.manual_seed(seed)
        m = GPT(cfg).to(device)
        return torch.compile(m)

    def make_muon_state(model):
        buffers, radii = {}, {}
        for name, p in model._orig_mod.named_parameters():
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

        # 1-D params (biases, LayerNorm) use AdamW — standard Muon recipe
        onedim_params = [
            p for p in model._orig_mod.parameters()
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
                for name, p in model._orig_mod.named_parameters():
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
                mfu = model._orig_mod.estimate_mfu(batch_size, dt)
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
        "steps": steps,
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
        ax.set_title(f"GPT-2 124M ({result['dataset']}) — {title}")
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
    steps: int = 2000,
    batch_size: int = 16,
    block_size: int = 1024,
    lr_muon: float = 0.02,
    seed: int = 42,
    output_dir: str = "results_nanogpt",
) -> None:
    result = train.remote(
        dataset=dataset,
        steps=steps,
        batch_size=batch_size,
        block_size=block_size,
        lr_muon=lr_muon,
        seed=seed,
    )

    out = Path(output_dir)
    save_plots(result, out)

    print(f"\nDevice  : {result['device']}")
    print(f"Dataset : {result['dataset']}")
    print(f"Steps   : {result['steps']}")
    print(f"\n{'Optimizer':<8} {'Final train':>12} {'Final val':>10}")
    print("-" * 35)
    for name in ("muon", "hmuon"):
        h = result[name]
        print(f"{name:<8} {h['train_loss'][-1]:>12.4f} {h['val_loss'][-1]:>10.4f}")
    print(f"\nPlots saved to {out.resolve()}/")
