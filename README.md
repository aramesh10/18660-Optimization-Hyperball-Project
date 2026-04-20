# CMU 18660 Optimization

This repository contains small experiments for comparing Muon-style optimizers:

- `optimizers/` implements Muon and hMuon update rules.
- `models/` contains simple model definitions used by the experiments.
- `src/` contains runnable scripts for local and Modal-based comparisons.
- `results/` stores generated plots and logs. Each script writes into a timestamped subfolder so multiple runs do not overwrite each other.

## Setup

Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

You will also need Modal configured locally before running the H100 scripts.

## Run Local Experiments

Compare Muon and hMuon on a synthetic MLP task:

```powershell
python src\compare_muon_hmuon.py
```

Compare this repository's Muon implementation against `torch.optim.Muon`:

```powershell
python src\compare_muon_pytorch.py
```

Both scripts save outputs under `results/<script-name>/<timestamp>/`.

## Run Modal Experiments

Run the larger Muon vs hMuon MLP comparison on a Modal H100:

```powershell
modal run src\compare_muon_hmuon_modal.py
```

Run the NanoGPT comparison on Modal:

```powershell
modal run src\train_nanogpt_modal.py
```

Optional NanoGPT datasets include `fineweb`, `wikitext103`, and `shakespeare`:

```powershell
modal run src\train_nanogpt_modal.py --dataset shakespeare
```

## Outputs

By default, scripts save into timestamped folders like:

```text
results/compare_muon_hmuon/20260420_191234/
results/train_nanogpt_modal/20260420_191500/
```

Each run folder includes:

- `terminal.log` with the terminal output from the run.
- One or more `.csv` files with the recorded training or comparison values.
- `.png` plots for scripts that generate figures.

`compare_muon_pytorch.py` also writes `comparison.txt` with the printed numerical comparison.
