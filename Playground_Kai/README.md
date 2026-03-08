# Playground_Kai

He Kai Lim — UCLA ECE C247, Winter 2026

---

## Project Parameters

- **Do not modify** anything under `emg2qwerty/`, but its contents may be used freely.
- All model code lives in `Playground_Kai/`.
- **Data**: single-user HDF5 files under `data/`. Train/val/test split defined in `config/user/single_user.yaml`.
- **Input**: log-spectrogram of raw EMG (`LogSpectrogram(n_fft=64, hop_length=16)`) → shape `(T, N, 2, 16, 33)`.
- **Output**: CTC log-softmax activations → shape `(T, N, num_classes)`.
- **Metric**: Character Error Rate (CER ↓).

---

## Quick Reference: Run Test Evaluation on a Pre-Trained Model

Loads the saved best checkpoint and evaluates on the held-out test set. No training occurs. Results are printed and appended to `results/results_summary_{MODEL}.csv`.

**RNN:**
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.train --model rnn --test-only
```

**Conformer:**
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.train --model conformer --test-only
```

> Requires `Playground_Kai/checkpoints/best_rnn.pt` or `best_conformer.pt` to exist (produced by a prior training run).

**Hyperparam Tuning**
* Raw pipeline (unchanged behaviour)
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.hyperparam_tuner --model conformer --coarse-trials 25 --coarse-epochs 15 --fine-top-k 1 --fine-trials 15 --fine-epochs 15 --trial-sessions 8 --early-stopping-patience 10 --trial-timeout 300
```

* Preprocessed pipeline
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.hyperparam_tuner --model conformer --preprocess --coarse-trials 25 --coarse-epochs 15 --fine-top-k 1 --fine-trials 15 --fine-epochs 15 --trial-sessions 8 --early-stopping-patience 10 --trial-timeout 300
```
---

## Back-to-Back: Hyperparameter Search → Full Train

Chain both steps with `;` so the best hyperparams are fed straight into training.

### Conformer (RPE)

```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.hyperparam_tuner --model conformer --coarse-trials 25 --coarse-epochs 15 --fine-trials 15 --fine-epochs 15 --trial-sessions 8 --early-stopping-patience 10 --trial-timeout 300; .\.venv\Scripts\python.exe -m Playground_Kai.train --model conformer --from-hyperparams D:\C247_mumbikaijonathanben\Playground_Kai\checkpoints\best_hyperparams_conformer.yaml --epochs 150
```

### RNN (BiLSTM)

```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.hyperparam_tuner --model rnn --coarse-trials 30 --coarse-epochs 10 --fine-top-k 1 --fine-trials 10 --fine-epochs 10 --trial-sessions 8 --early-stopping-patience 10 --trial-timeout 300; .\.venv\Scripts\python.exe -m Playground_Kai.train --model rnn --from-hyperparams D:\C247_mumbikaijonathanben\Playground_Kai\checkpoints\best_hyperparams_rnn.yaml --epochs 150
```

---

## Hyperparameter Search (`hyperparam_tuner.py`)

Default mode is **two-phase**: a broad coarse search followed by a focused fine search around the top-K coarse configs. Use `--search-mode coarse-only` for a single-phase random search.

### Flag Reference

| Flag | Default | Description |
|---|---|---|
| `--model` | `rnn` | `rnn` or `conformer` |
| `--search-mode` | `two-phase` | `two-phase` or `coarse-only` |
| `--coarse-trials` | 20 | Configs evaluated in coarse phase (`--num-trials` is an alias) |
| `--coarse-epochs` | 8 | Epochs per coarse proxy run (`--trial-epochs` is an alias) |
| `--fine-trials` | 10 | Configs evaluated per fine-phase anchor |
| `--fine-epochs` | 15 | Epochs per fine proxy run |
| `--fine-top-k` | 3 | Top coarse configs used as fine-phase anchors |
| `--fine-shrink` | 3.0 | Log-scale shrink factor for fine bounds (higher = tighter) |
| `--confirm-epochs` | 0 (off) | Re-run overall best for N extra epochs before saving |
| `--early-stopping-patience` | 0 (off) | Stop a trial after N non-improving epochs |
| `--trial-timeout` | 180 s | Per-trial wall-clock cap (0 = disable) |
| `--trial-sessions` | 5 | Training sessions per proxy trial (max 16) |

### Example Commands

**Conformer — quick two-phase search:**
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.hyperparam_tuner --model conformer --coarse-trials 10 --coarse-epochs 5 --fine-trials 5 --fine-epochs 10 --trial-sessions 8 --early-stopping-patience 10 --trial-timeout 300
```

**Conformer — full overnight search:**
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.hyperparam_tuner --model conformer --coarse-trials 25 --coarse-epochs 15 --fine-trials 15 --fine-epochs 15 --trial-sessions 8 --early-stopping-patience 10 --trial-timeout 300
```

**RNN — full overnight search:**
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.hyperparam_tuner --model rnn --coarse-trials 30 --coarse-epochs 10 --fine-top-k 1 --fine-trials 10 --fine-epochs 10 --trial-sessions 8 --early-stopping-patience 10 --trial-timeout 300
```

**Coarse-only (classic random search):**
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.hyperparam_tuner --model conformer --search-mode coarse-only --num-trials 50 --trial-epochs 10 --trial-sessions 8 --trial-timeout 300
```

Best hyperparams are saved to `Playground_Kai/checkpoints/best_hyperparams_{rnn,conformer}.yaml`.

---

## Full Training (`train.py`)

### Flag Reference

| Flag | Default | Description |
|---|---|---|
| `--model` | `rnn` | `rnn` or `conformer` |
| `--epochs` | 80 | Total training epochs |
| `--from-hyperparams` | — | Load hyperparams from a tuner YAML |
| `--batch-size` | 32 | Batch size |
| `--lr` | 5e-4 | Peak learning rate (AdamW) |
| `--weight-decay` | 1e-2 | AdamW weight decay |
| `--warmup-epochs` | 5 | Linear LR warmup epochs |
| `--num-layers` | 2 | LSTM layers (RNN) or Conformer blocks |
| `--hidden-size` | 512 | BiLSTM hidden size per direction (RNN only) |
| `--d-model` | 256 | Feature dimension (Conformer only) |
| `--num-heads` | 4 | Attention heads (Conformer only) |
| `--conv-kernel-size` | 31 | Depthwise conv kernel (Conformer only) |
| `--resume` | — | Resume from a checkpoint `.pt` file |
| `--notes` | `""` | Free-text tag written to the CSV log |

### Example Commands

**Train Conformer with best tuned hyperparams:**
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.train --model conformer --from-hyperparams D:\C247_mumbikaijonathanben\Playground_Kai\checkpoints\best_hyperparams_conformer.yaml --epochs 150
```

**Train RNN with best tuned hyperparams:**
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.train --model rnn --from-hyperparams D:\C247_mumbikaijonathanben\Playground_Kai\checkpoints\best_hyperparams_rnn.yaml --epochs 150
```

**Train with default hyperparams (quick test):**
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.train --model rnn --epochs 10
```

**Resume a stopped training run:**
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.train --model conformer --from-hyperparams D:\C247_mumbikaijonathanben\Playground_Kai\checkpoints\best_hyperparams_conformer.yaml --epochs 150 --resume D:\C247_mumbikaijonathanben\Playground_Kai\checkpoints\best_conformer.pt
```

Best checkpoint is auto-saved to `Playground_Kai/checkpoints/best_{rnn,conformer}.pt`.

---

## Test-Only Evaluation

Loads the saved best checkpoint and runs the test set without any training.

**RNN — test only:**
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.train --model rnn --test-only
```

**Conformer — test only:**
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.train --model conformer --test-only
```

**With a custom note tag:**
```powershell
.\.venv\Scripts\python.exe -m Playground_Kai.train --model rnn --test-only --notes "final_eval"
```

> **Logger behaviour:** `--test-only` **does** write to `results/results_summary_{MODEL}.csv` via the shared `scripts/logger.py`. It logs the checkpoint's best val CER, the live test CER, and appends `"test_only"` to the notes field. Epoch curves are not written (no training occurred). Training metrics (`final_train_loss`, `final_val_loss`) are recorded as `NaN`.

---

## CSV Logging (`scripts/logger.py`)

Every completed training run (and every `--test-only` run) appends to two shared CSVs under `results/`:

| File | Contents |
|---|---|
| `results/results_summary_RNN.csv` | One row per run: hyperparams, best val CER, test CER, training time |
| `results/results_curves_RNN.csv` | One row per epoch: train loss, val loss, val CER |
| `results/results_summary_CONFORMER.csv` | Same, for Conformer runs |
| `results/results_curves_CONFORMER.csv` | Same, for Conformer runs |

Use `--notes` to tag runs for filtering (e.g. `--notes "ablation_layers"`). The `results/` directory is created automatically.

---

## Environment Setup

> Use **PowerShell** — WSL has path shenanigans.

```powershell
# Create and activate venv
python -m venv .venv
.\.venv\Scripts\activate

# Install PyTorch (CUDA 12.8 — matches cards that support <= 12.9)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install project packages
pip install -r requirements.txt
```

CUDA version matrix: https://pytorch.org/get-started/locally/

### WSL (secondary)

```bash
python3 -m venv .venvWSL
source .venvWSL/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## Deprecated

- **Anaconda setup** — `conda create` approach abandoned; venv is simpler and works with the notebooks.
- **`--test-only` with full Conformer sessions (no chunking)** — plain self-attention is O(T²); whole-session sequences OOM. Use the windowed DataLoader or enable chunked attention instead.
