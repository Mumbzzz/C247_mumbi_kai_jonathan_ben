# Playground_Mumbi

**Mumbi Whidby** — C147/247 Final Project, Winter 2026

## Overview

This playground investigates three things:
1. A **CNN+LSTM hybrid architecture** as an alternative to the TDS CNN baseline
2. A **training data fraction ablation** on the TDS CNN to understand how much labeled data is actually needed
3. A **CNN+LSTM trained on AE-reconstructed EMG (recons v3)** to evaluate whether autoencoder-reconstructed signals can substitute for raw EMG

---

## Directory Structure

```
Playground_Mumbi/
├── data_utils.py               # DataLoader factory with train-fraction support
├── model.py                    # CNN+LSTM hybrid model (raw/biophys EMG)
├── train.py                    # Training & evaluation script (raw/biophys EMG)
├── train_recons.py             # Training on AE-reconstructed EMG (recons v3)
├── hyperparam_tuner.py         # Two-phase hyperparameter search (raw/biophys)
├── hyperparam_tuner_recons.py  # Two-phase hyperparameter search (recons v3)
├── plot_results.py             # Ablation visualization
├── scripts/
│   ├── plot_channel_ablation.py
│   └── run_channel_ablation.sh
└── checkpoints/
    ├── best_hyperparams_cnn_lstm.yaml
    └── final_models/
        ├── best_cnnlstm.pt               # CNN+LSTM baseline (spectrogram)
        ├── best_cnnlstm_biophys.pt       # CNN+LSTM with biophys preprocessing
        ├── best_cnnlstm_recons_v3.pt     # CNN+LSTM on AE-reconstructed data
        └── best_cnn_training_fraction_ablation.pt
    └── training_fraction_ablation/
        ├── best_cnn_25pct.pt
        ├── best_cnn_50pct.pt
        └── best_cnn_75pct.pt
```

---

## Models

### TDS CNN (Baseline)
The shared TDS Conv CTC baseline from `config/model/tds_conv_ctc.yaml`. Used for the training fraction ablation study.

### CNN+LSTM Hybrid
Defined in `model.py`. Architecture:
- **Preprocessing:** SpectrogramNorm + MultiBandRotationInvariantMLP (same as baseline)
- **1D CNN front-end:** Stacked Conv1d blocks with BatchNorm, GELU, dropout (same-padding, preserves time)
- **BiLSTM encoder:** Bidirectional LSTM for long-range sequential dependencies
- **Head:** Linear + LogSoftmax for CTCLoss

**Key hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cnn_channels` | 256 | Feature maps per CNN block |
| `cnn_kernel` | 5 | 1D conv kernel size |
| `cnn_layers` | 2 | Number of stacked CNN blocks |
| `lstm_hidden` | 256 | BiLSTM hidden size per direction |
| `lstm_layers` | 2 | Stacked BiLSTM layers |
| `dropout` | 0.3 | Applied after CNN and between LSTM layers |

**Input:** `(T, N, 2, 16, freq)` — time-first log-spectrograms (2 wrists, 16 electrodes)
**Output:** `(T, N, num_classes)` — log-softmax CTC emissions

### CNN+LSTM on Recons v3
Trained via `train_recons.py` on AE-reconstructed EMG from `data/89335547_recons_v3/`. Same CNN+LSTM architecture but operates on reconstructed signals (32 channels, ~62.5 Hz effective rate) with a learned linear projection in place of the standard SpectrogramNorm + MLP front-end.

**Key differences from baseline:**
- `--data-dir`: points to `data/89335547_recons_v3/`
- `--window-length 250`: matches the reconstructed signal's temporal resolution
- `--proj-features 384`: linear input projection replacing the MLP front-end

---

## Quick Start

### Train TDS CNN baseline (full data)
```bash
python -m Playground_Mumbi.train
```

### Train CNN+LSTM hybrid
```bash
python -m Playground_Mumbi.train --model cnn_lstm --epochs 150
```

### Resume from checkpoint
```bash
python -m Playground_Mumbi.train --model cnn_lstm --resume Playground_Mumbi/checkpoints/final_models/best_cnnlstm.pt --epochs 150
```

### Train CNN+LSTM on AE-reconstructed EMG (recons v3)
```bash
python -m Playground_Mumbi.train_recons --epochs 150
```

### Resume recons v3 training
```bash
python -m Playground_Mumbi.train_recons --resume Playground_Mumbi/checkpoints/final_models/best_cnnlstm_recons_v3.pt --epochs 150
```

### Run training fraction ablation (10%, 25%, 50%, 75%, 100%)
```bash
python -m Playground_Mumbi.train --run-all-fractions
```

### Run a single fraction
```bash
python -m Playground_Mumbi.train --train-fraction 0.5
```

---

## Hyperparameter Tuning

Two-phase search (coarse → fine) using random sampling:

```bash
python -m Playground_Mumbi.hyperparam_tuner \
    --coarse-trials 20 --coarse-epochs 8 \
    --fine-trials 10 --fine-epochs 15 --fine-top-k 3
```

Best hyperparams are saved to `checkpoints/best_hyperparams_cnn_lstm.yaml` and can be loaded in `train.py` via `--from-hyperparams`.

**Search space:**

| Parameter | Range |
|-----------|-------|
| `lr` | log-uniform [1e-4, 1e-2] |
| `cnn_channels` | categorical |
| `cnn_kernel` | categorical |
| `cnn_layers` | 1–3 |
| `lstm_hidden` | categorical |
| `lstm_layers` | categorical |
| `dropout` | uniform [0.1, 0.5] |
| `weight_decay` | log-uniform [1e-5, 1e-1] |

---

## Training Details

- **Loss:** CTCLoss with greedy decoding for val/test CER
- **LR schedule:** Linear warmup (5 epochs) + cosine decay
- **Gradient clipping:** max norm 5.0
- **Metric:** Character Error Rate (CER ↓)
- **Logging:** Appends to `results/results_summary_*.csv` and `results/results_curves_*.csv`

---

## Plotting

```bash
# CNN training fraction ablation plots
python -m Playground_Mumbi.plot_results --model CNN_training_fraction_ablation

# CNN+LSTM training curve
python -m Playground_Mumbi.plot_results --model CNN_LSTM_baseline_150 --display-model "CNN-LSTM (150 epochs)"
```

Outputs saved to `results/training_fraction_ablation_plots/`.

---

## Results

### Architecture Comparison

| Model | Input | Epochs | Val CER | Test CER |
|-------|-------|--------|---------|----------|
| TDS CNN (baseline) | spectrogram | 80 | 18.9% | 21.2% |
| CNN+LSTM | spectrogram | 150 | 15.8% | 19.0% |
| CNN+LSTM (biophys) | Mel spectrogram (8ch, 1kHz) | 150 | 17.9% | 21.4% |
| CNN+LSTM (recons v3) | AE-reconstructed EMG | 150 | 62.2% | 69.2% |

### TDS CNN Training Fraction Ablation

| Train Fraction | Val CER | Test CER |
|----------------|---------|----------|
| 25% | 28.2% | 30.1% |
| 50% | 23.0% | 24.9% |
| 75% | 21.1% | 22.2% |
| 100% | 18.9% | 21.2% |

Full per-epoch curves in `results/results_curves_CNN_training_fraction_ablation.csv`.
