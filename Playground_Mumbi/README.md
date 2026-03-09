# Playground_Mumbi

**Mumbi Whidby** — C147/247 Final Project, Winter 2026

## Overview

This playground investigates two things:
1. A **CNN+LSTM hybrid architecture** as an alternative to the TDS CNN baseline
2. A **training data fraction ablation** on the TDS CNN to understand how much labeled data is actually needed

---

## Directory Structure

```
Playground_Mumbi/
├── data_utils.py           # DataLoader factory with train-fraction support
├── model.py                # CNN+LSTM hybrid model
├── train.py                # Training & evaluation script
├── hyperparam_tuner.py     # Two-phase hyperparameter search
├── plot_results.py         # Ablation visualization
└── checkpoints/
    ├── best_hyperparams_cnn_lstm.yaml
    └── final_models/
        ├── best_cnn_training_fraction_ablation.pt
        └── best_cnnlstm_baseline_150.pt
    └── training_fraction_ablation/
        ├── best_cnn_10pct.pt
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
python -m Playground_Mumbi.train --model cnn_lstm --resume Playground_Mumbi/checkpoints/final_models/best_cnnlstm_baseline_150.pt --epochs 150
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

| Model | Epochs | Val CER | Test CER |
|-------|--------|---------|----------|
| TDS CNN (100% data) | 80 | — | — |
| CNN+LSTM hybrid | 150 | 16.0% | 18.9% |

Training fraction ablation results in `results/results_summary_CNN_training_fraction_ablation.csv`.
