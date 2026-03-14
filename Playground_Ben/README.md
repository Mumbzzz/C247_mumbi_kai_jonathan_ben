# Playground_Ben

Ben Forbes — C247 Final Project Contributions

## Overview

This playground contains Ben's experimental work on the `emg2qwerty` codebase. The work covers
four interconnected research directions:

1. **Channel Ablation** — how CTC decoding degrades as fewer sEMG electrode channels are used per hand
2. **Temporal Downsampling Ablation** — how performance degrades as effective sample rate is reduced
3. **Latent-Space and Reconstructed-Signal Models** — training on autoencoder outputs instead of raw EMG
4. **Hyperparameter Tuning** — two-phase Bayesian search for optimal CNN hyperparameters

All experiments target a single-user session (single-user Hydra config). A comprehensive notebook
(`notebooks/experiments_all.ipynb`) runs all experiments end-to-end.

---

## Directory Structure

```
Playground_Ben/
├── emg2qwerty/                  # Patched source files (copy over main module to enable ablations)
│   ├── lightning.py             # Dynamic electrode channel count from in_features
│   └── transforms.py           # + ChannelSelect, ChannelSubset, TemporalDownsample transforms
├── config/
│   └── transforms/              # Hydra override configs for ablation runs
│       ├── channel_stride2.yaml          # 8 ch/hand  (every other electrode)
│       ├── channel_stride4.yaml          # 4 ch/hand  (every 4th electrode)
│       ├── channel_stride8.yaml          # 2 ch/hand  (every 8th electrode)
│       ├── channel_stride2_2000hz.yaml   # 8 ch/hand at native 2000 Hz
│       ├── channel_stride4_2000hz.yaml   # 4 ch/hand at native 2000 Hz
│       ├── channel_stride8_2000hz.yaml   # 2 ch/hand at native 2000 Hz
│       ├── temporal_downsample_2.yaml    # 2× → 1000 Hz
│       ├── temporal_downsample_4.yaml    # 4× → 500 Hz
│       ├── temporal_downsample_8.yaml    # 8× → 250 Hz
│       └── temporal_downsample_16.yaml   # 16× → 125 Hz
├── scripts/
│   ├── train_latent_cnn.py       # Train CNN on AE latent vectors (emg_latent_ae_v2.hdf5)
│   ├── train_latent_cnn_sessions.py  # Variant with per-session train/val split
│   ├── train_recons_cnn.py       # Train CNN on AE-reconstructed EMG (*_recons_v3.hdf5)
│   ├── train_biophysics_cnn.py   # Train CNN with full biophysics preprocessing pipeline
│   ├── hyperparam_tuner_cnn.py   # Two-phase hyperparam search for latent CNN
│   ├── hyperparam_tuner_raw_cnn.py   # Two-phase hyperparam search for biophysics CNN
│   ├── run_channel_ablation.sh   # Orchestrate full channel ablation study
│   ├── run_temporal_ablation.sh  # Orchestrate full temporal ablation study
│   ├── analyze_emg.py            # EMG signal analysis (8 diagnostic plots)
│   ├── eval_plot.py              # Per-window inference visualization with CTC heatmap
│   ├── plot_results.py           # Multi-purpose training curves / comparison plots
│   ├── plot_channel_ablation.py  # Channel ablation result plots
│   ├── plot_sampling_ablation.py # Temporal ablation result plots
│   ├── plot_ablation_bars.py     # Combined ablation bar charts
│   ├── log_channel_results.py    # Log channel ablation metrics to CSV
│   ├── log_temporal_results.py   # Log temporal ablation metrics to CSV
│   └── find_best_temporal_factor.py  # Identify best temporal config from log dirs
├── notebooks/
│   ├── cnn_baseline.ipynb        # Original baseline-only interactive notebook
│   └── experiments_all.ipynb     # Comprehensive notebook covering all experiments
├── checkpoints/                  # Saved model weights and tuned hyperparameter YAMLs
│   ├── best_latent_cnn.pt        # Best CNN trained on AE latent vectors
│   ├── best_latent_cnn_sessions.pt  # CNN variant with per-session validation
│   ├── best_recons_cnn.pt        # Best CNN trained on AE-reconstructed EMG
│   ├── best_biophysics_cnn.pt    # Best CNN with biophysics preprocessing pipeline
│   ├── best_hyperparams_cnn_latent.yaml  # Tuned hyperparams for latent CNN
│   └── best_hyperparams_raw_cnn.yaml     # Tuned hyperparams for biophysics CNN
├── results/
│   ├── channel_ablation_table.csv    # Channel ablation metrics (2/4/8/16 ch)
│   ├── sampling_ablation_table.csv   # Temporal ablation metrics (125–2000 Hz)
│   └── ablation_tables.tex           # LaTeX table formatting
├── plots/
│   ├── eval/                     # Per-window inference examples (6 PNGs)
│   ├── emg_analysis/             # EMG signal analysis (8 PNGs)
│   ├── channel_ablation_bar.png
│   ├── channel_ablation_curves.png
│   ├── sampling_ablation_bar.png
│   ├── sampling_ablation_curves.png
│   ├── ablation_combined_bar.png
│   └── temporal_ablation.png
└── recons_data_utils.py          # Data loader for *_recons_v3.hdf5 files
```

---

## Setup

All scripts and notebooks run from the **repo root** with the venv active:

```bash
source /home/benforbes/emg2qwerty/venv/bin/activate
pip install -e .   # only needed once
cd ~/C247_mumbikaijonathanben
```

---

## Code Modifications to `emg2qwerty`

Two files in the main `emg2qwerty/` module were modified. Copy them over the originals before
running any ablation experiment.

```bash
cp Playground_Ben/emg2qwerty/lightning.py  emg2qwerty/lightning.py
cp Playground_Ben/emg2qwerty/transforms.py emg2qwerty/transforms.py
```

### `lightning.py` — Dynamic Channel Count

`TDSConvCTCModule` now derives `num_electrodes` from `in_features` rather than a hardcoded
`ELECTRODE_CHANNELS = 16`. This allows the model to adapt automatically when a channel-selection
transform reduces the input to fewer than 16 channels per band.

### `transforms.py` — Three New Transforms

**`ChannelSelect(indices: list[int])`**
Keeps only the electrode channels at the specified indices. Operates on tensors of shape
`(T, bands, C, freq)` (after `LogSpectrogram`).
Example: `[0, 2, 4, 6, 8, 10, 12, 14]` → every-other-channel (8 ch/hand).

**`ChannelSubset(num_channels: int)`**
Keeps the first N channels from each band.
Example: `num_channels=8` → first 8 of 16 channels.

**`TemporalDownsample(factor: int)`**
Anti-aliased resampling applied to raw EMG before `LogSpectrogram`. Uses
`torchaudio.functional.resample` with polyphase anti-aliasing.
Example: `factor=4` → 2000 Hz → 500 Hz. Does **not** change `in_features`.

---

## Study 1: Channel Ablation

Investigates how CTC decoding degrades as fewer sEMG electrode channels are used per hand.
Every-Nth-channel selection preserves spatial coverage even with fewer electrodes.

### Results

| Channels/hand | Val CER | Test CER | Training time |
|---|---|---|---|
| 16 (baseline) | 18.52% | 22.28% | 3 h 51 m |
| 8 | 18.65% | 23.30% | 1 h 13 m |
| 4 | 24.88% | 27.12% | 1 h 10 m |
| 2 | 40.85% | 45.00% | 1 h 09 m |

Key finding: performance degrades gracefully from 16 → 8 channels (minimal CER increase)
before becoming more pronounced at 4 and 2 channels. Training is 3.3× faster with 8 channels.

### Running the Study

**Option A — notebook:** open `notebooks/experiments_all.ipynb`, section 3.

**Option B — shell script (full automated run):**
```bash
bash Playground_Ben/scripts/run_channel_ablation.sh
```

**Option C — manual, one condition at a time:**
```bash
# First patch the module and copy configs
cp Playground_Ben/emg2qwerty/*.py emg2qwerty/
cp Playground_Ben/config/transforms/channel_stride*.yaml config/transforms/

# Train each condition
python -m emg2qwerty.train user=single_user                            # 16 ch baseline
python -m emg2qwerty.train +transforms=channel_stride2 model.in_features=264 user=single_user  # 8 ch
python -m emg2qwerty.train +transforms=channel_stride4 model.in_features=132 user=single_user  # 4 ch
python -m emg2qwerty.train +transforms=channel_stride8 model.in_features=66  user=single_user  # 2 ch
```

### Configs

| Config | Channels/hand | Indices | `in_features` |
|---|---|---|---|
| `channel_stride2.yaml` | 8 | 0,2,4,6,8,10,12,14 | 264 |
| `channel_stride4.yaml` | 4 | 0,4,8,12 | 132 |
| `channel_stride8.yaml` | 2 | 0,8 | 66 |

---

## Study 2: Temporal Downsampling Ablation

Investigates how CTC performance degrades as the effective EMG sample rate is reduced
by integer factors. The `TemporalDownsample(factor)` transform applies anti-aliased
resampling before `LogSpectrogram`, so `in_features` is unchanged.

### Results

| Sample rate | Factor | Val CER | Test CER | Training time |
|---|---|---|---|---|
| 2000 Hz (baseline) | 1× | 18.52% | 22.28% | 3 h 51 m |
| 1000 Hz | 2× | 52.53% | 38.38% | 58 m |
| 500 Hz | 4× | 58.42% | 46.57% | 55 m |
| 250 Hz | 8× | 79.15% | 76.12% | 58 m |
| 125 Hz | 16× | 99.41% | 99.98% | 1 h 01 m |

Key finding: even 2× downsampling causes a large CER jump. EMG at 2000 Hz has ample
temporal resolution for keystroke decoding; 1000 Hz and below loses critical information.

### Running the Study

**Option A — notebook:** open `notebooks/experiments_all.ipynb`, section 4.

**Option B — shell script:**
```bash
USER_CFG=single_user bash Playground_Ben/scripts/run_temporal_ablation.sh
```

**Option C — manual:**
```bash
cp Playground_Ben/emg2qwerty/*.py emg2qwerty/
cp Playground_Ben/config/transforms/temporal_downsample_*.yaml config/transforms/

python -m emg2qwerty.train transforms=temporal_downsample_2  user=single_user   # 1000 Hz
python -m emg2qwerty.train transforms=temporal_downsample_4  user=single_user   # 500 Hz
python -m emg2qwerty.train transforms=temporal_downsample_8  user=single_user   # 250 Hz
python -m emg2qwerty.train transforms=temporal_downsample_16 user=single_user   # 125 Hz
```

---

## Study 3: Latent-Space and Reconstructed-Signal Models

Three parallel CNN variants trained on alternative input representations derived from the
project's autoencoder (AE). These replace the raw EMG spectrogram input with lower-dimensional
or pre-processed signals, enabling faster experiments and exploring what information survives
AE compression.

### Data Files

| File | Description | Shape | Rate |
|---|---|---|---|
| `data/emg_latent_ae_v2.hdf5` | AE latent vectors | 27,971 × 1024-dim | 62.5 Hz (32 ms/frame) |
| `data/*_recons_v3.hdf5` | AE-reconstructed EMG | (T, 16) per hand | 62.5 Hz (16 ms/frame) |

### 3a. Latent AE CNN (`train_latent_cnn.py`)

Trains on pre-computed AE latent vectors (1024-dim per frame). The spectrogram front-end
(`SpectrogramNorm + MultiBandMLP`) is replaced by a single linear projection:

```
Linear(1024 → num_features) → ReLU → TDSConvEncoder → Linear → LogSoftmax
```

This gives a 16× smaller input dimension than raw EMG, enabling rapid hyperparameter
exploration on the same model backbone.

**Checkpoint:** `checkpoints/best_latent_cnn.pt`
**Data loader:** `Playground_Kai.data_utils.get_latent_dataloaders`

```bash
python Playground_Ben/scripts/train_latent_cnn.py --epochs 150 --lr 3e-4

# With tuned hyperparameters:
python Playground_Ben/scripts/train_latent_cnn.py \
    --from-hyperparams Playground_Ben/checkpoints/best_hyperparams_cnn_latent.yaml \
    --epochs 150
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 150 | Training epochs |
| `--lr` | 3e-4 | Learning rate |
| `--weight-decay` | 1e-5 | AdamW weight decay |
| `--window-length` | 500 | Latent frames per window |
| `--stride` | 50 | Window stride in frames |
| `--num-features` | 768 | TDS operating dimension |
| `--block-channels` | 24 | TDSConv block channels |
| `--num-blocks` | 4 | Number of TDSConv blocks |
| `--kernel-width` | 32 | TDSConv temporal kernel width |
| `--from-hyperparams` | None | YAML from hyperparameter tuner |

### 3b. Reconstructed EMG CNN (`train_recons_cnn.py`)

Trains on AE-decoded (reconstructed) EMG signals stored as `*_recons_v3.hdf5`. Unlike the
latent CNN, this operates in the decoded signal space — closer to raw EMG but at 32× lower rate.

```
Flatten(2×16 → 32) → Linear(32 → num_features) → ReLU → TDSConvEncoder → Linear → LogSoftmax
```

**Checkpoint:** `checkpoints/best_recons_cnn.pt`
**Data loader:** `Playground_Ben.recons_data_utils.get_recons_dataloaders`

```bash
python Playground_Ben/scripts/train_recons_cnn.py --epochs 150 --stride 250
```

The data loader (`recons_data_utils.py`) mirrors the API of `Playground_Kai/data_utils.py`
and reads the reconstructed HDF5 schema where `emg_left` and `emg_right` are stored as
separate `(T, 16)` float32 datasets.

### 3c. Biophysics Pipeline CNN (`train_biophysics_cnn.py`)

Trains with the **full biophysics EMG preprocessing pipeline** from `Playground_Kai/data_preprocess.py`:

```
ToTensor → TemporalAlignmentJitter(120) → RandomBandRotation
→ ChannelSelector (8 even-indexed channels/wrist)
→ TemporalFilter (60 Hz notch + 4th-order Butterworth 20–450 Hz)
→ Decimator (2× → 1000 Hz)
→ MelSpectrogram (n_fft=256, win=64, hop=8, 32-bin Mel 20–450 Hz, log10)
→ SpecAugment (train only)
```

Model mirrors the standard `TDSConvCTCModule` architecture:
```
SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten → TDSConvEncoder → Linear → LogSoftmax
```

- `in_features` = 8 channels × 32 Mel bins = **256**
- `num_features` = 2 bands × `mlp_features` = **768** (with default `mlp_features=384`)
- Window: 8000 raw samples (4 s at 2 kHz)

**Checkpoint:** `checkpoints/best_biophysics_cnn.pt`

```bash
python Playground_Ben/scripts/train_biophysics_cnn.py --epochs 150 --lr 1e-3

# With tuned hyperparameters:
python Playground_Ben/scripts/train_biophysics_cnn.py \
    --mlp-features 512 --block-channels 32 --num-blocks 3 --kernel-width 24 \
    --lr 1.27e-3 --weight-decay 1.92e-3 --epochs 150
```

---

## Study 4: Hyperparameter Tuning

Two-phase Bayesian hyperparameter search for the latent and biophysics CNN models.

**Phase 1 (coarse):** 20 random trials × 10 epochs; identifies top-3 configs.
**Phase 2 (fine):** 10 trials × 20 epochs per top-3 config, narrowing the search space 3×.

### Latent CNN Tuner (`hyperparam_tuner_cnn.py`)

Search space:

| Parameter | Type | Range |
|---|---|---|
| `lr` | log-uniform | [1e-4, 1e-3] |
| `weight_decay` | log-uniform | [1e-5, 1e-2] |
| `num_features` | choice | [384, 576, 768] |
| `block_channels` | choice | [16, 24, 32, 48] |
| `kernel_width` | choice | [16, 24, 32, 48] |
| `num_blocks` | choice | [2, 3, 4] |

Best found: `lr=4.36e-4`, `weight_decay=1.19e-5`, `num_features=576`, `block_channels=24`,
`num_blocks=2`, `kernel_width=24`.

```bash
python Playground_Ben/scripts/hyperparam_tuner_cnn.py \
    --coarse-trials 20 --coarse-epochs 10 --fine-trials 10 --fine-epochs 20
# Output: Playground_Ben/checkpoints/best_hyperparams_cnn_latent.yaml
```

### Biophysics CNN Tuner (`hyperparam_tuner_raw_cnn.py`)

Similar structure but sweeps `mlp_features` instead of `num_features`.

Best found: `lr=1.27e-3`, `weight_decay=1.92e-3`, `mlp_features=512`, `block_channels=32`,
`num_blocks=3`, `kernel_width=24` → **Val CER 82.41%** at trial.

```bash
python Playground_Ben/scripts/hyperparam_tuner_raw_cnn.py \
    --coarse-trials 20 --coarse-epochs 8 --fine-trials 10 --fine-epochs 15
# Output: Playground_Ben/checkpoints/best_hyperparams_raw_cnn.yaml
```

---

## Analysis & Visualization Scripts

All scripts run from the repo root:

### `analyze_emg.py` — EMG Signal Analysis

Generates 8 diagnostic figures for a single HDF5 session: raw signal, PSD overlay, FFT heatmap,
within-hand cross-correlation matrix and lag heatmap, keystroke-triggered average, and
between-hand cross-correlation.

```bash
python Playground_Ben/scripts/analyze_emg.py \
    --hdf5 data/<session>.hdf5 \
    --out_dir Playground_Ben/plots/emg_analysis
```

Output: `plots/emg_analysis/1_signal.png` through `8_crosscorr_lags_between_hands.png`.

### `eval_plot.py` — Per-Window Inference Visualization

Loads a checkpoint, runs inference on N consecutive windows, and saves a three-panel figure
per window: raw sEMG, CTC emission heatmap, and ground-truth vs. predicted text with
character-level diff colouring.

```bash
python Playground_Ben/scripts/eval_plot.py \
    --checkpoint logs/<date>/<time>/checkpoints/<epoch>.ckpt \
    --hdf5 data/<session>.hdf5 \
    --n_examples 6 \
    --out_dir Playground_Ben/plots/eval

# Channel ablation checkpoint:
python Playground_Ben/scripts/eval_plot.py \
    --checkpoint <ablation_ckpt>.ckpt \
    --hdf5 data/<session>.hdf5 \
    --channel_indices 0 2 4 6 8 10 12 14 \
    --out_dir Playground_Ben/plots/eval
```

### `plot_results.py` — Training Curves and Comparison Plots

```bash
# Training loss + CER curves for a single run
python Playground_Ben/scripts/plot_results.py training_curves \
    --log_dir logs/2026-03-04/13-49-32

# Overlay val/CER from multiple runs
python Playground_Ben/scripts/plot_results.py overlay \
    --log_dirs logs/run_baseline logs/run_8ch \
    --labels "Baseline (16ch)" "8ch/hand"
```

---

## Notebooks

| Notebook | Contents |
|---|---|
| `notebooks/cnn_baseline.ipynb` | Baseline CNN only — train, plot curves, log results |
| `notebooks/experiments_all.ipynb` | All experiments: baseline, channel ablation, temporal ablation, latent CNN, reconstructed CNN, biophysics CNN, hyperparameter tuning, combined results |

---

## Results

### Channel Ablation (`results/channel_ablation_table.csv`)

```
channels_per_hand  val_cer_pct  test_cer_pct  training_time_sec
               2       40.85         45.00            4161
               4       24.88         27.12            4253
               8       18.65         23.30            4425
              16       18.52         22.28           13908
```

### Temporal Downsampling (`results/sampling_ablation_table.csv`)

```
sample_rate_hz  val_cer_pct  test_cer_pct  training_time_sec
          125        99.41        99.98            3691
          250        79.15        76.12            3529
          500        58.42        46.57            3332
         1000        52.53        38.38            3494
         2000        18.52        22.28           13908
```

Pre-generated plots are in `plots/`. Pre-trained checkpoints are in `checkpoints/`.
