# Playground_Ben

Ben Forbes — C247 Final Project Contributions

## Overview

This playground contains Ben's experimental work on the emg2qwerty codebase, focused on **channel ablation**: investigating how CTC decoding performance (character error rate) degrades as fewer sEMG electrode channels are used per hand.

## Directory Structure

```
Playground_Ben/
├── emg2qwerty/             # Modified source files (patches to the main module)
│   ├── lightning.py        # Dynamic electrode channel count for ablation support
│   └── transforms.py       # + ChannelSelect and ChannelSubset transforms
├── config/
│   └── transforms/         # Hydra config overrides for channel ablation runs
│       ├── channel_stride2.yaml   # 8 channels/hand  (every other electrode)
│       ├── channel_stride4.yaml   # 4 channels/hand  (every 4th electrode)
│       └── channel_stride8.yaml   # 2 channels/hand  (every 8th electrode)
├── scripts/
│   ├── analyze_emg.py      # EMG signal analysis and visualization
│   ├── eval_plot.py        # Per-window inference visualization with CTC heatmap
│   └── plot_results.py     # Training curves, architecture compare, ablation plots
└── plots/
    ├── eval/               # Example inference plots (raw EMG + CTC emissions + text diff)
    └── emg_analysis/       # EMG signal analysis plots (FFT, cross-correlation, etc.)
```

## Code Changes

Two files in the main `emg2qwerty/` module were modified. The versions here are Ben's; copy them over the originals to enable channel ablation support.

### `emg2qwerty/lightning.py`

`TDSConvCTCModule` was changed to derive the electrode channel count dynamically from `in_features` rather than using a hardcoded `ELECTRODE_CHANNELS = 16`. This lets the model adapt automatically when a channel-selection transform reduces the input to fewer than 16 channels.

**To apply:**
```bash
cp Playground_Ben/emg2qwerty/lightning.py emg2qwerty/lightning.py
```

### `emg2qwerty/transforms.py`

Two new dataclass transforms were added:

- **`ChannelSelect(indices)`** — keeps only the electrode channels at the specified indices (e.g. `[0, 2, 4, 6, 8, 10, 12, 14]` for every-other-channel).
- **`ChannelSubset(num_channels)`** — keeps the first `num_channels` electrodes from each band.

Both operate on tensors of shape `(T, bands, C, freq)` — i.e. after `LogSpectrogram`.

**To apply:**
```bash
cp Playground_Ben/emg2qwerty/transforms.py emg2qwerty/transforms.py
```

## Channel Ablation Configs

The three YAML files in `config/transforms/` are Hydra overrides that plug `ChannelSelect` into the standard transform pipeline. They are designed to be used with the `transforms` config group at training time.

| Config | Channels/hand | Indices | `in_features` |
|---|---|---|---|
| `channel_stride2.yaml` | 8 | 0,2,4,6,8,10,12,14 | 264 |
| `channel_stride4.yaml` | 4 | 0,4,8,12 | 132 |
| `channel_stride8.yaml` | 2 | 0,8 | 66 |

**To use:** copy them into the main `config/transforms/` directory, then pass `+transforms=channel_stride2` (or 4, 8) as a Hydra override at train time. You must also override `model.in_features` to match (requires the patched `lightning.py` to do this automatically from `in_features`).

```bash
# First apply both code patches (see above), then:
cp Playground_Ben/config/transforms/channel_stride*.yaml config/transforms/

# Example: train with every-other-channel (8 ch/hand)
python -m emg2qwerty.train \
    +transforms=channel_stride2 \
    model.in_features=264 \
    user=single_user
```

For the full 16-channel baseline, use the standard `config/transforms/` setup (no override needed).

## Scripts

All scripts are run from the **repo root** (`C247_mumbikaijonathanben/`) with the venv active (`source /home/benforbes/emg2qwerty/venv/bin/activate`).

### `analyze_emg.py` — EMG Signal Analysis

Generates 8 diagnostic figures for a single HDF5 session file: raw signal, PSD overlay, PSD heatmap, within-hand cross-correlation matrix and lag heatmap, keystroke-triggered average, and between-hand cross-correlation.

```bash
python Playground_Ben/scripts/analyze_emg.py \
    --hdf5 data/<session>.hdf5 \
    --out_dir Playground_Ben/plots/emg_analysis
```

`--start` (seconds from session start) is optional — defaults to the densest typing window.

### `eval_plot.py` — Per-Window Inference Visualization

Loads a checkpoint, runs inference on N consecutive windows from a session, and saves a three-panel figure per window: raw sEMG, CTC emission heatmap, and ground-truth vs. predicted text with character-level diff colouring.

Requires the patched `lightning.py` and `transforms.py` if using channel ablation checkpoints.

```bash
python Playground_Ben/scripts/eval_plot.py \
    --checkpoint logs/<date>/<time>/checkpoints/<epoch>.ckpt \
    --hdf5 data/<session>.hdf5 \
    --n_examples 6 \
    --out_dir Playground_Ben/plots/eval

# Channel ablation example (8 channels per hand):
python Playground_Ben/scripts/eval_plot.py \
    --checkpoint <ablation_ckpt>.ckpt \
    --hdf5 data/<session>.hdf5 \
    --channel_indices 0 2 4 6 8 10 12 14 \
    --out_dir Playground_Ben/plots/eval
```

### `plot_results.py` — Training Curves and Comparison Plots

Subcommand-based plotting utility. Reads TensorBoard event files for training curves; also supports architecture comparison bar charts and channel ablation line plots.

```bash
# Training loss + CER curves for a single run
python Playground_Ben/scripts/plot_results.py training_curves \
    --log_dir logs/2026-03-04/13-49-32

# Overlay val/CER from multiple runs
python Playground_Ben/scripts/plot_results.py overlay \
    --log_dirs logs/run_baseline logs/run_ablation8ch \
    --labels "Baseline (16ch)" "8ch/hand"

# Channel ablation summary (manual CER values)
python Playground_Ben/scripts/plot_results.py channel_ablation \
    --channels 2 4 8 16 \
    --val_cer  85 65 45 32 \
    --test_cer 88 68 48 34
```

---

## Study 2: Temporal Downsampling Ablation

Investigates how CTC decoding performance degrades as the EMG signal is temporally downsampled by integer factors of 2, 4, 8, and 16 (reducing the effective sample rate from 2000 Hz to 1000, 500, 250, and 125 Hz respectively).

Unlike the channel ablation, **no `in_features` change is needed** — the model input shape is unchanged; the time dimension just shrinks proportionally.

### New transform: `TemporalDownsample(factor)`

Added to `Playground_Ben/emg2qwerty/transforms.py`. Operates on the raw EMG tensor of shape `(T, ...)` before `LogSpectrogram`, keeping every `factor`-th sample:

```python
TemporalDownsample(factor=4)  # 2000 Hz → 500 Hz
```

It sits in the pipeline between `TemporalAlignmentJitter` and `LogSpectrogram` so that the jitter `max_offset` still refers to original 2000 Hz samples.

### Configs

| Config | Factor | Effective SR |
|---|---|---|
| `config/transforms/temporal_downsample_2.yaml` | 2 | 1000 Hz |
| `config/transforms/temporal_downsample_4.yaml` | 4 | 500 Hz |
| `config/transforms/temporal_downsample_8.yaml` | 8 | 250 Hz |
| `config/transforms/temporal_downsample_16.yaml` | 16 | 125 Hz |

### Running the full study

The `run_temporal_ablation.sh` script handles everything: patching the main module, copying configs, launching all 4 training jobs sequentially, and producing a summary ablation plot.

```bash
source /home/benforbes/emg2qwerty/venv/bin/activate
pip install -e .   # only needed once to point the venv at this repo
cd ~/C247_mumbikaijonathanben
bash Playground_Ben/scripts/run_temporal_ablation.sh
```

To use a different user config (e.g. multi-user):
```bash
USER_CFG=multi_user bash Playground_Ben/scripts/run_temporal_ablation.sh
```

Output plot is saved to `Playground_Ben/plots/temporal_ablation.png`.

To run a single factor manually after patching:
```bash
python -m emg2qwerty.train transforms=temporal_downsample_4 user=single_user
```

---

## Results

Pre-generated plots from Ben's runs are in `plots/`:

- `plots/eval/` — 6 inference examples from a trained checkpoint (raw EMG, CTC heatmap, text diff with CER).
- `plots/emg_analysis/` — 8 EMG analysis figures from a representative session window.
