# EMG Data Preprocessing Pipeline

This document describes the signal processing chain implemented in
`Playground_Kai/data_preprocess.py` and how to use it during training.

---

## Motivation

The raw EMG recordings are 32-channel signals sampled at 2000 Hz (16 electrodes
per wrist band). Feeding this directly into the model works, but the signal
contains:
- **Power-line noise** at 60 Hz and its harmonics
- **High-frequency noise** above 450 Hz that is outside the physiological EMG
  band and would alias when downsampled
- **Redundant channels**: the study captures all 16 electrodes, but every other
  electrode provides sufficient spatial coverage

The preprocessing pipeline removes these artefacts before the spectrogram is
computed, concentrates the frequency resolution where it matters (20–450 Hz),
and halves both the sample rate and the channel count — producing a compact
256-dimensional feature vector instead of 528.

---

## Processing Steps

### Step 0 — Channel selection

**Input**: `(T, 2, 16)` — raw 2000 Hz samples, all 16 electrodes per band  
**Output**: `(T, 2, 8)`

The HDF5 files contain 16 electrodes per wrist band (32 total). We retain
every other electrode: 0-based indices `[0, 2, 4, 6, 8, 10, 12, 14]`,
which correspond to physical channels **1, 3, 5, 7, 9, 11, 13, 15** on each
wrist (1-indexed). This gives 8 electrodes × 2 wrists = **16 channels** total.

### Step 1 — 60 Hz Notch filter

**Rate**: 2000 Hz  
**Filter**: IIR notch at 60 Hz, quality factor Q = 30 (≈ 2 Hz −3 dB bandwidth)  
**Implementation**: `scipy.signal.iirnotch` → converted to SOS, applied via
`sosfiltfilt` (zero-phase, no time delay)

The notch removes power-line interference while keeping the EMG signal on both
sides of 60 Hz intact.

### Step 2 — 20–450 Hz Bandpass filter

**Rate**: 2000 Hz  
**Filter**: 4th-order Butterworth bandpass [20 Hz, 450 Hz]  
**Implementation**: `scipy.signal.butter(N=4, Wn=[20, 450], btype='bandpass')` →
SOS, same zero-phase `sosfiltfilt` call as Step 1 (both filters concatenated
into one SOS chain)

The lower cutoff (20 Hz) removes slow drift and motion artefacts. The upper
cutoff (450 Hz) limits the signal to the physiological EMG band and prevents
aliasing in Step 3.

> Both Steps 1 and 2 are applied in a **single `sosfiltfilt` call** by
> concatenating the two SOS matrices, so there is no additional computation.

### Step 3 — 2× Decimation

**Rate**: 2000 Hz → 1000 Hz  
**Method**: keep every other sample (`x[::2]`)

No extra anti-aliasing filter is needed here because the bandpass in Step 2
already removes all energy above 450 Hz, which is below the 500 Hz Nyquist
frequency of the 1000 Hz output.

### Step 4 — Mel Spectrogram (STFT + Mel filterbank)

**Rate**: 1000 Hz  
**Parameters**:

| Parameter | Value | Notes |
|---|---|---|
| `n_fft` | 64 | FFT window = 64 samples = 64 ms |
| `hop_length` | 8 | Stride = 8 samples = 8 ms → **125 Hz frame rate** |
| `n_mels` | 32 | Number of Mel filterbank bins |
| `f_min` | 20 Hz | Lowest Mel bin edge |
| `f_max` | 450 Hz | Highest Mel bin edge |
| `center` | True | Half-window padding at boundaries |

A 4-second window (8000 raw samples) decimates to 4000 samples at 1000 Hz
and produces **≈ 500 spectrogram frames** — identical to the existing raw
pipeline (`n_fft=64, hop=16` on 2000 Hz), so all downstream shapes are preserved.

The Mel filterbank warps the linear STFT frequencies onto the Mel scale,
concentrating resolution at lower frequencies where EMG energy is denser.

### Step 5 — Log compression

**Formula**: `log10(x + 1e-6)`

Compresses the dynamic range of the power spectrogram so that faint
high-frequency components are not drowned out by strong low-frequency bursts.
The ε = 1e-6 floor prevents `log(0)`.

> **SpectrogramNorm (BatchNorm2d)** remains inside the model and is applied
> as the first layer of both `RNNEncoder` and `ConformerEncoder`. It handles
> per-channel temporal normalisation across the mini-batch during training.

---

## Output Shape

| Stage | Shape | Notes |
|---|---|---|
| Raw HDF5 window | `(8000, 2, 16)` | 4 s @ 2000 Hz |
| After channel selection | `(8000, 2, 8)` | 8 electrodes/band |
| After filter + decimate | `(4000, 2, 8)` | 4 s @ 1000 Hz |
| After Mel spectrogram + log | `(500, 2, 8, 32)` | `(T, bands, channels, mels)` |
| Flattened per-band for MLP | `(500, 2, 256)` | `in_features = 8 × 32 = 256` |

Compare with the raw pipeline: `(500, 2, 16, 33)` → `in_features = 16 × 33 = 528`.

---

## Usage

### Train with preprocessing

```bash
python -m Playground_Kai.train --model rnn --preprocess
python -m Playground_Kai.train --model conformer --preprocess
```

### Train without preprocessing (original raw pipeline)

```bash
python -m Playground_Kai.train --model rnn
python -m Playground_Kai.train --model conformer
```

### Test-only (must pass `--preprocess` if checkpoint was trained with it)

```bash
python -m Playground_Kai.train --model rnn --preprocess --test-only
python -m Playground_Kai.train --model rnn --test-only
```

### Checkpoint naming

Checkpoints are named with the input feature count so the two pipelines never
overwrite each other:

| Pipeline | Checkpoint |
|---|---|
| `--preprocess` | `checkpoints/best_rnn_256.pt` |
| raw (default) | `checkpoints/best_rnn_528.pt` |

---

## Code Location

| File | Role |
|---|---|
| `Playground_Kai/data_preprocess.py` | All filter design, transform classes, `build_preprocess_transform()` |
| `Playground_Kai/data_utils.py` | `preprocess=True/False` selects the transform chain before building DataLoaders |
| `Playground_Kai/model.py` | `electrode_channels` parameter scales `SpectrogramNorm` to 8 or 16 channels |
| `Playground_Kai/train.py` | `--preprocess` flag wires everything together |

---

## Exported Constants

From `Playground_Kai/data_preprocess.py`:

```python
from Playground_Kai.data_preprocess import (
    IN_FEATURES,          # 256  (8 channels × 32 Mel bins)
    N_ELECTRODE_CHANNELS, # 8    (per wrist band)
    SAMPLE_RATE,          # 1000 (Hz, after decimation)
)
```
