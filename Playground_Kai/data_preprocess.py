"""EMG data preprocessing pipeline for Playground_Kai.

Processing chain (raw 2000 Hz → model-ready Mel spectrogram):

    1. Channel selection  — keep every other electrode: indices [0,2,...,14] per
                            band, yielding 8 electrodes × 2 wrists = 16 total.
    2. Notch filter       — zero-phase 60 Hz notch (Q=30, ≈ 2 Hz passband).
    3. Bandpass filter    — 4th-order Butterworth 20–450 Hz to retain physiological
                            content and prevent aliasing before decimation.
    4. Decimation         — keep every other sample (2000 Hz → 1000 Hz).
                            Safe: bandpass already removes all content > 450 Hz,
                            which is below the 500 Hz Nyquist of 1000 Hz.
    5. Mel spectrogram    — STFT (n_fft=64, hop=8) followed by 32-bin Mel filterbank
                            concentrated between 20 Hz and 450 Hz.
    6. Log compression    — log10(x + ε) to compress dynamic range.
                            (SpectrogramNorm / BatchNorm2d stays inside the model.)

Public API
----------
build_preprocess_transform(augment=False) -> Compose
    Returns the full transform chain ready to substitute the existing
    LogSpectrogram-based pipeline in data_utils.

Constants exported for model construction
------------------------------------------
IN_FEATURES          int = 256   (N_ELECTRODE_CHANNELS * N_MELS = 8 * 32)
N_ELECTRODE_CHANNELS int = 8     (electrodes kept per wrist band)
SAMPLE_RATE          int = 1000  (Hz, after decimation)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import torchaudio.transforms as _TA
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torchaudio is required for MelSpectrogramTransform. "
        "Install it with: pip install torchaudio"
    ) from exc

from emg2qwerty.transforms import (
    Compose,
    ForEach,
    RandomBandRotation,
    SpecAugment,
    TemporalAlignmentJitter,
    ToTensor,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORIG_SAMPLE_RATE: int = 2000           # raw recording rate (Hz)
DECIMATION_FACTOR: int = 2             # decimate 2000 Hz → 1000 Hz
SAMPLE_RATE: int = ORIG_SAMPLE_RATE // DECIMATION_FACTOR  # 1000 Hz

# Electrode channel selection.
# The HDF5 files contain 16 electrodes per wrist band.
# "Channels 1, 3, 5, …, 31" (1-based) map to 0-based even indices within each band.
DEFAULT_CHANNELS: list[int] = list(range(0, 16, 2))  # [0, 2, 4, 6, 8, 10, 12, 14]
N_ELECTRODE_CHANNELS: int = len(DEFAULT_CHANNELS)    # 8 per band

# Mel spectrogram parameters
N_FFT: int = 64
HOP_LENGTH: int = 8        # 8 ms at 1000 Hz → 125 Hz frame rate
N_MELS: int = 32
MEL_FMIN: float = 20.0
MEL_FMAX: float = 450.0

# Exported for model construction
IN_FEATURES: int = N_ELECTRODE_CHANNELS * N_MELS  # 8 * 32 = 256


# ---------------------------------------------------------------------------
# Pre-built digital filters (computed once at import time)
# ---------------------------------------------------------------------------

# 60 Hz notch filter — quality factor Q=30 gives ≈ 2 Hz −3 dB bandwidth
_b_notch, _a_notch = iirnotch(w0=60.0, Q=30.0, fs=ORIG_SAMPLE_RATE)
_notch_sos: np.ndarray = tf2sos(_b_notch, _a_notch)  # shape (1, 6)

# 4th-order Butterworth bandpass: 20–450 Hz at 2000 Hz
_bandpass_sos: np.ndarray = butter(
    N=4,
    Wn=[20.0, 450.0],
    btype="bandpass",
    fs=ORIG_SAMPLE_RATE,
    output="sos",
)  # shape (4, 6)

# Concatenate into a single SOS chain — one sosfiltfilt call does both filters
_combined_sos: np.ndarray = np.concatenate([_notch_sos, _bandpass_sos], axis=0)  # (5, 6)


# ---------------------------------------------------------------------------
# Transform classes
# ---------------------------------------------------------------------------

class ChannelSelector:
    """Select a fixed subset of electrode channels along the last dimension.

    Applied after ``ToTensor``, which produces ``(T, num_bands=2, C=16)`` tensors.

    Args:
        channels: 0-based column indices to retain, e.g. ``[0, 2, 4, …, 14]``.

    Input/output shape: ``(T, num_bands, C_in) → (T, num_bands, len(channels))``.
    """

    def __init__(self, channels: list[int] = DEFAULT_CHANNELS) -> None:
        self._indices = channels

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self._indices]


class TemporalFilter:
    """Zero-phase 60 Hz notch + 20–450 Hz bandpass filter applied along axis 0.

    Uses ``scipy.signal.sosfiltfilt`` (forward-backward pass) so the filter
    introduces no phase delay.  Both filters are chained in a single call via
    a concatenated SOS matrix.

    Input/output shape: ``(T, num_bands, C)`` — shape is unchanged.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        arr: np.ndarray = x.numpy()  # (T, 2, C)
        arr = sosfiltfilt(_combined_sos, arr, axis=0).astype(np.float32)
        return torch.from_numpy(arr.copy())


class Decimator:
    """2× decimate by retaining every alternate time sample.

    No explicit anti-aliasing filter is applied here because
    ``TemporalFilter`` already limits the signal to 450 Hz, well below
    the 500 Hz Nyquist frequency of the 1000 Hz output rate.

    Input shape:  ``(T,        num_bands, C)``
    Output shape: ``(T // 2,   num_bands, C)``
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x[::DECIMATION_FACTOR]


class MelSpectrogramTransform:
    """Compute a log Mel spectrogram from 1000 Hz raw EMG data.

    Processing steps:
        1. Reshape ``(T, 2, C)`` → ``(2*C, T)`` for batched processing.
        2. Apply ``torchaudio.transforms.MelSpectrogram`` (power=2 spectrogram
           passed through 32-bin Mel filterbank).
           Output: ``(2*C, N_MELS, T_spec)``.
        3. Apply ``log10(x + 1e-6)`` for dynamic range compression.
        4. Reshape to ``(T_spec, 2, C, N_MELS)`` — the ``(T, bands, channels,
           freq)`` convention consumed by ``SpectrogramNorm`` and both models.

    ``T_spec = ceil(T / HOP_LENGTH)`` when ``center=True``.
    For a 4-second window (4000 samples at 1000 Hz), ``T_spec ≈ 500``,
    matching the frame count produced by the raw ``LogSpectrogram`` pipeline.
    """

    def __init__(self) -> None:
        self._mel = _TA.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=MEL_FMIN,
            f_max=MEL_FMAX,
            center=True,
            power=2.0,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        T, num_bands, C = x.shape
        # (T, 2, C) → (2C, T) — MelSpectrogram expects (channel, time)
        x_flat = x.permute(1, 2, 0).reshape(num_bands * C, T)
        mel = self._mel(x_flat)                        # (2C, N_MELS, T_spec)
        mel = torch.log10(mel + 1e-6)                  # log compression
        T_spec = mel.shape[-1]
        # (2C, N_MELS, T_spec) → (T_spec, 2, C, N_MELS)
        mel = mel.view(num_bands, C, N_MELS, T_spec)   # (2, C, N_MELS, T_spec)
        mel = mel.permute(3, 0, 1, 2)                  # (T_spec, 2, C, N_MELS)
        return mel.contiguous()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_preprocess_transform(
    channels: list[int] = DEFAULT_CHANNELS,
    augment: bool = False,
) -> Compose:
    """Return the full EMG preprocessing transform chain.

    Args:
        channels: 0-based electrode indices to keep per wrist band.
            Defaults to ``DEFAULT_CHANNELS`` (8 even-indexed electrodes).
        augment: If ``True``, insert training-time augmentations
            (temporal jitter, per-band electrode rotation, SpecAugment).
            If ``False``, return the deterministic eval-time chain.

    Returns:
        A :class:`~emg2qwerty.transforms.Compose` transform suitable for
        passing as the ``transform`` argument to ``WindowedEMGDataset``.

    Pipeline (augment=False — eval/test):
        ToTensor → ChannelSelector → TemporalFilter → Decimator
        → MelSpectrogramTransform

    Pipeline (augment=True — train):
        ToTensor → TemporalAlignmentJitter(120) → ChannelSelector
        → ForEach(RandomBandRotation) → TemporalFilter → Decimator
        → MelSpectrogramTransform → SpecAugment
    """
    mel_transform = MelSpectrogramTransform()

    if augment:
        return Compose([
            ToTensor(fields=["emg_left", "emg_right"]),
            # Temporal jitter before channel selection so all 16 raw channels
            # are aligned — the jitter only touches the time axis.
            TemporalAlignmentJitter(max_offset=120),
            ChannelSelector(channels),
            # Per-band electrode rotation for rotational augmentation
            ForEach(RandomBandRotation(offsets=[-1, 0, 1])),
            TemporalFilter(),
            Decimator(),
            mel_transform,
            SpecAugment(
                n_time_masks=3,
                time_mask_param=25,
                n_freq_masks=2,
                freq_mask_param=4,
            ),
        ])
    else:
        return Compose([
            ToTensor(fields=["emg_left", "emg_right"]),
            ChannelSelector(channels),
            TemporalFilter(),
            Decimator(),
            mel_transform,
        ])
