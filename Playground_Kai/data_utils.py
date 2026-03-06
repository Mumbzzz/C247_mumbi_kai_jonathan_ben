"""Data utilities for Playground_Kai.

Parses config/user/single_user.yaml and creates PyTorch DataLoaders
backed by emg2qwerty.data.WindowedEMGDataset.

Typical usage:
    from Playground_Kai.data_utils import get_dataloaders
    loaders = get_dataloaders(data_root=Path("data"), config_path=Path("config/user/single_user.yaml"))
    for batch in loaders["train"]:
        inputs = batch["inputs"]   # (T, N, 2, 16, freq)
        targets = batch["targets"] # (T_tgt, N)
        ...
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import yaml
from torch.utils.data import ConcatDataset, DataLoader

# Ensure workspace root is on sys.path so `emg2qwerty` is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from emg2qwerty.data import WindowedEMGDataset
from emg2qwerty.transforms import (
    Compose,
    ForEach,
    LogSpectrogram,
    RandomBandRotation,
    SpecAugment,
    TemporalAlignmentJitter,
    ToTensor,
)


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def _build_train_transform(
    n_fft: int = 64,
    hop_length: int = 16,
    max_offset: int = 120,
    n_time_masks: int = 3,
    time_mask_param: int = 25,
    n_freq_masks: int = 2,
    freq_mask_param: int = 4,
):
    """Build the augmented transform pipeline used during training.

    Pipeline: ToTensor → ForEach(RandomBandRotation) → TemporalAlignmentJitter
              → LogSpectrogram → SpecAugment

    Output shape: (T', 2, 16, freq) where T' = (T - max_offset) / hop_length
    and freq = n_fft // 2 + 1.
    """
    return Compose([
        # Raw EMG structured array → (T, 2, 16) float tensor
        ToTensor(fields=["emg_left", "emg_right"]),
        # Independently rotate electrode channels for each band (i.i.d. offset)
        ForEach(RandomBandRotation(offsets=[-1, 0, 1])),
        # Randomly jitter temporal alignment between left and right wrist
        TemporalAlignmentJitter(max_offset=max_offset),
        # Compute log-spectrogram: (T, 2, 16) → (T', 2, 16, freq)
        LogSpectrogram(n_fft=n_fft, hop_length=hop_length),
        # Time and frequency masking augmentation
        SpecAugment(
            n_time_masks=n_time_masks,
            time_mask_param=time_mask_param,
            n_freq_masks=n_freq_masks,
            freq_mask_param=freq_mask_param,
        ),
    ])


def _build_eval_transform(n_fft: int = 64, hop_length: int = 16):
    """Build the deterministic transform pipeline used for val/test.

    Pipeline: ToTensor → LogSpectrogram
    """
    return Compose([
        ToTensor(fields=["emg_left", "emg_right"]),
        LogSpectrogram(n_fft=n_fft, hop_length=hop_length),
    ])


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def get_session_paths(
    data_root: Path,
    config_path: Path,
) -> dict[str, list[Path]]:
    """Parse single_user.yaml and return the HDF5 paths for each data split.

    Args:
        data_root: Directory containing the ``*.hdf5`` session files.
        config_path: Path to ``config/user/single_user.yaml``.

    Returns:
        Dict with keys ``'train'``, ``'val'``, ``'test'`` mapping to lists of
        resolved :class:`pathlib.Path` objects pointing to HDF5 files.

    Raises:
        FileNotFoundError: If a session file referenced in the YAML is missing.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    result: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        entries = cfg["dataset"].get(split, [])
        paths: list[Path] = []
        for entry in entries:
            session = entry["session"]
            hdf5_path = data_root / f"{session}.hdf5"
            if not hdf5_path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
            paths.append(hdf5_path)
        result[split] = paths
    return result


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_root: Path,
    config_path: Path,
    window_length: int = 8000,
    stride: Optional[int] = None,
    padding: tuple[int, int] = (1800, 200),
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from the single_user split config.

    The train split uses windowed, jitter-augmented sessions with augmented
    transforms.  The val split uses the same windowing without jitter.
    The test split feeds each session as a single un-windowed sequence (no
    padding) to match realistic deployment conditions.

    Args:
        data_root: Directory containing ``*.hdf5`` session files.
        config_path: Path to ``config/user/single_user.yaml``.
        window_length: Raw EMG samples per training window (default 8000 = 4 s
            at 2 kHz). ``None`` disables windowing (whole session per sample).
        stride: Stride in raw EMG samples between consecutive windows.
            Defaults to ``window_length`` (no overlap).
        padding: ``(left, right)`` contextual EMG samples appended to each
            window before the spectrogram transform (default: 900 ms / 100 ms).
        batch_size: Batch size for train and val loaders.  Test always uses 1.
        num_workers: DataLoader worker processes.  Use ``0`` (the default) to
            keep everything in the main process — safest on Windows.

    Returns:
        Dict with keys ``'train'``, ``'val'``, ``'test'`` mapping to
        :class:`torch.utils.data.DataLoader` instances.
    """
    session_paths = get_session_paths(data_root=data_root, config_path=config_path)
    train_transform = _build_train_transform()
    eval_transform = _build_eval_transform()

    train_dataset = ConcatDataset([
        WindowedEMGDataset(
            hdf5_path=p,
            window_length=window_length,
            stride=stride,
            padding=padding,
            jitter=True,
            transform=train_transform,
        )
        for p in session_paths["train"]
    ])

    val_dataset = ConcatDataset([
        WindowedEMGDataset(
            hdf5_path=p,
            window_length=window_length,
            stride=None,   # no overlap for val
            padding=padding,
            jitter=False,
            transform=eval_transform,
        )
        for p in session_paths["val"]
    ])

    # Test: feed entire session at once — no windowing, no padding
    test_dataset = ConcatDataset([
        WindowedEMGDataset(
            hdf5_path=p,
            window_length=None,
            stride=None,
            padding=(0, 0),
            jitter=False,
            transform=eval_transform,
        )
        for p in session_paths["test"]
    ])

    persistent = num_workers > 0
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
    }


def build_loaders_from_paths(
    train_paths: list[Path],
    val_paths: list[Path],
    window_length: int = 8000,
    stride: Optional[int] = None,
    padding: tuple[int, int] = (1800, 200),
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build train and val DataLoaders from explicit session path lists.

    Lighter-weight alternative to ``get_dataloaders`` that accepts path lists
    directly instead of reading from a YAML split file.  Used by the
    hyperparameter tuner to run proxy training on subsets of sessions.

    Args:
        train_paths: HDF5 session paths to use for training.
        val_paths: HDF5 session paths to use for validation.
        window_length: Raw EMG samples per training window.
        stride: Stride between windows; defaults to ``window_length``.
        padding: ``(left, right)`` contextual EMG samples per window.
        batch_size: Batch size for both train and val loaders.
        num_workers: DataLoader worker processes.

    Returns:
        Dict with keys ``'train'`` and ``'val'`` mapping to DataLoaders.
    """
    train_transform = _build_train_transform()
    eval_transform = _build_eval_transform()

    train_dataset = ConcatDataset([
        WindowedEMGDataset(
            hdf5_path=p,
            window_length=window_length,
            stride=stride,
            padding=padding,
            jitter=True,
            transform=train_transform,
        )
        for p in train_paths
    ])

    val_dataset = ConcatDataset([
        WindowedEMGDataset(
            hdf5_path=p,
            window_length=window_length,
            stride=None,
            padding=padding,
            jitter=False,
            transform=eval_transform,
        )
        for p in val_paths
    ])

    persistent = num_workers > 0
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
    }
