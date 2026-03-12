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

from Playground_Kai.data_preprocess import (
    build_preprocess_transform,
    ChannelSelector,
    DEFAULT_CHANNELS,
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
    channel_half: bool = False,
):
    """Build the augmented transform pipeline used during training.

    Pipeline (channel_half=False): ToTensor → ForEach(RandomBandRotation)
        → TemporalAlignmentJitter → LogSpectrogram → SpecAugment
    Pipeline (channel_half=True):  same but ChannelSelector inserted after
        TemporalAlignmentJitter, before LogSpectrogram.

    Output shape: (T', 2, C, freq) where C=16 normally, C=8 with channel_half,
    and freq = n_fft // 2 + 1.
    """
    steps = [
        # Raw EMG structured array → (T, 2, 16) float tensor
        ToTensor(fields=["emg_left", "emg_right"]),
        # Independently rotate electrode channels for each band (i.i.d. offset)
        # Run on all 16 channels before any selection so rotation is well-defined.
        ForEach(RandomBandRotation(offsets=[-1, 0, 1])),
        # Randomly jitter temporal alignment between left and right wrist
        TemporalAlignmentJitter(max_offset=max_offset),
    ]
    if channel_half:
        # Keep 8 of 16 electrodes per band before the spectrogram step.
        steps.append(ChannelSelector(DEFAULT_CHANNELS))
    steps += [
        # Compute log-spectrogram: (T, 2, C) → (T', 2, C, freq)
        LogSpectrogram(n_fft=n_fft, hop_length=hop_length),
        # Time and frequency masking augmentation
        SpecAugment(
            n_time_masks=n_time_masks,
            time_mask_param=time_mask_param,
            n_freq_masks=n_freq_masks,
            freq_mask_param=freq_mask_param,
        ),
    ]
    return Compose(steps)


def _build_eval_transform(n_fft: int = 64, hop_length: int = 16, channel_half: bool = False):
    """Build the deterministic transform pipeline used for val/test.

    Pipeline (channel_half=False): ToTensor → LogSpectrogram
    Pipeline (channel_half=True):  ToTensor → ChannelSelector → LogSpectrogram
    """
    steps = [ToTensor(fields=["emg_left", "emg_right"])]
    if channel_half:
        steps.append(ChannelSelector(DEFAULT_CHANNELS))
    steps.append(LogSpectrogram(n_fft=n_fft, hop_length=hop_length))
    return Compose(steps)


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


def get_latent_session_paths(
    data_root: Path,
    config_path: Path,
) -> dict[str, list[Path]]:
    """Parse single_user.yaml and return the latent HDF5 paths for each split.

    Identical to :func:`get_session_paths` but builds paths with the
    ``_latent.hdf5`` suffix used by the ``data_latent/`` folder.

    Args:
        data_root: Directory containing the ``*_500hz.hdf5`` latent files.
        config_path: Path to ``config/user/single_user.yaml``.

    Returns:
        Dict with keys ``'train'``, ``'val'``, ``'test'`` mapping to lists of
        resolved :class:`pathlib.Path` objects pointing to latent HDF5 files.

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
            hdf5_path = data_root / f"{session}_latent.hdf5"
            if not hdf5_path.exists():
                raise FileNotFoundError(f"Latent HDF5 file not found: {hdf5_path}")
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
    test_window_length: Optional[int] = None,
    preprocess: bool = False,
    channel_half: bool = False,
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
    if preprocess:
        train_transform = build_preprocess_transform(augment=True)
        eval_transform  = build_preprocess_transform(augment=False)
    else:
        train_transform = _build_train_transform(channel_half=channel_half)
        eval_transform  = _build_eval_transform(channel_half=channel_half)

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

    # Test: whole session by default (RNN); windowed when test_window_length is set (Conformer)
    test_dataset = ConcatDataset([
        WindowedEMGDataset(
            hdf5_path=p,
            window_length=test_window_length,
            stride=None,
            padding=(0, 0) if test_window_length is None else padding,
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

# ===========================================================================
# Latent EMG dataset helpers
# (for emg_latent_ae_v2.hdf5 — pre-computed AE latent vectors, 1024-dim)
# ===========================================================================

import torch  # noqa: E402 — deferred alongside latent-only deps
import h5py   # noqa: E402 — deferred to avoid top-level dependency for non-latent callers
import json   # noqa: E402


class LatentEMGDataset(torch.utils.data.Dataset):
    """Windowed dataset over pre-computed latent EMG vectors.

    Reads ``emg2qwerty/latent`` (N_frames × 256 float32) and
    ``emg2qwerty/time`` (N_frames float64) from a latent HDF5 file,
    extracts keystroke labels from the ``keystrokes`` group attribute, and
    yields ``(latent_window, labels)`` tuples compatible with
    ``WindowedEMGDataset.collate``.

    Args:
        hdf5_path: Path to the latent HDF5 file (e.g. ``data_latent/*_latent.hdf5``).
        window_length: Number of latent frames per sample (default 125 ≈ 4 s
            at 32 ms/frame).
        stride: Latent-frame stride between consecutive windows. Defaults to
            ``window_length`` (no overlap).
        frame_start: First frame index to include (default 0). Use this to
            carve out a temporal split from a single file.
        frame_end: One-past-the-last frame index (default: full length).
            Enables ConcatDataset-based multi-file expansion in the future.
        jitter: If True, randomly jitter each window offset by up to one
            stride. Use for training.
    """

    def __init__(
        self,
        hdf5_path: Path,
        window_length: int = 125,
        stride: Optional[int] = None,
        frame_start: int = 0,
        frame_end: Optional[int] = None,
        jitter: bool = False,
    ) -> None:
        import numpy as np  # local import — already available at module level
        self.hdf5_path = hdf5_path
        self.window_length = window_length
        self.stride = stride if stride is not None else window_length
        self.jitter = jitter

        # Read metadata once at construction time (lightweight).
        with h5py.File(hdf5_path, "r") as f:
            grp = f["emg2qwerty"]
            total_frames = grp["latent"].shape[0]
            self.keystrokes: list = json.loads(grp.attrs.get("keystrokes", "[]"))

        self.frame_start = frame_start
        self.frame_end = frame_end if frame_end is not None else total_frames
        self._n_frames = self.frame_end - self.frame_start

        assert self.window_length > 0 and self.stride > 0, \
            "window_length and stride must be positive integers"
        assert self._n_frames >= self.window_length, (
            f"Split has only {self._n_frames} frames, need at least {self.window_length}"
        )

        # h5py file handle — opened lazily in __getitem__ so the dataset can
        # be pickled for multi-process DataLoader workers.
        self._file = None

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")

    def __len__(self) -> int:
        return int((self._n_frames - self.window_length) // self.stride + 1)

    def __getitem__(self, idx: int):
        import numpy as np
        from emg2qwerty.data import LabelData

        self._ensure_open()
        grp = self._file["emg2qwerty"]

        offset = self.frame_start + idx * self.stride

        # Random jitter: shift start up to one stride, staying in-bounds.
        if self.jitter:
            leftover = (self.frame_end - self.window_length) - offset
            if leftover > 0:
                offset += int(np.random.randint(0, min(self.stride, leftover)))

        # Latent frames: (window_length, 1024) → float32 tensor
        latent = grp["latent"][offset : offset + self.window_length]
        latent_tensor = torch.from_numpy(latent.astype("float32"))      # (T, 1024)

        # Timestamps for the first and last frame in this window
        start_t = float(grp["time"][offset])
        end_t   = float(grp["time"][offset + self.window_length - 1])

        # Labels from keystrokes JSON attribute
        label_data = LabelData.from_keystrokes(self.keystrokes, start_t, end_t)
        labels = torch.as_tensor(label_data.labels, dtype=torch.long)

        return latent_tensor, labels

    # Reuse the identical collate logic from WindowedEMGDataset.
    collate = staticmethod(WindowedEMGDataset.collate)


def get_latent_dataloaders_single(
    hdf5_path: Path,
    window_length: int = 125,
    stride: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build train / val / test DataLoaders for a **single** latent HDF5 file.

    Splits the session temporally: 70 % train → 15 % val → 15 % test.
    Used by :mod:`Playground_Kai.hyperparam_tuner_latent` for lightweight
    proxy trials on a single session.

    .. note::
        For full multi-session training use :func:`get_latent_dataloaders`
        which reads the train/val/test split from ``single_user.yaml``.

    Args:
        hdf5_path: Path to ``emg_latent_ae_v2.hdf5`` (or equivalent).
        window_length: Latent frames per sample (default 125 ≈ 4 s at 32 ms/frame).
        stride: Stride between windows; defaults to ``window_length``.
        batch_size: Batch size for train and val. Test always uses batch_size=1.
        num_workers: DataLoader workers (0 = main process, safest on Windows).

    Returns:
        Dict with keys ``'train'``, ``'val'``, ``'test'`` → :class:`DataLoader`.
    """
    # Determine total frame count and compute temporal split boundaries
    with h5py.File(hdf5_path, "r") as f:
        total_frames = f["emg2qwerty"]["latent"].shape[0]

    n_train = int(0.70 * total_frames)
    n_val   = int(0.15 * total_frames)
    # test gets the remainder so no frames are discarded
    n_test  = total_frames - n_train - n_val

    train_dataset = LatentEMGDataset(
        hdf5_path, window_length=window_length, stride=stride,
        frame_start=0,               frame_end=n_train,               jitter=True,
    )
    val_dataset = LatentEMGDataset(
        hdf5_path, window_length=window_length, stride=stride,
        frame_start=n_train,         frame_end=n_train + n_val,       jitter=False,
    )
    test_dataset = LatentEMGDataset(
        hdf5_path, window_length=window_length, stride=stride,
        frame_start=n_train + n_val, frame_end=total_frames,          jitter=False,
    )

    print(
        f"Latent split — total frames: {total_frames}"
        f" | train: {n_train} ({len(train_dataset)} windows)"
        f" | val: {n_val} ({len(val_dataset)} windows)"
        f" | test: {n_test} ({len(test_dataset)} windows)"
    )

    persistent = num_workers > 0
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=LatentEMGDataset.collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=LatentEMGDataset.collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=LatentEMGDataset.collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
    }


def get_latent_dataloaders(
    data_root: Path,
    config_path: Path,
    window_length: int = 125,
    stride: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build train / val / test DataLoaders from the single_user YAML split.

    Reads ``config/user/single_user.yaml`` to resolve train/val/test session
    lists, maps each session name to ``data_root/{session}_500hz.hdf5``, and
    builds a :class:`~torch.utils.data.ConcatDataset` of
    :class:`LatentEMGDataset` instances per split.

    Args:
        data_root: Directory containing the ``*_latent.hdf5`` latent files
            (typically ``data_latent/``).
        config_path: Path to ``config/user/single_user.yaml``.
        window_length: Latent frames per sample (default 125 ≈ 4 s at 32 ms/frame).
        stride: Stride between windows; defaults to ``window_length``.
        batch_size: Batch size for train and val. Test always uses batch_size=1.
        num_workers: DataLoader workers (0 = main process, safest on Windows).

    Returns:
        Dict with keys ``'train'``, ``'val'``, ``'test'`` → :class:`DataLoader`.
    """
    session_paths = get_latent_session_paths(data_root=data_root, config_path=config_path)

    train_dataset = ConcatDataset([
        LatentEMGDataset(p, window_length=window_length, stride=stride, jitter=True)
        for p in session_paths["train"]
    ])
    val_dataset = ConcatDataset([
        LatentEMGDataset(p, window_length=window_length, stride=stride, jitter=False)
        for p in session_paths["val"]
    ])
    test_dataset = ConcatDataset([
        LatentEMGDataset(p, window_length=window_length, stride=stride, jitter=False)
        for p in session_paths["test"]
    ])

    print(
        f"Latent multi-session split"
        f" | train: {len(session_paths['train'])} sessions ({len(train_dataset)} windows)"
        f" | val: {len(session_paths['val'])} sessions ({len(val_dataset)} windows)"
        f" | test: {len(session_paths['test'])} sessions ({len(test_dataset)} windows)"
    )

    persistent = num_workers > 0
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=LatentEMGDataset.collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=LatentEMGDataset.collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=LatentEMGDataset.collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
    }


def build_latent_loaders_from_paths(
    train_paths: list[Path],
    val_paths: list[Path],
    window_length: int = 125,
    stride: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build train and val DataLoaders from explicit latent session path lists.

    Lighter-weight alternative to :func:`get_latent_dataloaders` that accepts
    path lists directly instead of reading from a YAML split file.  Used by
    :mod:`Playground_Kai.hyperparam_tuner_latent` to run proxy training on
    subsets of sessions.

    Args:
        train_paths: Latent HDF5 paths to use for training.
        val_paths: Latent HDF5 paths to use for validation.
        window_length: Latent frames per sample (default 125 ≈ 4 s at 32 ms/frame).
        stride: Stride between windows; defaults to ``window_length``.
        batch_size: Batch size for both train and val loaders.
        num_workers: DataLoader worker processes.

    Returns:
        Dict with keys ``'train'`` and ``'val'`` mapping to DataLoaders.
    """
    train_dataset = ConcatDataset([
        LatentEMGDataset(p, window_length=window_length, stride=stride, jitter=True)
        for p in train_paths
    ])
    val_dataset = ConcatDataset([
        LatentEMGDataset(p, window_length=window_length, stride=stride, jitter=False)
        for p in val_paths
    ])

    persistent = num_workers > 0
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=LatentEMGDataset.collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=LatentEMGDataset.collate,
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
    preprocess: bool = False,
    channel_half: bool = False,
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
    if preprocess:
        train_transform = build_preprocess_transform(augment=True)
        eval_transform  = build_preprocess_transform(augment=False)
    else:
        train_transform = _build_train_transform(channel_half=channel_half)
        eval_transform  = _build_eval_transform(channel_half=channel_half)

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
