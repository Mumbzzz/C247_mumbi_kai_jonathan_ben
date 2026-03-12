"""Data utilities for Playground_Mumbi.

Parses config/user/single_user.yaml and creates PyTorch DataLoaders
backed by emg2qwerty.data.WindowedEMGDataset.

Supports a train_fraction parameter to use only the first X% of
training windows chronologically, enabling data-fraction ablation
studies without changing the val/test splits.

Also provides latent-space data loading via LatentEMGDataset and
get_latent_dataloaders for use with data/preprocessed/*.hdf5 files.

Typical usage (raw EMG):
    from Playground_Mumbi.data_utils import get_dataloaders
    loaders = get_dataloaders(
        data_root=Path("data"),
        config_path=Path("config/user/single_user.yaml"),
        train_fraction=0.5,
    )
    for batch in loaders["train"]:
        inputs = batch["inputs"]   # (T, N, 2, 16, freq)
        targets = batch["targets"] # (T_tgt, N)
        ...

Typical usage (latent):
    from Playground_Mumbi.data_utils import get_latent_dataloaders
    loaders = get_latent_dataloaders(Path("data/preprocessed/emg_latent_ae_v2.hdf5"))
    for batch in loaders["train"]:
        inputs = batch["inputs"]   # (T, N, 1024)
        targets = batch["targets"] # (T_tgt, N)
        ...
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset

# Ensure workspace root is on sys.path so `emg2qwerty` is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from emg2qwerty.data import WindowedEMGDataset
from emg2qwerty.transforms import (
    ChannelSelect,
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
    channel_indices: list[int] | None = None,
):
    """Build the augmented transform pipeline used during training.

    Pipeline: ToTensor → ForEach(RandomBandRotation) → TemporalAlignmentJitter
              → LogSpectrogram → [ChannelSelect] → SpecAugment

    Output shape: (T', 2, C, freq) where C = len(channel_indices) or 16,
    T' = (T - max_offset) / hop_length, freq = n_fft // 2 + 1.
    """
    steps = [
        # Raw EMG structured array → (T, 2, 16) float tensor
        ToTensor(fields=["emg_left", "emg_right"]),
        # Independently rotate electrode channels for each band (i.i.d. offset)
        ForEach(RandomBandRotation(offsets=[-1, 0, 1])),
        # Randomly jitter temporal alignment between left and right wrist
        TemporalAlignmentJitter(max_offset=max_offset),
        # Compute log-spectrogram: (T, 2, 16) → (T', 2, 16, freq)
        LogSpectrogram(n_fft=n_fft, hop_length=hop_length),
    ]
    if channel_indices is not None:
        steps.append(ChannelSelect(indices=channel_indices))
    steps.append(
        # Time and frequency masking augmentation
        SpecAugment(
            n_time_masks=n_time_masks,
            time_mask_param=time_mask_param,
            n_freq_masks=n_freq_masks,
            freq_mask_param=freq_mask_param,
        )
    )
    return Compose(steps)


def _build_eval_transform(
    n_fft: int = 64,
    hop_length: int = 16,
    channel_indices: list[int] | None = None,
):
    """Build the deterministic transform pipeline used for val/test.

    Pipeline: ToTensor → LogSpectrogram → [ChannelSelect]
    """
    steps = [
        ToTensor(fields=["emg_left", "emg_right"]),
        LogSpectrogram(n_fft=n_fft, hop_length=hop_length),
    ]
    if channel_indices is not None:
        steps.append(ChannelSelect(indices=channel_indices))
    return Compose(steps)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def get_session_paths(
    data_root: Path,
    config_path: Path,
    data_subdir: str = "89335547",
    file_suffix: str = "",
) -> dict[str, list[Path]]:
    """Parse single_user.yaml and return the HDF5 paths for each data split.

    Args:
        data_root:   Directory containing the user data subdirectory.
        config_path: Path to ``config/user/single_user.yaml``.
        data_subdir: Subdirectory under data_root containing session files
                     (default: ``"89335547"``).
        file_suffix: Extra suffix appended to session name before ``.hdf5``
                     (e.g. ``"_recons_v3"`` → ``{session}_recons_v3.hdf5``).

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
            hdf5_path = data_root / data_subdir / f"{session}{file_suffix}.hdf5"
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
    test_window_length: Optional[int] = None,
    train_fraction: float = 1.0,
    channel_indices: list[int] | None = None,
    data_subdir: str = "89335547",
    file_suffix: str = "",
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from the single_user split config.

    The train split uses windowed, jitter-augmented sessions with augmented
    transforms.  The val split uses the same windowing without jitter.
    The test split feeds each session as a single un-windowed sequence (no
    padding) to match realistic deployment conditions.

    The ``train_fraction`` parameter selects the first X% of training windows
    chronologically via :class:`torch.utils.data.Subset`, enabling data-fraction
    ablation studies.  Val and test splits are always kept at 100%.

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
        test_window_length: Window length for the test split.  When ``None``
            (default) each test session is fed as a single un-windowed sequence.
        train_fraction: Fraction of training windows to use, in (0.0, 1.0].
            Windows are selected from the start of the dataset chronologically.
            For example, ``0.5`` uses the first 50% of windows.

    Returns:
        Dict with keys ``'train'``, ``'val'``, ``'test'`` mapping to
        :class:`torch.utils.data.DataLoader` instances.
    """
    assert 0.0 < train_fraction <= 1.0, (
        f"train_fraction must be in (0.0, 1.0], got {train_fraction}"
    )

    session_paths = get_session_paths(data_root=data_root, config_path=config_path,
                                       data_subdir=data_subdir, file_suffix=file_suffix)
    train_transform = _build_train_transform(channel_indices=channel_indices)
    eval_transform = _build_eval_transform(channel_indices=channel_indices)

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

    # Optionally restrict to the first train_fraction of windows
    n_total = len(train_dataset)
    if train_fraction < 1.0:
        n_subset = max(1, int(train_fraction * n_total))
        print(
            f"  train_fraction={train_fraction:.2f}: "
            f"using {n_subset} / {n_total} training windows"
        )
        train_dataset = Subset(train_dataset, list(range(n_subset)))
    else:
        print(f"  train_fraction=1.00: using all {n_total} training windows")

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

    # Test: whole session by default; windowed when test_window_length is set
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


def get_dataloaders_biophys(
    data_root: Path,
    config_path: Path,
    window_length: int = 8000,
    stride: Optional[int] = None,
    padding: tuple[int, int] = (1800, 200),
    batch_size: int = 32,
    num_workers: int = 0,
    test_window_length: Optional[int] = None,
    train_fraction: float = 1.0,
    data_subdir: str = "89335547",
    file_suffix: str = "",
) -> dict[str, DataLoader]:
    """Like get_dataloaders but uses the biophysics preprocessing pipeline.

    Swaps in Kai's ``build_preprocess_transform`` (notch + bandpass + decimation
    + Mel spectrogram) in place of the standard LogSpectrogram chain.
    Channel selection (8 per band) and all augmentations are handled internally
    by the transform — no ``channel_indices`` parameter is needed.
    """
    from Playground_Kai.data_preprocess import build_preprocess_transform

    assert 0.0 < train_fraction <= 1.0

    session_paths = get_session_paths(data_root=data_root, config_path=config_path,
                                       data_subdir=data_subdir, file_suffix=file_suffix)
    train_transform = build_preprocess_transform(augment=True)
    eval_transform  = build_preprocess_transform(augment=False)

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

    n_total = len(train_dataset)
    if train_fraction < 1.0:
        n_subset = max(1, int(train_fraction * n_total))
        print(f"  train_fraction={train_fraction:.2f}: using {n_subset} / {n_total} training windows")
        train_dataset = Subset(train_dataset, list(range(n_subset)))
    else:
        print(f"  train_fraction=1.00: using all {n_total} training windows")

    val_dataset = ConcatDataset([
        WindowedEMGDataset(
            hdf5_path=p,
            window_length=window_length,
            stride=None,
            padding=padding,
            jitter=False,
            transform=eval_transform,
        )
        for p in session_paths["val"]
    ])

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
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=WindowedEMGDataset.collate,
            pin_memory=True, persistent_workers=persistent,
        ),
        "val": DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=WindowedEMGDataset.collate,
            pin_memory=True, persistent_workers=persistent,
        ),
        "test": DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=num_workers, collate_fn=WindowedEMGDataset.collate,
            pin_memory=True, persistent_workers=persistent,
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
    channel_indices: list[int] | None = None,
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
    train_transform = _build_train_transform(channel_indices=channel_indices)
    eval_transform = _build_eval_transform(channel_indices=channel_indices)

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


# ===========================================================================
# Latent EMG dataset helpers
# (for data/preprocessed/*.hdf5 — pre-computed AE latent vectors, 1024-dim)
# ===========================================================================

import torch   # noqa: E402 — deferred alongside latent-only deps
import h5py    # noqa: E402
import json    # noqa: E402


class LatentEMGDataset(torch.utils.data.Dataset):
    """Windowed dataset over pre-computed latent EMG vectors.

    Reads ``emg2qwerty/latent`` (N_frames × 1024 float32) and
    ``emg2qwerty/time`` (N_frames float64) from a latent HDF5 file,
    extracts keystroke labels from the ``keystrokes`` group attribute, and
    yields ``(latent_window, labels)`` tuples compatible with
    ``WindowedEMGDataset.collate``.

    Args:
        hdf5_path:     Path to the latent HDF5 file.
        window_length: Number of latent frames per sample (default 125 ≈ 4 s
                       at 32 ms/frame).
        stride:        Latent-frame stride between consecutive windows.
                       Defaults to ``window_length`` (no overlap).
        frame_start:   First frame index to include (default 0).
        frame_end:     One-past-the-last frame index (default: full length).
        jitter:        If True, randomly jitter each window offset by up to
                       one stride. Use for training only.
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
        self.hdf5_path = hdf5_path
        self.window_length = window_length
        self.stride = stride if stride is not None else window_length
        self.jitter = jitter

        with h5py.File(hdf5_path, "r") as f:
            grp = f["emg2qwerty"]
            total_frames = grp["latent"].shape[0]
            self.keystrokes: list = json.loads(grp.attrs.get("keystrokes", "[]"))

        self.frame_start = frame_start
        self.frame_end = frame_end if frame_end is not None else total_frames
        self._n_frames = self.frame_end - self.frame_start

        assert self.window_length > 0 and self.stride > 0
        assert self._n_frames >= self.window_length, (
            f"Split has only {self._n_frames} frames, need at least {self.window_length}"
        )

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

        if self.jitter:
            leftover = (self.frame_end - self.window_length) - offset
            if leftover > 0:
                offset += int(np.random.randint(0, min(self.stride, leftover)))

        latent = grp["latent"][offset : offset + self.window_length]
        latent_tensor = torch.from_numpy(latent.astype("float32"))   # (T, 1024)

        start_t = float(grp["time"][offset])
        end_t   = float(grp["time"][offset + self.window_length - 1])

        label_data = LabelData.from_keystrokes(self.keystrokes, start_t, end_t)
        labels = torch.as_tensor(label_data.labels, dtype=torch.long)

        return latent_tensor, labels

    collate = staticmethod(WindowedEMGDataset.collate)


def get_latent_dataloaders(
    data_dir: Path,
    config_path: Path,
    window_length: int = 125,
    stride: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    train_fraction: float = 1.0,
) -> dict[str, DataLoader]:
    """Build train / val / test DataLoaders from a latent session split YAML.

    Mirrors the raw ``get_dataloaders`` pattern: reads a YAML file listing
    session names per split, resolves each to
    ``<data_dir>/<user>/<session>_latent_v2.hdf5``, and concatenates all
    sessions in each split into a single dataset.

    Args:
        data_dir:      Directory containing latent HDF5 files
                       (e.g. ``data/preprocessed``).
        config_path:   Path to the split YAML (e.g.
                       ``config/user/latent_split_placeholder.yaml``).
        window_length: Latent frames per sample (default 125 ≈ 4 s at 32 ms/frame).
        stride:        Stride between windows; defaults to ``window_length``.
        batch_size:    Batch size for train and val. Test always uses batch_size=1.
        num_workers:   DataLoader workers (0 = main process).
        train_fraction: Fraction of training windows to use (0.0, 1.0].

    Returns:
        Dict with keys ``'train'``, ``'val'``, ``'test'`` → DataLoader.
    """
    assert 0.0 < train_fraction <= 1.0

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    def _resolve_paths(split: str) -> list[Path]:
        paths = []
        for entry in cfg["dataset"].get(split, []):
            p = data_dir / f"{entry['session']}_latent_v2.hdf5"  # same session names as raw, latent suffix appended
            if not p.exists():
                raise FileNotFoundError(f"Latent HDF5 not found: {p}")
            paths.append(p)
        return paths

    train_paths = _resolve_paths("train")
    val_paths   = _resolve_paths("val")
    test_paths  = _resolve_paths("test")

    train_dataset: torch.utils.data.Dataset = ConcatDataset([
        LatentEMGDataset(p, window_length=window_length, stride=stride, jitter=True)
        for p in train_paths
    ])

    n_total = len(train_dataset)
    if train_fraction < 1.0:
        n_subset = max(1, int(train_fraction * n_total))
        print(f"  train_fraction={train_fraction:.2f}: using {n_subset} / {n_total} training windows")
        train_dataset = Subset(train_dataset, list(range(n_subset)))
    else:
        print(f"  train_fraction=1.00: using all {n_total} training windows")

    val_dataset = ConcatDataset([
        LatentEMGDataset(p, window_length=window_length, stride=stride, jitter=False)
        for p in val_paths
    ])
    test_dataset = ConcatDataset([
        LatentEMGDataset(p, window_length=window_length, stride=stride, jitter=False)
        for p in test_paths
    ])

    print(
        f"Latent split — train sessions: {len(train_paths)}"
        f" | val sessions: {len(val_paths)}"
        f" | test sessions: {len(test_paths)}"
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


# ---------------------------------------------------------------------------
# Reconstructed EMG v3 dataset + DataLoader factory
# ---------------------------------------------------------------------------

class ReconstructedEMGDataset(torch.utils.data.Dataset):
    """Windowed dataset over AE-reconstructed EMG (_recons_v3.hdf5) files.

    Preloads all sessions into RAM at init for fast random access.
    Each item is a ``(window_length, 32)`` float32 tensor (left+right stacked)
    plus a label tensor, compatible with ``WindowedEMGDataset.collate``.

    Args:
        paths:         List of HDF5 file paths to load.
        window_length: Frames per window (default 250 ≈ 4 s at 62.5 Hz).
        stride:        Frame stride between consecutive windows.
        jitter:        If True, randomly shift window offset by up to one stride.
    """

    def __init__(
        self,
        paths: list[Path],
        window_length: int = 250,
        stride: int = 250,
        jitter: bool = False,
    ) -> None:
        import numpy as np
        from emg2qwerty.data import LabelData as _LabelData

        self.window_length = window_length
        self.stride        = stride
        self.jitter        = jitter

        self._sessions: list[tuple[torch.Tensor, list, list]] = []
        self._windows:  list[tuple[int, int]] = []  # (session_idx, frame_offset)

        for p in paths:
            with h5py.File(p, "r") as f:
                ts    = f["emg2qwerty"]["timeseries"]
                left  = ts["emg_left"][:].astype("float32")   # (T, 16)
                right = ts["emg_right"][:].astype("float32")  # (T, 16)
                times = ts["time"][:].tolist()
                ks    = json.loads(f["emg2qwerty"].attrs.get("keystrokes", "[]"))

            data = torch.from_numpy(np.concatenate([left, right], axis=1))  # (T, 32)
            sess_idx = len(self._sessions)
            self._sessions.append((data, times, ks))

            n_frames = data.shape[0]
            for offset in range(0, n_frames - window_length + 1, stride):
                self._windows.append((sess_idx, offset))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        from emg2qwerty.data import LabelData

        sess_idx, offset = self._windows[idx]
        data, times, ks = self._sessions[sess_idx]

        if self.jitter:
            max_jitter = min(self.stride, data.shape[0] - self.window_length - offset)
            if max_jitter > 0:
                offset += int(torch.randint(0, max_jitter, (1,)).item())

        window  = data[offset : offset + self.window_length]   # (T, 32)
        start_t = times[offset]
        end_t   = times[offset + self.window_length - 1]

        label_data = LabelData.from_keystrokes(ks, start_t, end_t)
        labels = torch.as_tensor(label_data.labels, dtype=torch.long)
        return window, labels

    collate = staticmethod(WindowedEMGDataset.collate)


def get_recons_dataloaders(
    data_dir: Path,
    config_path: Path,
    window_length: int = 250,
    stride: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    train_fraction: float = 1.0,
) -> dict[str, DataLoader]:
    """Build train / val / test DataLoaders for reconstructed EMG v3 files.

    Mirrors ``get_latent_dataloaders`` but reads ``_recons_v3.hdf5`` files and
    uses :class:`ReconstructedEMGDataset` (preloads sessions into RAM).

    Args:
        data_dir:      Directory containing ``*_recons_v3.hdf5`` session files.
        config_path:   Path to split YAML (same format as raw ``single_user.yaml``).
        window_length: Frames per window (default 250 ≈ 4 s at 62.5 Hz).
        stride:        Stride between windows; defaults to ``window_length``.
        batch_size:    Batch size for train/val. Test always uses 1.
        num_workers:   DataLoader worker processes.
        train_fraction: Fraction of training windows to use (0.0, 1.0].

    Returns:
        Dict with keys ``'train'``, ``'val'``, ``'test'`` → DataLoader.
    """
    assert 0.0 < train_fraction <= 1.0
    _stride = stride if stride is not None else window_length

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    def _resolve_paths(split: str) -> list[Path]:
        paths = []
        for entry in cfg["dataset"].get(split, []):
            p = data_dir / f"{entry['session']}_recons_v3.hdf5"
            if not p.exists():
                raise FileNotFoundError(f"Recons HDF5 not found: {p}")
            paths.append(p)
        return paths

    train_paths = _resolve_paths("train")
    val_paths   = _resolve_paths("val")
    test_paths  = _resolve_paths("test")

    print("Preloading reconstructed EMG into RAM...", flush=True)
    train_dataset: torch.utils.data.Dataset = ReconstructedEMGDataset(
        train_paths, window_length=window_length, stride=_stride, jitter=True
    )
    val_dataset   = ReconstructedEMGDataset(
        val_paths,   window_length=window_length, stride=_stride, jitter=False
    )
    test_dataset  = ReconstructedEMGDataset(
        test_paths,  window_length=window_length, stride=_stride, jitter=False
    )
    print("Done.", flush=True)

    n_total = len(train_dataset)
    if train_fraction < 1.0:
        n_subset = max(1, int(train_fraction * n_total))
        print(f"  train_fraction={train_fraction:.2f}: using {n_subset} / {n_total} training windows")
        train_dataset = Subset(train_dataset, list(range(n_subset)))
    else:
        print(f"  train_fraction=1.00: using all {n_total} training windows")

    print(
        f"Recons split — train sessions: {len(train_paths)}"
        f" | val sessions: {len(val_paths)}"
        f" | test sessions: {len(test_paths)}"
    )

    persistent = num_workers > 0
    collate = ReconstructedEMGDataset.collate
    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=collate,
                            pin_memory=True, persistent_workers=persistent),
        "val":   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate,
                            pin_memory=True, persistent_workers=persistent),
        "test":  DataLoader(test_dataset,  batch_size=1,          shuffle=False,
                            num_workers=num_workers, collate_fn=collate,
                            pin_memory=True, persistent_workers=persistent),
    }
