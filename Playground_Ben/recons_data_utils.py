"""Data utilities for AE-reconstructed EMG (_recons_v3.hdf5) files.

Mirrors the API of Playground_Kai/data_utils.py so reconstructed-EMG
training scripts can swap in get_recons_dataloaders() with minimal changes.

Key differences from the original pipeline:

    1. Schema:  _recons_v3.hdf5 stores separate datasets
                emg2qwerty/timeseries/emg_left  (T, 16) float32
                emg2qwerty/timeseries/emg_right (T, 16) float32
                emg2qwerty/timeseries/time       (T,)   float64
                and keystrokes as a JSON string in emg2qwerty.attrs.
                Raw files use a single structured numpy array.

    2. Rate:    Reconstructed signals are at ~62.5 Hz (32x decimated from
                2000 Hz).  window_length is therefore in recons frames, not
                raw EMG samples.  A 4-second window = 250 frames.

    3. Content: Signals are AE decoder outputs, not raw EMG.

Output batch dict (identical to WindowedEMGDataset):
    inputs         (T, N, 2, 16)  — float32, time-first
    targets        (T_tgt, N)     — int32 label indices
    input_lengths  (N,)           — unpadded input lengths
    target_lengths (N,)           — unpadded target lengths

Typical usage:
    from Playground_Ben.recons_data_utils import get_recons_dataloaders
    loaders = get_recons_dataloaders(
        data_root=Path("data"),
        config_path=Path("config/user/single_user.yaml"),
        window_length=250,   # 4 s at 62.5 Hz
    )
    for batch in loaders["train"]:
        inputs = batch["inputs"]   # (T, N, 2, 16)
        targets = batch["targets"] # (T_tgt, N)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from emg2qwerty.data import LabelData
from emg2qwerty.transforms import ForEach, RandomBandRotation, TemporalAlignmentJitter


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowedReconstructedEMGDataset(torch.utils.data.Dataset):
    """Windowed dataset over AE-reconstructed EMG (_recons_v3.hdf5) files.

    Reads emg_left (T, 16) and emg_right (T, 16) from the HDF5 file,
    stacks them into a (T, 2, 16) tensor, extracts keystroke labels from
    the group's 'keystrokes' attribute, and applies an optional transform.

    The __getitem__ contract and collate format are identical to
    WindowedEMGDataset so existing training loops work without changes.

    Args:
        hdf5_path:     Path to a *_recons_v3.hdf5 file.
        window_length: Recons frames per sample (default 250 = 4 s at 62.5 Hz).
        stride:        Frame stride between windows. Defaults to window_length
                       (no overlap).
        padding:       (left, right) extra frames appended to each window for
                       context. Mirrored from WindowedEMGDataset.
        jitter:        If True, randomly shift each window by up to one stride.
                       Use for training only.
        transform:     Optional callable applied to the (T+pad, 2, 16) tensor
                       before returning. Receives a torch.Tensor, returns a
                       torch.Tensor.
    """

    def __init__(
        self,
        hdf5_path: Path,
        window_length: int = 250,
        stride: Optional[int] = None,
        padding: tuple[int, int] = (0, 0),
        jitter: bool = False,
        transform=None,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.window_length = window_length
        self.stride = stride if stride is not None else window_length
        self.left_pad, self.right_pad = padding
        self.jitter = jitter
        self.transform = transform

        # Read metadata once — keep file closed so the dataset is picklable
        # for multi-process DataLoader workers.
        with h5py.File(self.hdf5_path, "r") as f:
            grp = f["emg2qwerty"]
            self._total_frames: int = grp["timeseries"]["emg_left"].shape[0]
            self._keystrokes: list = json.loads(grp.attrs.get("keystrokes", "[]"))

        assert self.window_length > 0 and self.stride > 0
        assert self._total_frames >= self.window_length, (
            f"{self.hdf5_path.name}: only {self._total_frames} frames, "
            f"need at least {self.window_length}"
        )

        self._file: Optional[h5py.File] = None

    # ------------------------------------------------------------------
    # Lazy file handle (survives fork into DataLoader workers)
    # ------------------------------------------------------------------

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __len__(self) -> int:
        return int((self._total_frames - self.window_length) // self.stride + 1)

    # ------------------------------------------------------------------
    # Item retrieval
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        f = self._open()
        ts = f["emg2qwerty"]["timeseries"]

        offset = idx * self.stride

        # Optional per-window random jitter (train only)
        if self.jitter:
            leftover = (self._total_frames - self.window_length) - offset
            if leftover > 0:
                offset += int(np.random.randint(0, min(self.stride, leftover)))

        # Padded window bounds (clamped to valid range)
        start = max(0, offset - self.left_pad)
        end   = min(self._total_frames, offset + self.window_length + self.right_pad)

        # Load left and right bands → (T_win, 2, 16)
        left  = ts["emg_left"][start:end].astype("float32")   # (T_win, 16)
        right = ts["emg_right"][start:end].astype("float32")  # (T_win, 16)
        data  = np.stack([left, right], axis=1)               # (T_win, 2, 16)
        tensor = torch.from_numpy(data)                       # (T_win, 2, 16)

        # Apply optional transform (e.g. augmentation)
        if self.transform is not None:
            tensor = self.transform(tensor)

        # Keystroke labels for this window
        start_t = float(ts["time"][start])
        end_t   = float(ts["time"][end - 1])
        label_data = LabelData.from_keystrokes(self._keystrokes, start_t, end_t)
        labels = torch.as_tensor(label_data.labels, dtype=torch.long)

        return tensor, labels

    # ------------------------------------------------------------------
    # Collate — identical contract to WindowedEMGDataset.collate
    # ------------------------------------------------------------------

    @staticmethod
    def collate(samples):
        """Pad a list of (tensor, labels) pairs into a batch dict.

        Returns the same keys as WindowedEMGDataset.collate so existing
        training loops work without modification:
            inputs, targets, input_lengths, target_lengths
        """
        inputs  = [s[0] for s in samples]  # list of (T, ...)
        targets = [s[1] for s in samples]  # list of (T_tgt,)

        input_batch  = nn.utils.rnn.pad_sequence(inputs)   # (T_max, N, ...)
        target_batch = nn.utils.rnn.pad_sequence(targets)  # (T_tgt_max, N)

        input_lengths  = torch.as_tensor([len(x) for x in inputs],  dtype=torch.int32)
        target_lengths = torch.as_tensor([len(t) for t in targets], dtype=torch.int32)

        return {
            "inputs":         input_batch,
            "targets":        target_batch,
            "input_lengths":  input_lengths,
            "target_lengths": target_lengths,
        }


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def _build_train_transform(max_offset: int = 4):
    """Augmented transform for reconstructed EMG (training).

    Pipeline:  ForEach(RandomBandRotation) → TemporalAlignmentJitter

    No spectrogram step: at 62.5 Hz the STFT produces too few time frames
    (<15 per 4-second window) to be useful for CTC decoding. The raw
    channel amplitudes are used directly as features.

    Input/output shape: (T, 2, 16)

    Args:
        max_offset: Maximum temporal alignment jitter in recons frames
                    (default 4 ≈ 64 ms at 62.5 Hz, analogous to the 120-sample
                    jitter at 2000 Hz in the raw pipeline).
    """
    from emg2qwerty.transforms import Compose
    return Compose([
        ForEach(RandomBandRotation(offsets=[-1, 0, 1])),
        TemporalAlignmentJitter(max_offset=max_offset),
    ])


def _build_eval_transform():
    """Deterministic eval transform — identity (no augmentation)."""
    return None


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def get_recons_session_paths(
    data_root: Path,
    config_path: Path,
    suffix: str = "_recons_v3",
) -> dict[str, list[Path]]:
    """Parse single_user.yaml and return _recons_v3.hdf5 paths per split.

    Args:
        data_root:   Directory containing the *_recons_v3.hdf5 files.
        config_path: Path to config/user/single_user.yaml.
        suffix:      File suffix to append to session name (default '_recons_v3').

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to lists of Path objects.

    Raises:
        FileNotFoundError: If a required session file is missing.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    result: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        entries = cfg["dataset"].get(split, [])
        paths: list[Path] = []
        for entry in entries:
            p = data_root / f"{entry['session']}{suffix}.hdf5"
            if not p.exists():
                raise FileNotFoundError(f"Recons HDF5 not found: {p}")
            paths.append(p)
        result[split] = paths
    return result


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_recons_dataloaders(
    data_root: Path,
    config_path: Path,
    window_length: int = 250,
    stride: Optional[int] = None,
    padding: tuple[int, int] = (0, 0),
    batch_size: int = 32,
    num_workers: int = 0,
    augment: bool = True,
    test_window_length: Optional[int] = None,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders for _recons_v3.hdf5 sessions.

    Mirrors get_dataloaders() from Playground_Kai/data_utils.py.

    Args:
        data_root:          Directory containing *_recons_v3.hdf5 files.
        config_path:        Path to config/user/single_user.yaml.
        window_length:      Recons frames per training window (default 250 = 4 s
                            at 62.5 Hz).
        stride:             Frame stride between windows. Defaults to
                            window_length (no overlap).
        padding:            (left, right) extra frames for context.
        batch_size:         Batch size for train and val. Test uses 1.
        num_workers:        DataLoader worker processes (0 = main process).
        augment:            If True, apply band rotation + temporal jitter to
                            the training split.
        test_window_length: Window length for test split. If None, the whole
                            session is used as one sample.

    Returns:
        Dict with keys 'train', 'val', 'test' → DataLoader.
    """
    paths = get_recons_session_paths(data_root=data_root, config_path=config_path)

    train_transform = _build_train_transform() if augment else None
    eval_transform  = _build_eval_transform()

    train_dataset = ConcatDataset([
        WindowedReconstructedEMGDataset(
            hdf5_path=p,
            window_length=window_length,
            stride=stride,
            padding=padding,
            jitter=augment,
            transform=train_transform,
        )
        for p in paths["train"]
    ])

    val_dataset = ConcatDataset([
        WindowedReconstructedEMGDataset(
            hdf5_path=p,
            window_length=window_length,
            stride=None,      # no overlap for val
            padding=padding,
            jitter=False,
            transform=eval_transform,
        )
        for p in paths["val"]
    ])

    # Test: whole session by default; windowed when test_window_length is set
    _test_wlen = test_window_length
    test_dataset = ConcatDataset([
        WindowedReconstructedEMGDataset(
            hdf5_path=p,
            window_length=_test_wlen if _test_wlen is not None else (
                # fall back to full session length
                WindowedReconstructedEMGDataset(p)._total_frames
            ),
            stride=None,
            padding=(0, 0) if _test_wlen is None else padding,
            jitter=False,
            transform=eval_transform,
        )
        for p in paths["test"]
    ])

    persistent = num_workers > 0
    collate = WindowedReconstructedEMGDataset.collate

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
            persistent_workers=persistent,
        ),
    }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.chdir(_ROOT)

    loaders = get_recons_dataloaders(
        data_root=Path("data"),
        config_path=Path("config/user/single_user.yaml"),
        window_length=250,
        batch_size=4,
    )
    print(f"train: {len(loaders['train'].dataset)} windows")
    print(f"val:   {len(loaders['val'].dataset)} windows")
    print(f"test:  {len(loaders['test'].dataset)} windows")

    batch = next(iter(loaders["train"]))
    print(f"inputs shape:  {batch['inputs'].shape}")   # (T, N, 2, 16)
    print(f"targets shape: {batch['targets'].shape}")
    print(f"input_lengths: {batch['input_lengths']}")
    print(f"target_lengths:{batch['target_lengths']}")
