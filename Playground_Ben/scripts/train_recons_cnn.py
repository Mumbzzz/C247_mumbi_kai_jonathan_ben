"""CNN training on AE-reconstructed EMG data (_recons_v3.hdf5 files).

The reconstructed files contain AE-decoded EMG signals at 62.5 Hz (16 ms/frame)
with the same 16-electrode layout as the originals but stored as plain float32
arrays (not the structured numpy dtype of the raw HDF5 files).

Schema per session file:
    emg2qwerty/timeseries/emg_left  : (T, 16) float32
    emg2qwerty/timeseries/emg_right : (T, 16) float32
    emg2qwerty/timeseries/time      : (T,)    float64
    emg2qwerty attrs                : keystrokes (JSON)

Input to model: (T, N, 32)  — left and right channels flattened per frame
All 18 sessions are available (vs 17 for latent_v2).

Results are appended to results/results_summary_CNN.csv with
notes="reconstructed_emg_cnn" — existing rows are never overwritten.

Run from repo root:
    python Playground_Ben/scripts/train_recons_cnn.py
    python Playground_Ben/scripts/train_recons_cnn.py --epochs 150 --stride 250
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import h5py
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.decoder import CTCGreedyDecoder
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import TDSConvEncoder

from scripts.logger import log_epoch, log_summary, make_run_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_BANDS:          int = 2   # left + right wrist
ELECTRODE_CHANNELS: int = 16  # per band
IN_FEATURES:        int = NUM_BANDS * ELECTRODE_CHANNELS  # 32 per frame


# ---------------------------------------------------------------------------
# Dataset — preloaded in RAM for fast random access
# ---------------------------------------------------------------------------

class ReconstructedEMGDataset(torch.utils.data.Dataset):
    """Windowed dataset over AE-reconstructed EMG (_recons_v3.hdf5) files.

    Preloads all sessions into RAM at init to avoid repeated HDF5 seeks
    during shuffled training. 18 sessions × ~74k frames × 32 floats ≈ 340 MB.

    Each item is a (window_length, 32) float32 tensor (left+right stacked).
    """

    def __init__(
        self,
        paths: list[Path],
        window_length: int,
        stride: int,
        jitter: bool,
    ) -> None:
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

            # Stack and flatten: (T, 32)
            data = torch.from_numpy(
                __import__("numpy").concatenate([left, right], axis=1)
            )
            sess_idx = len(self._sessions)
            self._sessions.append((data, times, ks))

            n_frames = data.shape[0]
            for offset in range(0, n_frames - window_length + 1, stride):
                self._windows.append((sess_idx, offset))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ReconsTDSConvCTC(nn.Module):
    """TDSConvCTC for reconstructed EMG input.

    Input shape: (T, N, 32)  — 16 left + 16 right channels per frame.
    Projects to num_features, then TDSConvEncoder + CTC head.
    """

    def __init__(
        self,
        in_features: int = IN_FEATURES,
        num_features: int = 768,
        block_channels: int = 24,
        num_blocks: int = 4,
        kernel_width: int = 32,
    ) -> None:
        super().__init__()
        bc = block_channels
        num_features = max(bc, round(num_features / bc) * bc)

        self.model = nn.Sequential(
            nn.Linear(in_features, num_features),
            nn.ReLU(),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=[bc] * num_blocks,
                kernel_width=kernel_width,
            ),
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _lr_lambda(step, warmup_steps, total_steps, min_lr_ratio=0.0):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss = 0.0
    for batch in loader:
        inputs         = batch["inputs"].to(device)
        targets        = batch["targets"].to(device)
        input_lengths  = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad()
        emissions = model(inputs)

        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = (input_lengths - T_diff).clamp(min=1)

        loss = criterion(emissions, targets.transpose(0, 1),
                         emission_lengths, target_lengths)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, decoder):
    model.eval()
    total_loss = 0.0
    criterion  = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    cer_metric = CharacterErrorRates()
    n_valid    = 0

    for batch in loader:
        inputs         = batch["inputs"].to(device)
        targets        = batch["targets"]
        input_lengths  = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        with torch.backends.cudnn.flags(enabled=False):
            emissions = model(inputs)

        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = (input_lengths - T_diff).clamp(min=1)

        loss = criterion(
            emissions,
            targets.transpose(0, 1).to(device),
            emission_lengths.to(device),
            target_lengths.to(device),
        )
        total_loss += loss.item()

        preds = decoder.decode_batch(
            emissions=emissions.cpu().numpy(),
            emission_lengths=emission_lengths.numpy(),
        )
        targets_np = targets.numpy()
        tgt_lens   = target_lengths.numpy()
        for i, pred in enumerate(preds):
            tgt_len = int(tgt_lens[i])
            if tgt_len == 0:
                continue
            target = LabelData.from_labels(targets_np[:tgt_len, i])
            cer_metric.update(prediction=pred, target=target)
            n_valid += tgt_len

    if n_valid == 0:
        return total_loss / max(len(loader), 1), {"CER": float("nan")}
    return total_loss / max(len(loader), 1), cer_metric.compute()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _build_loaders(
    config_path: Path,
    data_root: Path,
    window_length: int,
    stride: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    split_paths: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    missing: list[str] = []

    for split, entries in cfg["dataset"].items():
        for e in entries:
            p = data_root / f'{e["session"]}_recons_v3.hdf5'
            if p.exists():
                split_paths[split].append(p)
            else:
                missing.append(f"  {split}: {p.name}")

    if missing:
        print(f"WARNING — {len(missing)} file(s) not found, skipping:")
        for m in missing:
            print(m)

    print("Preloading reconstructed EMG into RAM...", flush=True)
    train_ds = ReconstructedEMGDataset(split_paths["train"], window_length, stride,        jitter=True)
    val_ds   = ReconstructedEMGDataset(split_paths["val"],   window_length, window_length, jitter=False)
    test_ds  = ReconstructedEMGDataset(split_paths["test"],  window_length, window_length, jitter=False)
    print("Done.", flush=True)

    collate    = ReconstructedEMGDataset.collate
    persistent = num_workers > 0
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=collate,
                            pin_memory=True, persistent_workers=persistent),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate,
                            pin_memory=True, persistent_workers=persistent),
        "test":  DataLoader(test_ds,  batch_size=1,          shuffle=False,
                            num_workers=num_workers, collate_fn=collate,
                            pin_memory=True, persistent_workers=persistent),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train CNN on AE-reconstructed EMG (_recons_v3.hdf5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root",      type=Path,  default=_ROOT / "data")
    p.add_argument("--config",         type=Path,  default=_ROOT / "config" / "user" / "single_user.yaml")
    p.add_argument("--checkpoint-dir", type=Path,  default=Path(__file__).resolve().parents[1] / "checkpoints")
    p.add_argument("--epochs",         type=int,   default=150)
    p.add_argument("--batch-size",     type=int,   default=32)
    p.add_argument("--num-workers",    type=int,   default=0)
    p.add_argument("--window-length",  type=int,   default=250,
                   help="Frames per window (~4 s at 16 ms/frame = 62.5 Hz)")
    p.add_argument("--stride",         type=int,   default=250,
                   help="Stride between windows in frames")
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight-decay",   type=float, default=1e-5)
    p.add_argument("--warmup-frac",    type=float, default=0.05)
    p.add_argument("--num-features",   type=int,   default=768)
    p.add_argument("--block-channels", type=int,   default=24)
    p.add_argument("--num-blocks",     type=int,   default=4)
    p.add_argument("--kernel-width",   type=int,   default=32)
    p.add_argument("--resume",         type=Path,  default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:      {device}")
    print(f"in_features: {IN_FEATURES}  ({NUM_BANDS} bands × {ELECTRODE_CHANNELS} channels)")

    loaders = _build_loaders(
        config_path=args.config,
        data_root=args.data_root,
        window_length=args.window_length,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Dataset  — train: {len(loaders['train'].dataset):,} windows"
        f" | val: {len(loaders['val'].dataset):,} windows"
        f" | test: {len(loaders['test'].dataset):,} windows"
    )

    model = ReconsTDSConvCTC(
        in_features=IN_FEATURES,
        num_features=args.num_features,
        block_channels=args.block_channels,
        num_blocks=args.num_blocks,
        kernel_width=args.kernel_width,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters:  {n_params / 1e6:.2f} M")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")

    optimizer    = torch.optim.AdamW(model.parameters(),
                                     lr=args.lr, weight_decay=args.weight_decay)
    total_steps  = args.epochs * len(loaders["train"])
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps),
    )
    criterion = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    decoder   = CTCGreedyDecoder()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt    = args.checkpoint_dir / "best_recons_cnn.pt"
    best_val_cer = float("inf")

    run_id  = make_run_id("CNN", num_channels=IN_FEATURES,
                          sampling_rate_hz=63, train_fraction=1.0)
    t_start = time.time()
    print(f"\nRun ID: {run_id}")
    print(f"Starting training for {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        t_epoch = time.time()

        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, scheduler
        )
        val_loss, val_metrics = evaluate(model, loaders["val"], device, decoder)

        val_cer   = val_metrics["CER"]
        epoch_sec = time.time() - t_epoch

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_CER={val_cer:.2f}% | "
            f"{epoch_sec:.0f}s"
        )

        log_epoch(run_id=run_id, model="CNN", epoch=epoch + 1,
                  train_loss=train_loss, val_loss=val_loss, val_cer=val_cer)

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                        "val_cer": val_cer}, best_ckpt)
            print(f"  --> New best val CER: {best_val_cer:.2f}% (saved)")

    print("\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    _, test_metrics = evaluate(model, loaders["test"], device, decoder)
    test_cer        = test_metrics["CER"]
    training_sec    = time.time() - t_start

    print(f"\nTest CER:      {test_cer:.2f}%")
    print(f"Training time: {training_sec / 3600:.2f} hrs")

    log_summary(
        run_id=run_id,
        model="CNN",
        epochs=args.epochs,
        num_channels=IN_FEATURES,
        sampling_rate_hz=63,
        train_fraction=1.0,
        input_type="reconstructed_emg_v3",
        final_train_loss=train_loss,
        final_val_loss=val_loss,
        final_val_cer=best_val_cer,
        test_cer=test_cer,
        training_time_sec=training_sec,
        notes="reconstructed_emg_cnn",
    )
    print(f"\nDone. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
