"""CNN baseline training on per-session latent v2 HDF5 files.

Uses the individual session *_latent_v2.hdf5 files with the same
train/val/test split defined in config/user/single_user.yaml.

Key differences from train_latent_cnn.py (single concatenated file):
  - Latent dim auto-detected from first file (256 in new files vs 1024 old)
  - Frame rate ~16 ms/frame → window_length=250 for a 4-second window
  - Proper session-based split (not 70/15/15 temporal split)
  - Missing session files are skipped with a warning
  - Results written to results/results_summary_CNN_latent_sessions.csv

Run from repo root:
    python Playground_Ben/scripts/train_latent_cnn_sessions.py
    python Playground_Ben/scripts/train_latent_cnn_sessions.py --epochs 150
"""

from __future__ import annotations

import argparse
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
# Model
# ---------------------------------------------------------------------------

class LatentTDSConvCTC(nn.Module):
    """TDSConvCTC for latent-vector input.

    Replaces SpectrogramNorm + MultiBandMLP with a single Linear projection,
    then feeds into the standard TDSConvEncoder + CTC head.
    """

    def __init__(
        self,
        latent_dim: int,
        num_features: int = 768,
        block_channels: list[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()
        # Snap num_features to multiple of block_channels
        bc = block_channels[0] if block_channels else 24
        num_features = max(bc, round(num_features / bc) * bc)

        self.model = nn.Sequential(
            nn.Linear(latent_dim, num_features),
            nn.ReLU(),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=list(block_channels),
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
# Data loading — preloaded in-memory dataset for fast random access
# ---------------------------------------------------------------------------

class PreloadedLatentDataset(torch.utils.data.Dataset):
    """Loads all latent frames from a list of HDF5 files into RAM at init.

    This eliminates repeated HDF5 seeks during training (which are slow when
    shuffle=True forces random access across many files). With 256-dim latents
    at ~16ms/frame, 15 sessions occupy ~1 GB — fits comfortably in RAM.
    """

    def __init__(
        self,
        paths: list[Path],
        window_length: int,
        stride: int,
        jitter: bool,
    ) -> None:
        import json
        self.window_length = window_length
        self.stride        = stride
        self.jitter        = jitter

        # Preload all sessions into a list of (latent_tensor, time_array, keystrokes)
        self._sessions: list[tuple[torch.Tensor, list]] = []
        self._windows:  list[tuple[int, int]] = []  # (session_idx, frame_offset)

        for p in paths:
            with h5py.File(p, "r") as f:
                grp     = f["emg2qwerty"]
                latent  = torch.from_numpy(grp["latent"][:].astype("float32"))
                times   = grp["time"][:].tolist()
                ks      = json.loads(grp.attrs.get("keystrokes", "[]"))
            sess_idx = len(self._sessions)
            self._sessions.append((latent, times, ks))

            n_frames = latent.shape[0]
            for offset in range(0, n_frames - window_length + 1, stride):
                self._windows.append((sess_idx, offset))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        sess_idx, offset = self._windows[idx]
        latent, times, ks = self._sessions[sess_idx]

        if self.jitter:
            max_jitter = min(self.stride, latent.shape[0] - self.window_length - offset)
            if max_jitter > 0:
                offset += int(torch.randint(0, max_jitter, (1,)).item())

        window = latent[offset : offset + self.window_length]  # (T, D)
        start_t = times[offset]
        end_t   = times[offset + self.window_length - 1]

        label_data = LabelData.from_keystrokes(ks, start_t, end_t)
        labels = torch.as_tensor(label_data.labels, dtype=torch.long)
        return window, labels

    collate = staticmethod(WindowedEMGDataset.collate)


def _detect_latent_dim(paths: list[Path]) -> int:
    for p in paths:
        if p.exists():
            with h5py.File(p, "r") as f:
                return f["emg2qwerty"]["latent"].shape[1]
    raise FileNotFoundError("No latent files found to detect latent dim")


def _build_loaders(
    config_path: Path,
    data_root: Path,
    window_length: int,
    stride: int,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, DataLoader], int]:
    """Build dataloaders from per-session latent v2 files.

    All latent data is preloaded into RAM at startup for fast random access.
    Returns (loaders_dict, latent_dim).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    split_paths: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    missing: list[str] = []

    for split, entries in cfg["dataset"].items():
        for e in entries:
            p = data_root / f'{e["session"]}_latent_v2.hdf5'
            if p.exists():
                split_paths[split].append(p)
            else:
                missing.append(f"  {split}: {p.name}")

    if missing:
        print(f"WARNING — {len(missing)} session file(s) not found, skipping:")
        for m in missing:
            print(m)

    latent_dim = _detect_latent_dim(
        split_paths["train"] + split_paths["val"] + split_paths["test"]
    )

    print("Preloading latent data into RAM...", flush=True)
    train_ds = PreloadedLatentDataset(split_paths["train"], window_length, stride,        jitter=True)
    val_ds   = PreloadedLatentDataset(split_paths["val"],   window_length, window_length, jitter=False)
    test_ds  = PreloadedLatentDataset(split_paths["test"],  window_length, window_length, jitter=False)
    print("Done preloading.", flush=True)

    persistent = num_workers > 0
    collate = PreloadedLatentDataset.collate

    loaders = {
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
    return loaders, latent_dim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train latent CNN on per-session latent_v2 HDF5 files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root",      type=Path,  default=_ROOT / "data")
    p.add_argument("--config",         type=Path,  default=_ROOT / "config" / "user" / "single_user.yaml")
    p.add_argument("--checkpoint-dir", type=Path,  default=Path(__file__).resolve().parents[1] / "checkpoints")
    p.add_argument("--epochs",         type=int,   default=150)
    p.add_argument("--batch-size",     type=int,   default=32)
    p.add_argument("--num-workers",    type=int,   default=0)
    p.add_argument("--window-length",  type=int,   default=250,
                   help="Latent frames per window (~4 s at 16 ms/frame)")
    p.add_argument("--stride",         type=int,   default=25,
                   help="Stride between windows in latent frames")
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight-decay",   type=float, default=1e-5)
    p.add_argument("--warmup-frac",    type=float, default=0.05)
    p.add_argument("--num-features",   type=int,   default=768,
                   help="TDS operating dimension")
    p.add_argument("--block-channels", type=int,   default=24)
    p.add_argument("--num-blocks",     type=int,   default=4)
    p.add_argument("--kernel-width",   type=int,   default=32)
    p.add_argument("--resume",         type=Path,  default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:  {device}")
    print(f"Config:  {args.config}")

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders, latent_dim = _build_loaders(
        config_path=args.config,
        data_root=args.data_root,
        window_length=args.window_length,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Latent dim: {latent_dim}  |  window: {args.window_length} frames  |  stride: {args.stride} frames"
        f"\nDataset  — train: {len(loaders['train'].dataset):,} windows"
        f" | val: {len(loaders['val'].dataset):,} windows"
        f" | test: {len(loaders['test'].dataset):,} windows"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LatentTDSConvCTC(
        latent_dim=latent_dim,
        num_features=args.num_features,
        block_channels=[args.block_channels] * args.num_blocks,
        kernel_width=args.kernel_width,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.2f} M")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")

    # ── Optimiser / scheduler ─────────────────────────────────────────────────
    optimizer    = torch.optim.AdamW(model.parameters(),
                                     lr=args.lr, weight_decay=args.weight_decay)
    total_steps  = args.epochs * len(loaders["train"])
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps),
    )
    criterion = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    decoder   = CTCGreedyDecoder()

    # ── Checkpointing ─────────────────────────────────────────────────────────
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt    = args.checkpoint_dir / "best_latent_cnn_sessions.pt"
    best_val_cer = float("inf")

    # ── Logging ───────────────────────────────────────────────────────────────
    run_id  = make_run_id("CNN", num_channels=latent_dim,
                          sampling_rate_hz=62, train_fraction=1.0)
    t_start = time.time()
    print(f"\nRun ID: {run_id}")
    print(f"Starting training for {args.epochs} epochs...\n")

    # ── Training loop ─────────────────────────────────────────────────────────
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

    # ── Final test evaluation ─────────────────────────────────────────────────
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
        num_channels=latent_dim,
        sampling_rate_hz=62,
        train_fraction=1.0,
        input_type=f"latent_v2_{latent_dim}dim_sessions",
        final_train_loss=train_loss,
        final_val_loss=val_loss,
        final_val_cer=best_val_cer,
        test_cer=test_cer,
        training_time_sec=training_sec,
        notes="latent_cnn_per_session_split",
    )
    print(f"\nDone. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
