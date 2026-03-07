"""Training and evaluation script for the TDS CNN baseline model.

Run from the workspace root with:
    python -m Playground_Mumbi.train [options]
    python Playground_Mumbi/train.py [options]

Key flags (all have defaults, so a bare invocation will start training):
    --epochs            Number of training epochs          (default: 80)
    --batch-size        Batch size                         (default: 32)
    --lr                Peak learning rate                 (default: 1e-3)
    --train-fraction    Fraction of training windows to use (default: 1.0)
    --run-all-fractions Run sequentially for fractions [0.10, 0.25, 0.50, 0.75, 1.00]
                        and log each run with Playground_Mumbi.logger
    --resume            Path to a checkpoint .pt to continue training
    --num-workers       DataLoader workers (default: 0, safe on Windows)

Model hyperparameters match config/model/tds_conv_ctc.yaml exactly:
    in_features=528, mlp_features=[384], block_channels=[24,24,24,24],
    kernel_width=32

The best checkpoint (lowest val CER) is saved to
    Playground_Mumbi/checkpoints/final_models/best_cnn.pt          (--train-fraction 1.0)
    Playground_Mumbi/checkpoints/training_fraction_ablation/best_cnn_{pct}pct.pt  (otherwise)
and automatically loaded for the final test evaluation.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

# Ensure workspace root is on sys.path so both `emg2qwerty` and `Playground_Mumbi`
# are importable regardless of where the script is invoked from.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData
from emg2qwerty.decoder import CTCGreedyDecoder
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)

from Playground_Mumbi.data_utils import get_dataloaders
from scripts.logger import log_epoch, log_summary, make_run_id


# ---------------------------------------------------------------------------
# TDS model constants (from config/model/tds_conv_ctc.yaml)
# ---------------------------------------------------------------------------

_NUM_BANDS: int = 2
_ELECTRODE_CHANNELS: int = 16
_IN_FEATURES: int = 528           # (n_fft // 2 + 1) * 16 = 33 * 16
_MLP_FEATURES: list[int] = [384]
_BLOCK_CHANNELS: list[int] = [24, 24, 24, 24]
_KERNEL_WIDTH: int = 32
_NUM_FEATURES: int = _NUM_BANDS * _MLP_FEATURES[-1]  # 2 * 384 = 768

# Training fractions for --run-all-fractions mode
_ALL_FRACTIONS: list[float] = [0.10, 0.25, 0.50, 0.75, 1.00]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    """Build the TDS CNN model matching config/model/tds_conv_ctc.yaml.

    Architecture mirrors TDSConvCTCModule.model:
        SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten
        → TDSConvEncoder → Linear → LogSoftmax

    Returns:
        An :class:`nn.Sequential` whose forward expects ``(T, N, 2, 16, freq)``
        and outputs log-probabilities ``(T', N, num_classes)`` where T' < T
        due to the temporal receptive field of TDSConvEncoder.
    """
    return nn.Sequential(
        # Normalise spectrogram across all band-electrode channels
        SpectrogramNorm(channels=_NUM_BANDS * _ELECTRODE_CHANNELS),
        # Per-band rotation-invariant MLP: (T, N, 2, 16, freq) → (T, N, 2, 384)
        MultiBandRotationInvariantMLP(
            in_features=_IN_FEATURES,
            mlp_features=_MLP_FEATURES,
            num_bands=_NUM_BANDS,
        ),
        # (T, N, 2, 384) → (T, N, 768)
        nn.Flatten(start_dim=2),
        # Temporal conv blocks: (T, N, 768) → (T', N, 768)
        TDSConvEncoder(
            num_features=_NUM_FEATURES,
            block_channels=_BLOCK_CHANNELS,
            kernel_width=_KERNEL_WIDTH,
        ),
        # (T', N, 768) → (T', N, num_classes)
        nn.Linear(_NUM_FEATURES, charset().num_classes),
        nn.LogSoftmax(dim=-1),
    )


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup + cosine annealing (step-level)
# ---------------------------------------------------------------------------

def _lr_lambda(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    """Multiplier for LambdaLR: linear warmup then cosine decay.

    Returns a value in [min_lr_ratio, 1.0].  The scheduler is expected to be
    stepped once per *batch* (not per epoch).
    """
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
    scheduler,
) -> float:
    """Run one full training epoch and return the mean CTC loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)           # (T, N, 2, 16, freq)
        targets = batch["targets"].to(device)         # (T_tgt, N)
        input_lengths = batch["input_lengths"].to(device)   # (N,)
        target_lengths = batch["target_lengths"].to(device) # (N,)

        optimizer.zero_grad()

        emissions = model(inputs)   # (T', N, num_classes)

        # TDSConvEncoder shrinks the temporal dimension due to its receptive
        # field.  Adjust input_lengths accordingly for CTCLoss.
        t_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - t_diff

        # CTCLoss expects targets as (N, T_tgt) — transpose from (T_tgt, N).
        # Emissions are already log-probabilities (LogSoftmax in model).
        loss = criterion(
            emissions,                      # (T', N, num_classes)
            targets.transpose(0, 1),        # (N, T_tgt)
            emission_lengths,               # (N,)
            target_lengths,                 # (N,)
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    decoder: CTCGreedyDecoder,
) -> tuple[float, dict[str, float]]:
    """Evaluate the model on a DataLoader.

    Returns:
        Tuple of (mean_loss, metrics_dict) where metrics_dict contains
        CER, IER, DER, SER as percentages.
    """
    model.eval()
    total_loss = 0.0
    criterion = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    cer_metric = CharacterErrorRates()

    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"]            # keep on CPU for numpy conversion
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        # cuDNN's LSTM kernel doesn't support very long sequences (entire test
        # sessions can be 40k+ frames after the spectrogram hop).  Disabling
        # cuDNN here falls back to PyTorch's generic CUDA kernel which is correct
        # for any length.
        with torch.backends.cudnn.flags(enabled=False):
            emissions = model(inputs)         # (T', N, num_classes)

        t_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = (input_lengths - t_diff).to(device)

        loss = criterion(
            emissions,
            targets.transpose(0, 1).to(device),
            emission_lengths,
            target_lengths.to(device),
        )
        total_loss += loss.item()

        preds = decoder.decode_batch(
            emissions=emissions.cpu().numpy(),
            emission_lengths=emission_lengths.cpu().numpy(),
        )

        targets_np = targets.numpy()
        tgt_lens = target_lengths.numpy()
        for i, pred in enumerate(preds):
            target = LabelData.from_labels(targets_np[: tgt_lens[i], i])
            cer_metric.update(prediction=pred, target=target)

    metrics = cer_metric.compute()
    return total_loss / max(len(loader), 1), metrics


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace, train_fraction: float, notes: str) -> None:
    """Execute one complete train + test cycle for a given train_fraction.

    Args:
        args:           Parsed CLI arguments.
        train_fraction: Fraction of training windows to use (0.0, 1.0].
        notes:          Value written to the ``notes`` field in the run summary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"train_fraction = {train_fraction:.2f}  |  notes = {notes}")
    print(f"Device         : {device}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Building data loaders …")
    loaders = get_dataloaders(
        data_root=args.data_root,
        config_path=args.config,
        window_length=args.window_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_window_length=args.window_length,
        train_fraction=train_fraction,
    )
    print(
        f"  train batches : {len(loaders['train'])}"
        f" | val batches : {len(loaders['val'])}"
        f" | test batches : {len(loaders['test'])}"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model().to(device)
    print(f"Model       : CNN (TDS baseline)")
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Logger run ID
    # ------------------------------------------------------------------
    run_id = make_run_id(
        model="CNN",
        num_channels=32,
        sampling_rate_hz=2000,
        train_fraction=train_fraction,
    )
    print(f"Run ID      : {run_id}")

    # ------------------------------------------------------------------
    # Loss / optimiser / LR schedule
    # ------------------------------------------------------------------
    # CTCLoss receives log-probabilities (LogSoftmax is part of the model).
    # zero_infinity=True prevents NaN loss for very short windows.
    criterion = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = len(loaders["train"])
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    min_lr_ratio = args.min_lr / args.lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps, min_lr_ratio),
    )

    decoder = CTCGreedyDecoder()

    # ------------------------------------------------------------------
    # Optional resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_cer = float("inf")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_cer = ckpt.get("best_cer", float("inf"))
        print(f"Resumed from epoch {start_epoch} | best val CER {best_cer:.2f}%")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    total_epochs = args.epochs - start_epoch
    print(f"\nTraining for {total_epochs} epochs …")
    print("Tip: press Ctrl+C at any time to stop early and jump to test evaluation.\n")

    t_train_start = time.perf_counter()
    epoch_duration: float = 0.0
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    val_cer: float = float("nan")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        try:
            train_loss = train_one_epoch(
                model, loaders["train"], optimizer, criterion, device, scheduler
            )
            val_loss, val_metrics = evaluate(model, loaders["val"], device, decoder)
        except KeyboardInterrupt:
            print("\n[Interrupted] Stopping training early — jumping to test evaluation.")
            break
        val_cer = val_metrics["CER"]

        epoch_duration = time.perf_counter() - t0
        epochs_left = args.epochs - (epoch + 1)
        eta_secs = epoch_duration * epochs_left
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_secs))

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs}"
            f" | train_loss={train_loss:.4f}"
            f" | val_loss={val_loss:.4f}"
            f" | val_CER={val_cer:.2f}%"
            f" | epoch_time={epoch_duration:.1f}s"
            f" | ETA={eta_str}"
        )

        # Log epoch to CSV
        log_epoch(
            run_id=run_id,
            model="CNN",
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            val_cer=val_cer,
        )

        # Save checkpoint whenever val CER improves
        if val_cer < best_cer:
            best_cer = val_cer
            fraction_pct = int(round(train_fraction * 100))
            if train_fraction < 1.0:
                ckpt_dir = args.checkpoint_dir / "training_fraction_ablation"
                ckpt_path = ckpt_dir / f"best_cnn_{fraction_pct}pct.pt"
            else:
                ckpt_dir = args.checkpoint_dir / "final_models"
                ckpt_path = ckpt_dir / "best_cnn.pt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            safe_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_cer": best_cer,
                    "args": safe_args,
                },
                ckpt_path,
            )
            print(f"  -> Saved best checkpoint  (val CER={best_cer:.2f}%)")

    training_time_sec = time.perf_counter() - t_train_start
    epochs_completed = epoch + 1 - start_epoch if "epoch" in dir() else 0

    # ------------------------------------------------------------------
    # Test evaluation using the best checkpoint
    # ------------------------------------------------------------------
    print("\n--- Test Evaluation ---")
    fraction_pct = int(round(train_fraction * 100))
    if train_fraction < 1.0:
        best_ckpt = args.checkpoint_dir / "training_fraction_ablation" / f"best_cnn_{fraction_pct}pct.pt"
    else:
        best_ckpt = args.checkpoint_dir / "final_models" / "best_cnn.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch'] + 1}")

    _, test_metrics = evaluate(model, loaders["test"], device, decoder)
    test_cer = test_metrics["CER"]
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.2f}%")

    # ------------------------------------------------------------------
    # Log summary
    # ------------------------------------------------------------------
    log_summary(
        run_id=run_id,
        model="CNN",
        epochs=epochs_completed,
        num_channels=32,
        sampling_rate_hz=2000,
        train_fraction=train_fraction,
        input_type="spectrogram",
        final_train_loss=train_loss,
        final_val_loss=val_loss,
        final_val_cer=val_cer,
        test_cer=test_cer,
        training_time_sec=training_time_sec,
        notes=notes,
    )
    print(f"Results logged — run_id: {run_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the TDS CNN baseline model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--data-root", type=Path, default=_ROOT / "data",
                   help="Directory containing *.hdf5 session files")
    p.add_argument("--config", type=Path,
                   default=_ROOT / "config" / "user" / "single_user.yaml",
                   help="Path to the train/val/test split YAML")
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path(__file__).resolve().parent / "checkpoints",
                   help="Directory to write model checkpoints")
    # Training
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (0 = main process, safest on Windows)")
    p.add_argument("--window-length", type=int, default=8000,
                   help="Raw EMG samples per training/test window (8000 = 4 s @ 2 kHz)")
    # Optimiser / scheduler
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Peak (post-warmup) learning rate for Adam")
    p.add_argument("--min-lr", type=float, default=1e-5,
                   help="Minimum learning rate at the end of cosine decay")
    p.add_argument("--warmup-epochs", type=int, default=10,
                   help="Epochs of linear LR warmup before cosine decay begins")
    # Data fraction
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="Fraction of training windows to use, in (0.0, 1.0]")
    p.add_argument("--run-all-fractions", action="store_true",
                   help=(
                       "Run training sequentially for fractions "
                       f"{_ALL_FRACTIONS} and log each run with Playground_Mumbi.logger"
                   ))
    # Resume
    p.add_argument("--resume", type=Path, default=None,
                   help="Checkpoint .pt file to resume training from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.run_all_fractions:
        print(f"--run-all-fractions: will train for fractions {_ALL_FRACTIONS}")
        for frac in _ALL_FRACTIONS:
            run_training(args, train_fraction=frac, notes="ablation_train_fraction")
    else:
        notes = "arch_comparison" if args.train_fraction == 1.0 else "ablation_train_fraction"
        run_training(args, train_fraction=args.train_fraction, notes=notes)


if __name__ == "__main__":
    main()
