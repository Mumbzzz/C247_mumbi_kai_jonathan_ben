"""Training and evaluation script for the latent-space CNN+LSTM hybrid model.

Operates on pre-computed AE latent vectors stored in
``data/preprocessed/emg_latent_ae_v2.hdf5``
(shape: N_frames × 1024 float32, at 32 ms / frame).

The SpectrogramNorm + MultiBandRotationInvariantMLP front-end from the raw-EMG
pipeline is removed; instead a single ``nn.Linear(1024, proj_features)`` projects
the latent vectors into the CNN+LSTM stack.

Run from the workspace root with:
    python -m Playground_Mumbi.train_latent [options]
    python Playground_Mumbi/train_latent.py [options]

Key flags:
    --epochs         Number of training epochs           (default: 150)
    --batch-size     Batch size                          (default: 32)
    --window-length  Latent frames per window            (default: 125, ≈ 4 s @ 32 ms)
    --hdf5-path      Path to latent HDF5 file            (default: data/preprocessed/emg_latent_ae_v2.hdf5)
    --resume         Path to a checkpoint .pt to continue training
    --from-hyperparams  YAML file of hyperparameters (from hyperparam_tuner_latent.py)
    --early-stopping-patience  Stop if val CER doesn't improve for N epochs (0 = disabled)

CNN+LSTM architecture flags:
    --proj-features  Linear projection output size       (default: 384)
    --cnn-channels   Channels in each 1D CNN block       (default: 256)
    --cnn-kernel     Kernel size for 1D convolutions     (default: 3)
    --cnn-layers     Number of 1D CNN blocks             (default: 2)
    --lstm-hidden    BiLSTM hidden units per direction   (default: 256)
    --lstm-layers    Number of stacked BiLSTM layers     (default: 2)
    --dropout        Dropout probability                 (default: 0.3)

The best checkpoint (lowest val CER) is saved to:
    Playground_Mumbi/checkpoints/best_latent_cnn_lstm.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import yaml

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData
from emg2qwerty.decoder import CTCGreedyDecoder
from emg2qwerty.metrics import CharacterErrorRates

from Playground_Mumbi.data_utils import get_latent_dataloaders
from Playground_Mumbi.model_latent import LatentCNNLSTMModel
from scripts.logger import log_epoch, log_summary, make_run_id

# Fixed latent dimension — defined by the autoencoder
LATENT_DIM: int = 1024


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup + cosine annealing (step-level)
# ---------------------------------------------------------------------------

def _lr_lambda(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
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
        inputs = batch["inputs"].to(device)           # (T, N, 1024)
        targets = batch["targets"].to(device)         # (T_tgt, N)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad()

        emissions = model(inputs)   # (T, N, num_classes)

        loss = criterion(
            emissions,
            targets.transpose(0, 1),
            input_lengths,
            target_lengths,
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
    """Evaluate the model; returns (mean_loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    criterion = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    cer_metric = CharacterErrorRates()

    for batch in loader:
        inputs = batch["inputs"].to(device)           # (T, N, 1024)
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        with torch.backends.cudnn.flags(enabled=False):
            emissions = model(inputs)                 # (T, N, num_classes)

        loss = criterion(
            emissions,
            targets.transpose(0, 1).to(device),
            input_lengths.to(device),
            target_lengths.to(device),
        )
        total_loss += loss.item()

        preds = decoder.decode_batch(
            emissions=emissions.cpu().numpy(),
            emission_lengths=input_lengths.numpy(),
        )

        targets_np = targets.numpy()
        tgt_lens = target_lengths.numpy()
        for i, pred in enumerate(preds):
            target = LabelData.from_labels(targets_np[: tgt_lens[i], i])
            cer_metric.update(prediction=pred, target=target)

    metrics = cer_metric.compute()
    return total_loss / max(len(loader), 1), metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--from-hyperparams", type=Path, default=None)
    _pre_args, _ = _pre.parse_known_args()

    _hp: dict = {}
    if _pre_args.from_hyperparams is not None:
        with open(_pre_args.from_hyperparams) as _f:
            _hp = yaml.safe_load(_f) or {}
        print(f"Loaded hyperparameters from {_pre_args.from_hyperparams}")
        _hp_display = ", ".join(
            f"{k}={v}" for k, v in _hp.items()
            if k in {"lr", "proj_features", "cnn_channels", "cnn_kernel", "cnn_layers",
                     "lstm_hidden", "lstm_layers", "dropout", "weight_decay"}
        )
        print(f"  {_hp_display}")

    p = argparse.ArgumentParser(
        description="Train the latent-space CNN+LSTM hybrid model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--hdf5-path", type=Path,
                   default=_ROOT / "data" / "preprocessed" / "emg_latent_ae_v2.hdf5",
                   help="Path to the latent EMG HDF5 file")
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path(__file__).resolve().parent / "checkpoints",
                   help="Directory to write model checkpoints")
    # Training
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (0 = main process, safest on Windows)")
    p.add_argument("--window-length", type=int, default=125,
                   help="Latent frames per training window (125 ≈ 4 s at 32 ms/frame)")
    # Model architecture
    p.add_argument("--proj-features", type=int, default=_hp.get("proj_features", 384),
                   help="Linear projection output size before CNN")
    p.add_argument("--cnn-channels", type=int, default=_hp.get("cnn_channels", 256),
                   help="Channels in each 1D CNN block")
    p.add_argument("--cnn-kernel", type=int, default=_hp.get("cnn_kernel", 3),
                   help="Kernel size for 1D CNN blocks")
    p.add_argument("--cnn-layers", type=int, default=_hp.get("cnn_layers", 2),
                   help="Number of 1D CNN blocks")
    p.add_argument("--lstm-hidden", type=int, default=_hp.get("lstm_hidden", 256),
                   help="BiLSTM hidden units per direction")
    p.add_argument("--lstm-layers", type=int, default=_hp.get("lstm_layers", 2),
                   help="Number of stacked BiLSTM layers")
    p.add_argument("--dropout", type=float, default=_hp.get("dropout", 0.3),
                   help="Dropout probability")
    # Optimiser / scheduler
    p.add_argument("--lr", type=float, default=_hp.get("lr", 5e-4))
    p.add_argument("--weight-decay", type=float, default=_hp.get("weight_decay", 1e-2))
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=10)
    # Resume / test-only / hyperparams
    p.add_argument("--resume", type=Path, default=None,
                   help="Checkpoint .pt file to resume training from")
    p.add_argument("--test-only", action="store_true",
                   help="Skip training: load best checkpoint and run test evaluation only")
    p.add_argument("--from-hyperparams", type=Path, default=None,
                   help="YAML file of hyperparameters (from hyperparam_tuner_latent.py)")
    p.add_argument("--early-stopping-patience", type=int, default=0,
                   help="Stop training if val CER does not improve for this many epochs. 0 = disabled.")
    p.add_argument("--notes", type=str, default="",
                   help="Free-text annotation written to the CSV log")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _pipeline_label = "latent (emg_latent_ae_v2, 1024-dim @ 32ms/frame)"
    _sampling_rate  = 31.25   # Hz equivalent (1 frame per 32 ms)
    _input_type     = "latent"
    _num_channels   = 32      # 2 wrists × 16 channels (informational only)
    ckpt_stem       = "best_latent_cnn_lstm"

    print(f"Pipeline    : {_pipeline_label}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Building data loaders …")
    loaders = get_latent_dataloaders(
        hdf5_path=args.hdf5_path,
        window_length=args.window_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if args.test_only:
        print(f"  test batches  : {len(loaders['test'])}")
    else:
        print(
            f"  train batches : {len(loaders['train'])}"
            f" | val batches : {len(loaders['val'])}"
            f" | test batches : {len(loaders['test'])}"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _build_model(hparams: dict) -> nn.Module:
        effective_dropout = (
            float(hparams.get("dropout", args.dropout))
            if int(hparams.get("lstm_layers", args.lstm_layers)) > 1
            else 0.0
        )
        return LatentCNNLSTMModel(
            latent_dim=LATENT_DIM,
            proj_features=int(hparams.get("proj_features", args.proj_features)),
            cnn_channels=int(hparams.get("cnn_channels", args.cnn_channels)),
            cnn_kernel=int(hparams.get("cnn_kernel", args.cnn_kernel)),
            cnn_layers=int(hparams.get("cnn_layers", args.cnn_layers)),
            lstm_hidden=int(hparams.get("lstm_hidden", args.lstm_hidden)),
            lstm_layers=int(hparams.get("lstm_layers", args.lstm_layers)),
            dropout=effective_dropout,
            num_classes=charset().num_classes,
        ).to(device)

    model = _build_model(vars(args))
    print(f"Model       : CNN+LSTM (latent)")
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Test-only shortcut
    # ------------------------------------------------------------------
    if args.test_only:
        best_ckpt = args.checkpoint_dir / f"{ckpt_stem}.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {best_ckpt}. "
                "Train first or pass --checkpoint-dir pointing at an existing checkpoint."
            )
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        saved_args = ckpt.get("args", {})
        model = _build_model(saved_args)
        print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,} (from checkpoint args)")

        model.load_state_dict(ckpt["model"])
        saved_epoch = ckpt.get("epoch", "?")
        saved_cer   = ckpt.get("best_cer", float("nan"))
        print(f"Loaded checkpoint — epoch {saved_epoch + 1 if isinstance(saved_epoch, int) else saved_epoch}  val CER={saved_cer:.2f}%")
        print("\n--- Test Evaluation ---")
        _, test_metrics = evaluate(model, loaders["test"], device, CTCGreedyDecoder())
        print("Test metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.2f}%")

        _n_train  = len(loaders["train"].dataset)
        _n_total  = _n_train + len(loaders["val"].dataset) + len(loaders["test"].dataset)
        _train_fraction = _n_train / _n_total
        _run_id = make_run_id(
            model="CNN_LSTM",
            num_channels=_num_channels,
            sampling_rate_hz=_sampling_rate,
            train_fraction=_train_fraction,
        )
        log_summary(
            _run_id,
            model="CNN_LSTM",
            epochs=saved_epoch + 1 if isinstance(saved_epoch, int) else 0,
            num_channels=_num_channels,
            sampling_rate_hz=_sampling_rate,
            train_fraction=_train_fraction,
            input_type=_input_type,
            final_train_loss=float("nan"),
            final_val_loss=float("nan"),
            final_val_cer=saved_cer,
            test_cer=test_metrics.get("CER", float("nan")),
            training_time_sec=0.0,
            notes=" ".join(filter(None, [args.notes, "test_only", "latent"])),
        )
        return

    # ------------------------------------------------------------------
    # Loss / optimiser / LR schedule
    # ------------------------------------------------------------------
    criterion = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    patience_counter = 0

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
    if args.early_stopping_patience > 0:
        print(f"Early stopping : patience={args.early_stopping_patience} epochs")
    print("Tip: press Ctrl+C at any time to stop early and jump to test evaluation.\n")

    epoch_duration: float = 0.0
    train_start_time = time.perf_counter()

    # Logger setup
    _n_train = len(loaders["train"].dataset)
    _n_total = _n_train + len(loaders["val"].dataset) + len(loaders["test"].dataset)
    _train_fraction = _n_train / _n_total
    run_id = make_run_id(
        model="CNN_LSTM",
        num_channels=_num_channels,
        sampling_rate_hz=_sampling_rate,
        train_fraction=_train_fraction,
    )
    _final_train_loss: float = float("nan")
    _final_val_loss:   float = float("nan")
    _final_val_cer:    float = float("nan")

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

        log_epoch(run_id, model="CNN_LSTM", epoch=epoch + 1,
                  train_loss=train_loss, val_loss=val_loss, val_cer=val_cer)
        _final_train_loss, _final_val_loss, _final_val_cer = train_loss, val_loss, val_cer

        if val_cer < best_cer:
            best_cer = val_cer
            patience_counter = 0
            ckpt_path = args.checkpoint_dir / f"{ckpt_stem}.pt"
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
        else:
            patience_counter += 1
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                print(f"\n[Early stopping] No improvement for {patience_counter} epochs. Stopping.")
                break

    # ------------------------------------------------------------------
    # Test evaluation using the best checkpoint
    # ------------------------------------------------------------------
    print("\n--- Test Evaluation ---")
    best_ckpt = args.checkpoint_dir / f"{ckpt_stem}.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch'] + 1}")

    total_training_time = time.perf_counter() - train_start_time
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_training_time))

    _, test_metrics = evaluate(model, loaders["test"], device, decoder)
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.2f}%")

    print(f"\nTotal training time: {total_time_str} ({total_training_time:.1f}s)")

    epochs_completed = epoch + 1 - start_epoch
    log_summary(
        run_id,
        model="CNN_LSTM",
        epochs=epochs_completed,
        num_channels=_num_channels,
        sampling_rate_hz=_sampling_rate,
        train_fraction=_train_fraction,
        input_type=_input_type,
        final_train_loss=_final_train_loss,
        final_val_loss=_final_val_loss,
        final_val_cer=_final_val_cer,
        test_cer=test_metrics.get("CER", float("nan")),
        training_time_sec=total_training_time,
        notes=" ".join(filter(None, ["latent", args.notes])),
    )

    # ------------------------------------------------------------------
    # Append structured report to log file
    # ------------------------------------------------------------------
    log_path = args.checkpoint_dir / "log_model_training.txt"
    run_timestamp = time.strftime("%Y%m%d_%H%Mhrs")

    with open(log_path, "a") as log_f:
        log_f.write("\n################### New Entry ###################\n")
        log_f.write(f"Run timestamp   : {run_timestamp}\n")
        log_f.write(f"Model           : CNN+LSTM (latent)\n")
        log_f.write(f"Pipeline        : {_pipeline_label}\n")
        log_f.write(f"HDF5            : {args.hdf5_path}\n")
        log_f.write(f"Device          : {device}\n")
        log_f.write(f"Parameters      : {sum(p.numel() for p in model.parameters()):,}\n")
        _stopped_early = args.early_stopping_patience > 0 and epochs_completed < (args.epochs - start_epoch)
        log_f.write(f"Epochs planned  : {args.epochs}  |  Epochs completed: {epochs_completed}"
                    + ("  [early stopped]\n" if _stopped_early else "\n"))
        log_f.write(f"Total train time: {total_time_str} ({total_training_time:.1f}s)\n")
        log_f.write("--- Hyperparameters ---\n")
        log_f.write(f"  lr={args.lr}\n")
        log_f.write(f"  weight_decay={args.weight_decay}\n")
        log_f.write(f"  proj_features={args.proj_features}\n")
        log_f.write(f"  cnn_channels={args.cnn_channels}\n")
        log_f.write(f"  cnn_kernel={args.cnn_kernel}\n")
        log_f.write(f"  cnn_layers={args.cnn_layers}\n")
        log_f.write(f"  lstm_hidden={args.lstm_hidden}\n")
        log_f.write(f"  lstm_layers={args.lstm_layers}\n")
        log_f.write(f"  dropout={args.dropout}\n")
        log_f.write(f"  batch_size={args.batch_size}\n")
        log_f.write(f"  window_length={args.window_length} latent frames\n")
        log_f.write("--- Results ---\n")
        log_f.write(f"  Best val CER    : {best_cer:.2f}%\n")
        for k, v in test_metrics.items():
            log_f.write(f"  Test {k:<12}: {v:.2f}%\n")
        log_f.write("#################################################\n")

    print(f"Log appended to {log_path}")


if __name__ == "__main__":
    main()
