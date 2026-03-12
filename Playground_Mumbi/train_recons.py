"""Training script for CNN+LSTM on AE-reconstructed EMG v3 data.

Operates on ``_recons_v3.hdf5`` files (62.5 Hz, 32 channels = 16 left + 16 right,
plain float32 arrays).  Uses ``LatentCNNLSTMModel`` with ``latent_dim=32`` —
the linear projection layer accepts the 32-dim reconstructed frame directly.

Run from the workspace root:
    python -m Playground_Mumbi.train_recons [options]

Key flags:
    --epochs                  Number of training epochs          (default: 150)
    --batch-size              Batch size                         (default: 32)
    --window-length           Frames per window (250 ≈ 4 s @ 62.5 Hz)
    --data-dir                Directory containing _recons_v3.hdf5 files
    --recons-config           YAML defining train/val/test split
    --from-hyperparams        YAML from hyperparam_tuner_recons.py
    --early-stopping-patience Stop if val CER doesn't improve for N epochs

The best checkpoint is saved to:
    Playground_Mumbi/checkpoints/best_recons_cnn_lstm.pt
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

from Playground_Mumbi.data_utils import get_recons_dataloaders
from Playground_Mumbi.model_latent import LatentCNNLSTMModel
from scripts.logger import log_epoch, log_summary, make_run_id

# Fixed input dimensionality for reconstructed EMG: 16 left + 16 right channels
RECONS_DIM: int = 32


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def _lr_lambda(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        inputs         = batch["inputs"].to(device)           # (T, N, 32)
        targets        = batch["targets"].to(device)
        input_lengths  = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad()
        emissions = model(inputs)   # (T, N, num_classes) — no T shrinkage

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
def evaluate(model, loader, device, decoder) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    criterion  = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    cer_metric = CharacterErrorRates()

    for batch in loader:
        inputs         = batch["inputs"].to(device)
        targets        = batch["targets"]
        input_lengths  = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        with torch.backends.cudnn.flags(enabled=False):
            emissions = model(inputs)

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
        tgt_lens   = target_lengths.numpy()
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
        description="Train CNN+LSTM on AE-reconstructed EMG v3 data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data-dir", type=Path,
                   default=_ROOT / "data" / "89335547_recons_v3",
                   help="Directory containing _recons_v3.hdf5 session files")
    p.add_argument("--recons-config", type=Path,
                   default=_ROOT / "config" / "user" / "single_user.yaml",
                   help="YAML defining train/val/test split")
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path(__file__).resolve().parent / "checkpoints",
                   help="Directory to write model checkpoints")
    # Training
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--window-length", type=int, default=250,
                   help="Frames per window (250 ≈ 4 s @ 62.5 Hz)")
    # Model architecture
    p.add_argument("--proj-features", type=int, default=_hp.get("proj_features", 384))
    p.add_argument("--cnn-channels",  type=int, default=_hp.get("cnn_channels",  256))
    p.add_argument("--cnn-kernel",    type=int, default=_hp.get("cnn_kernel",    3))
    p.add_argument("--cnn-layers",    type=int, default=_hp.get("cnn_layers",    2))
    p.add_argument("--lstm-hidden",   type=int, default=_hp.get("lstm_hidden",   256))
    p.add_argument("--lstm-layers",   type=int, default=_hp.get("lstm_layers",   2))
    p.add_argument("--dropout",       type=float, default=_hp.get("dropout",     0.3))
    # Optimiser / scheduler
    p.add_argument("--lr",           type=float, default=_hp.get("lr",           5e-4))
    p.add_argument("--weight-decay", type=float, default=_hp.get("weight_decay", 1e-2))
    p.add_argument("--min-lr",       type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int,  default=10)
    # Misc
    p.add_argument("--resume",    type=Path, default=None)
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--from-hyperparams", type=Path, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--notes", type=str, default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Pipeline    : recons_v3 (32-dim @ 62.5 Hz)")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = "best_recons_cnn_lstm"

    # Data
    print("Building data loaders …")
    loaders = get_recons_dataloaders(
        data_dir=args.data_dir,
        config_path=args.recons_config,
        window_length=args.window_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"  train batches : {len(loaders['train'])}"
        f" | val batches : {len(loaders['val'])}"
        f" | test batches : {len(loaders['test'])}"
    )

    # Model
    def _build_model(hparams: dict) -> nn.Module:
        effective_dropout = (
            float(hparams.get("dropout", args.dropout))
            if int(hparams.get("lstm_layers", args.lstm_layers)) > 1 else 0.0
        )
        return LatentCNNLSTMModel(
            latent_dim=RECONS_DIM,
            proj_features=int(hparams.get("proj_features", args.proj_features)),
            cnn_channels=int(hparams.get("cnn_channels",  args.cnn_channels)),
            cnn_kernel=int(hparams.get("cnn_kernel",      args.cnn_kernel)),
            cnn_layers=int(hparams.get("cnn_layers",      args.cnn_layers)),
            lstm_hidden=int(hparams.get("lstm_hidden",    args.lstm_hidden)),
            lstm_layers=int(hparams.get("lstm_layers",    args.lstm_layers)),
            dropout=effective_dropout,
            num_classes=charset().num_classes,
        ).to(device)

    model = _build_model(vars(args))
    print(f"Model       : CNN+LSTM (recons_v3)")
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")

    # Logger setup
    _n_train     = len(loaders["train"].dataset)
    _n_total     = _n_train + len(loaders["val"].dataset) + len(loaders["test"].dataset)
    _train_frac  = _n_train / _n_total
    run_id = make_run_id("CNN_LSTM", num_channels=RECONS_DIM,
                         sampling_rate_hz=62.5, train_fraction=_train_frac)
    print(f"Run ID      : {run_id}")

    # Test-only shortcut
    if args.test_only:
        best_ckpt = args.checkpoint_dir / f"{ckpt_stem}.pt"
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model = _build_model(ckpt.get("args", {}))
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint — epoch {ckpt.get('epoch', '?')+1}  val CER={ckpt.get('best_cer', float('nan')):.2f}%")
        _, test_metrics = evaluate(model, loaders["test"], device, CTCGreedyDecoder())
        print("Test metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.2f}%")
        return

    # Loss / optimiser / scheduler
    criterion    = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(loaders["train"])
    total_steps  = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    min_lr_ratio = args.min_lr / args.lr
    scheduler    = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps, min_lr_ratio)
    )
    decoder = CTCGreedyDecoder()

    # Resume
    start_epoch     = 0
    best_cer        = float("inf")
    patience_counter = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_cer    = ckpt.get("best_cer", float("inf"))
        print(f"Resumed from epoch {start_epoch} | best val CER {best_cer:.2f}%")

    # Training loop
    print(f"\nTraining for {args.epochs - start_epoch} epochs …")
    if args.early_stopping_patience > 0:
        print(f"Early stopping : patience={args.early_stopping_patience} epochs")
    print("Tip: press Ctrl+C at any time to stop early.\n")

    train_start    = time.perf_counter()
    _final_train_loss = _final_val_loss = _final_val_cer = float("nan")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        try:
            train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device, scheduler)
            val_loss, val_metrics = evaluate(model, loaders["val"], device, decoder)
        except KeyboardInterrupt:
            print("\n[Interrupted] Jumping to test evaluation.")
            break

        val_cer       = val_metrics["CER"]
        epoch_dur     = time.perf_counter() - t0
        eta_str       = time.strftime("%H:%M:%S", time.gmtime(epoch_dur * (args.epochs - epoch - 1)))

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}"
            f" | train_loss={train_loss:.4f}"
            f" | val_loss={val_loss:.4f}"
            f" | val_CER={val_cer:.2f}%"
            f" | epoch_time={epoch_dur:.1f}s"
            f" | ETA={eta_str}"
        )

        log_epoch(run_id, model="CNN_LSTM", epoch=epoch+1,
                  train_loss=train_loss, val_loss=val_loss, val_cer=val_cer)
        _final_train_loss, _final_val_loss, _final_val_cer = train_loss, val_loss, val_cer

        if val_cer < best_cer:
            best_cer = val_cer
            patience_counter = 0
            ckpt_path = args.checkpoint_dir / f"{ckpt_stem}.pt"
            safe_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "best_cer": best_cer, "args": safe_args}, ckpt_path)
            print(f"  -> Saved best checkpoint  (val CER={best_cer:.2f}%)")
        else:
            patience_counter += 1
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                print(f"\n[Early stopping] No improvement for {patience_counter} epochs.")
                break

    # Test evaluation
    print("\n--- Test Evaluation ---")
    best_ckpt = args.checkpoint_dir / f"{ckpt_stem}.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch']+1}")

    total_time     = time.perf_counter() - train_start
    _, test_metrics = evaluate(model, loaders["test"], device, decoder)
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.2f}%")
    print(f"\nTotal training time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    epochs_completed = epoch + 1 - start_epoch
    log_summary(
        run_id, model="CNN_LSTM",
        epochs=epochs_completed,
        num_channels=RECONS_DIM,
        sampling_rate_hz=62.5,
        train_fraction=_train_frac,
        input_type="reconstructed_emg_v3",
        final_train_loss=_final_train_loss,
        final_val_loss=_final_val_loss,
        final_val_cer=_final_val_cer,
        test_cer=test_metrics.get("CER", float("nan")),
        training_time_sec=total_time,
        notes=" ".join(filter(None, ["recons_v3", args.notes])),
    )

    # Append to log file
    log_path = args.checkpoint_dir / "log_model_training.txt"
    with open(log_path, "a") as log_f:
        log_f.write("\n################### New Entry ###################\n")
        log_f.write(f"Run timestamp   : {time.strftime('%Y%m%d_%H%Mhrs')}\n")
        log_f.write(f"Model           : CNN+LSTM (recons_v3)\n")
        log_f.write(f"Pipeline        : recons_v3 (32-dim @ 62.5 Hz)\n")
        log_f.write(f"Data dir        : {args.data_dir}\n")
        log_f.write(f"Device          : {device}\n")
        log_f.write(f"Parameters      : {sum(p.numel() for p in model.parameters()):,}\n")
        _stopped_early = args.early_stopping_patience > 0 and epochs_completed < (args.epochs - start_epoch)
        log_f.write(f"Epochs planned  : {args.epochs}  |  Epochs completed: {epochs_completed}"
                    + ("  [early stopped]\n" if _stopped_early else "\n"))
        log_f.write(f"Total train time: {time.strftime('%H:%M:%S', time.gmtime(total_time))} ({total_time:.1f}s)\n")
        log_f.write("--- Hyperparameters ---\n")
        for k in ("lr", "weight_decay", "proj_features", "cnn_channels", "cnn_kernel",
                  "cnn_layers", "lstm_hidden", "lstm_layers", "dropout", "batch_size", "window_length"):
            log_f.write(f"  {k}={getattr(args, k, 'n/a')}\n")
        log_f.write("--- Results ---\n")
        log_f.write(f"  Best val CER    : {best_cer:.2f}%\n")
        for k, v in test_metrics.items():
            log_f.write(f"  Test {k:<12}: {v:.2f}%\n")
        log_f.write("#################################################\n")
    print(f"Log appended to {log_path}")


if __name__ == "__main__":
    main()
