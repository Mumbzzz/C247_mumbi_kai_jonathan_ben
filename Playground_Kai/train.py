"""Training and evaluation script for the RNN EMG-to-keystroke model.

Run from the workspace root with:
    python -m Playground_Kai.train [options]
    python Playground_Kai/train.py [options]

Key flags (all have defaults, so a bare invocation will start training):
    --epochs         Number of training epochs  (default: 80)
    --batch-size     Batch size                 (default: 32)
    --hidden-size    LSTM hidden size / dir      (default: 512)
    --num-layers     Stacked BiLSTM layers       (default: 2)
    --lr             Peak learning rate          (default: 5e-4)
    --resume         Path to a checkpoint .pt to continue training
    --num-workers    DataLoader workers          (default: 0, safe on Windows)

The best checkpoint (lowest val CER) is saved to
    Playground_Kai/checkpoints/best_{rnn,conformer}.pt
and automatically loaded for the final test evaluation.
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

# Ensure workspace root is on sys.path so both `emg2qwerty` and `Playground_Kai`
# are importable regardless of where the script is invoked from.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData
from emg2qwerty.decoder import CTCGreedyDecoder
from emg2qwerty.metrics import CharacterErrorRates

from Playground_Kai.data_utils import get_dataloaders
from Playground_Kai.model import RNNEncoder, ConformerEncoder
from scripts.logger import log_epoch, log_summary, make_run_id


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

        emissions = model(inputs)   # (T, N, num_classes)

        # The BiLSTM preserves temporal length, so emission_lengths == input_lengths.
        # CTCLoss expects targets as (N, T_tgt) — transpose from (T_tgt, N).
        loss = criterion(
            emissions,                      # (T, N, num_classes)
            targets.transpose(0, 1),        # (N, T_tgt)
            input_lengths,                  # (N,)
            target_lengths,                 # (N,)
        )

        loss.backward()
        # Gradient clipping guards against LSTM exploding gradients
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
        # cuDNN here falls back to PyTorch's generic CUDA LSTM which is correct
        # for any length.  Training windows are short enough that cuDNN is used
        # there without issue.
        with torch.backends.cudnn.flags(enabled=False):
            emissions = model(inputs)         # (T, N, num_classes)

        loss = criterion(
            emissions,
            targets.transpose(0, 1).to(device),
            input_lengths.to(device),
            target_lengths.to(device),
        )
        total_loss += loss.item()

        # Greedy CTC decode — decoder operates in numpy
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
    # Pre-parse --from-hyperparams so its values can seed the main parser defaults.
    # CLI flags always win over YAML values since argparse uses these only as defaults.
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
            if k in {"lr", "hidden_size", "num_layers", "dropout", "weight_decay",
                     "d_model", "num_heads", "conv_kernel_size"}
        )
        print(f"  {_hp_display}")

    p = argparse.ArgumentParser(
        description="Train the RNN EMG-to-keystroke model",
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
                   help="Raw EMG samples per training window (8000 = 4 s @ 2 kHz)")
    # Model — defaults overridden by --from-hyperparams if provided
    p.add_argument("--model", choices=["rnn", "conformer"], default="rnn",
                   help="Model architecture to train")
    p.add_argument("--hidden-size", type=int, default=_hp.get("hidden_size", 512),
                   help="(RNN) LSTM hidden size per direction")
    p.add_argument("--num-layers", type=int, default=_hp.get("num_layers", 2),
                   help="Stacked BiLSTM layers (rnn) or Conformer blocks (conformer)")
    p.add_argument("--dropout", type=float, default=_hp.get("dropout", 0.2),
                   help="Dropout probability")
    # Conformer-specific — only used when --model conformer
    p.add_argument("--d-model", type=int, default=_hp.get("d_model", 256),
                   help="(Conformer) feature dimension; must be divisible by --num-heads")
    p.add_argument("--num-heads", type=int, default=_hp.get("num_heads", 4),
                   help="(Conformer) self-attention heads; must evenly divide --d-model")
    p.add_argument("--conv-kernel-size", type=int, default=_hp.get("conv_kernel_size", 31),
                   help="(Conformer) depthwise conv kernel size; must be odd")
    # Optimiser / scheduler — defaults overridden by --from-hyperparams if provided
    p.add_argument("--lr", type=float, default=_hp.get("lr", 5e-4),
                   help="Peak (post-warmup) learning rate for AdamW")
    p.add_argument("--weight-decay", type=float, default=_hp.get("weight_decay", 1e-2),
                   help="AdamW weight decay")
    p.add_argument("--min-lr", type=float, default=1e-5,
                   help="Minimum learning rate at the end of cosine decay")
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="Epochs of linear LR warmup before cosine decay begins")
    # Resume / test-only / hyperparams
    p.add_argument("--resume", type=Path, default=None,
                   help="Checkpoint .pt file to resume training from")
    p.add_argument("--test-only", action="store_true",
                   help="Skip training: load best_{model}.pt and run test evaluation only")
    p.add_argument("--from-hyperparams", type=Path, default=None,
                   help="YAML file of hyperparameters (e.g. from hyperparam_tuner.py)")
    p.add_argument("--notes", type=str, default="",
                   help="Free-text annotation written to the CSV log (e.g. 'ablation_layers')")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")

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
        test_window_length=args.window_length if args.model == "conformer" else None,
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
    if args.model == "conformer":
        model = ConformerEncoder(
            in_features=528,
            mlp_features=(384,),
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            conv_kernel_size=args.conv_kernel_size,
            dropout=args.dropout,
        ).to(device)
    else:
        model = RNNEncoder(
            in_features=528,        # (n_fft // 2 + 1) * electrode_channels = 33 * 16
            mlp_features=(384,),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    print(f"Model       : {args.model}")
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Test-only shortcut: load checkpoint and evaluate, then exit
    # ------------------------------------------------------------------
    if args.test_only:
        best_ckpt = args.checkpoint_dir / f"best_{args.model}.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {best_ckpt}. "
                "Train first or pass --checkpoint-dir pointing at an existing checkpoint."
            )
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)

        # Rebuild the model from the hyperparams that were used during training,
        # not from CLI defaults.  The checkpoint stores args as a plain dict.
        saved_args = ckpt.get("args", {})
        if args.model == "conformer":
            model = ConformerEncoder(
                in_features=528,
                mlp_features=(384,),
                d_model=int(saved_args.get("d_model", args.d_model)),
                num_heads=int(saved_args.get("num_heads", args.num_heads)),
                num_layers=int(saved_args.get("num_layers", args.num_layers)),
                conv_kernel_size=int(saved_args.get("conv_kernel_size", args.conv_kernel_size)),
                dropout=float(saved_args.get("dropout", args.dropout)),
            ).to(device)
        else:
            model = RNNEncoder(
                in_features=528,
                mlp_features=(384,),
                hidden_size=int(saved_args.get("hidden_size", args.hidden_size)),
                num_layers=int(saved_args.get("num_layers", args.num_layers)),
                dropout=float(saved_args.get("dropout", args.dropout)),
            ).to(device)
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

        # Log summary to CSV even for test-only runs
        _logger_model = "CONFORMER" if args.model == "conformer" else "RNN"
        _n_train  = len(loaders["train"].dataset)
        _n_total  = _n_train + len(loaders["val"].dataset) + len(loaders["test"].dataset)
        _train_fraction = _n_train / _n_total
        _run_id = make_run_id(
            model=_logger_model,
            num_channels=32,
            sampling_rate_hz=2000,
            train_fraction=_train_fraction,
        )
        _notes = (args.notes + " " if args.notes else "") + "test_only"
        log_summary(
            _run_id,
            model=_logger_model,
            epochs=saved_epoch + 1 if isinstance(saved_epoch, int) else 0,
            num_channels=32,
            sampling_rate_hz=2000,
            train_fraction=_train_fraction,
            input_type="spectrogram",
            final_train_loss=float("nan"),
            final_val_loss=float("nan"),
            final_val_cer=saved_cer,
            test_cer=test_metrics.get("CER", float("nan")),
            training_time_sec=0.0,
            notes=_notes,
        )
        print(f"Test result logged to results/results_summary_{_logger_model}.csv")
        return

    # ------------------------------------------------------------------
    # Loss / optimiser / LR schedule
    # ------------------------------------------------------------------
    # zero_infinity=True prevents NaN loss when a target is longer than
    # the emission sequence (can happen for very short windows at the start).
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

    epoch_duration: float = 0.0  # updated each epoch for ETA estimation
    train_start_time = time.perf_counter()

    # --- Logger setup ---
    _logger_model = "CONFORMER" if args.model == "conformer" else "RNN"
    _n_train = len(loaders["train"].dataset)
    _n_total = _n_train + len(loaders["val"].dataset) + len(loaders["test"].dataset)
    _train_fraction = _n_train / _n_total
    run_id = make_run_id(
        model=_logger_model,
        num_channels=32,          # 16 electrodes × 2 wrists
        sampling_rate_hz=2000,
        train_fraction=_train_fraction,
    )
    # Sentinels — updated each epoch; used in log_summary even on early stop
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

        # Log epoch to CSV
        log_epoch(run_id, model=_logger_model, epoch=epoch + 1,
                  train_loss=train_loss, val_loss=val_loss, val_cer=val_cer)
        _final_train_loss, _final_val_loss, _final_val_cer = train_loss, val_loss, val_cer

        # Save checkpoint whenever val CER improves
        if val_cer < best_cer:
            best_cer = val_cer
            ckpt_path = args.checkpoint_dir / f"best_{args.model}.pt"
            # Convert Path objects to strings so torch.load(weights_only=True) works
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

    # ------------------------------------------------------------------
    # Test evaluation using the best checkpoint
    # ------------------------------------------------------------------
    print("\n--- Test Evaluation ---")
    best_ckpt = args.checkpoint_dir / f"best_{args.model}.pt"
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

    # --- Log summary to CSV ---
    log_summary(
        run_id,
        model=_logger_model,
        epochs=epochs_completed,
        num_channels=32,
        sampling_rate_hz=2000,
        train_fraction=_train_fraction,
        input_type="spectrogram",
        final_train_loss=_final_train_loss,
        final_val_loss=_final_val_loss,
        final_val_cer=_final_val_cer,
        test_cer=test_metrics.get("CER", float("nan")),
        training_time_sec=total_training_time,
        notes=args.notes,
    )

    # ------------------------------------------------------------------
    # Append structured report to log file
    # ------------------------------------------------------------------
    log_path = args.checkpoint_dir / "log_model_training.txt"
    run_timestamp = time.strftime("%Y%m%d_%H%Mhrs")
    epochs_completed = epoch + 1 - start_epoch  # epoch is still in scope from loop

    with open(log_path, "a") as log_f:
        log_f.write("\n################### New Entry ###################\n")
        log_f.write(f"Run timestamp   : {run_timestamp}\n")
        log_f.write(f"Model           : {args.model}\n")
        log_f.write(f"Device          : {device}\n")
        log_f.write(f"Parameters      : {sum(p.numel() for p in model.parameters()):,}\n")
        log_f.write(f"Epochs planned  : {args.epochs}  |  Epochs completed: {epochs_completed}\n")
        log_f.write(f"Total train time: {total_time_str} ({total_training_time:.1f}s)\n")
        log_f.write("--- Hyperparameters ---\n")
        log_f.write(f"  lr={args.lr}\n")
        log_f.write(f"  weight_decay={args.weight_decay}\n")
        if args.model == "conformer":
            log_f.write(f"  d_model={args.d_model}\n")
            log_f.write(f"  num_heads={args.num_heads}\n")
            log_f.write(f"  conv_kernel_size={args.conv_kernel_size}\n")
        else:
            log_f.write(f"  hidden_size={args.hidden_size}\n")
        log_f.write(f"  num_layers={args.num_layers}\n")
        log_f.write(f"  dropout={args.dropout}\n")
        log_f.write(f"  batch_size={args.batch_size}\n")
        log_f.write(f"  window_length={args.window_length}\n")
        log_f.write("--- Results ---\n")
        log_f.write(f"  Best val CER    : {best_cer:.2f}%\n")
        for k, v in test_metrics.items():
            log_f.write(f"  Test {k:<12}: {v:.2f}%\n")
        log_f.write("#################################################\n")

    print(f"Log appended to {log_path}")


if __name__ == "__main__":
    main()
