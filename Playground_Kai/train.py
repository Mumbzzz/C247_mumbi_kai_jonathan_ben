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
    Playground_Kai/checkpoints/best.pt
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
from Playground_Kai.model import RNNEncoder


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
            if k in {"lr", "hidden_size", "num_layers", "dropout", "weight_decay"}
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
    p.add_argument("--hidden-size", type=int, default=_hp.get("hidden_size", 512),
                   help="LSTM hidden size per direction")
    p.add_argument("--num-layers", type=int, default=_hp.get("num_layers", 2),
                   help="Number of stacked BiLSTM layers")
    p.add_argument("--dropout", type=float, default=_hp.get("dropout", 0.2),
                   help="Dropout between LSTM layers (ignored when num-layers=1)")
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
                   help="Skip training: load best.pt and run test evaluation only")
    p.add_argument("--from-hyperparams", type=Path, default=None,
                   help="YAML file of hyperparameters (e.g. from hyperparam_tuner.py)")
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
    model = RNNEncoder(
        in_features=528,        # (n_fft // 2 + 1) * electrode_channels = 33 * 16
        mlp_features=(384,),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Test-only shortcut: load checkpoint and evaluate, then exit
    # ------------------------------------------------------------------
    if args.test_only:
        best_ckpt = args.checkpoint_dir / "best.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {best_ckpt}. "
                "Train first or pass --checkpoint-dir pointing at an existing checkpoint."
            )
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        saved_epoch = ckpt.get("epoch", "?")  
        saved_cer   = ckpt.get("best_cer", float("nan"))
        print(f"Loaded checkpoint — epoch {saved_epoch + 1 if isinstance(saved_epoch, int) else saved_epoch}  val CER={saved_cer:.2f}%")
        print("\n--- Test Evaluation ---")
        _, test_metrics = evaluate(model, loaders["test"], device, CTCGreedyDecoder())
        print("Test metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.2f}%")
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

        # Save checkpoint whenever val CER improves
        if val_cer < best_cer:
            best_cer = val_cer
            ckpt_path = args.checkpoint_dir / "best.pt"
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
    best_ckpt = args.checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch'] + 1}")

    _, test_metrics = evaluate(model, loaders["test"], device, decoder)
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.2f}%")


if __name__ == "__main__":
    main()
