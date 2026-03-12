"""Training and evaluation script for the TDS CNN baseline and CNN+LSTM hybrid models.

Run from the workspace root with:
    python -m Playground_Mumbi.train [options]
    python Playground_Mumbi/train.py [options]

Key flags (all have defaults, so a bare invocation will start training):
    --model             Which model to train: cnn (default) or cnn_lstm
    --epochs            Number of training epochs          (default: 150)
    --batch-size        Batch size                         (default: 32)
    --lr                Peak learning rate                 (default: 1e-3)
    --train-fraction    Fraction of training windows to use (default: 1.0)
    --run-all-fractions Run sequentially for fractions [0.10, 0.25, 0.50, 0.75, 1.00]
                        and log each run with Playground_Mumbi.logger
    --resume            Path to a checkpoint .pt to continue training
    --num-workers       DataLoader workers (default: 0, safe on Windows)

CNN baseline hyperparameters match config/model/tds_conv_ctc.yaml exactly:
    in_features=528, mlp_features=[384], block_channels=[24,24,24,24],
    kernel_width=32

CNN+LSTM architecture flags (only used when --model cnn_lstm):
    --cnn-channels      Number of channels in 1D CNN blocks   (default: 256)
    --cnn-kernel        Kernel size for 1D convolutions        (default: 5)
    --cnn-layers        Number of CNN blocks                   (default: 2)
    --lstm-hidden       BiLSTM hidden units per direction      (default: 256)
    --lstm-layers       Number of stacked BiLSTM layers        (default: 2)
    --dropout           Dropout probability                    (default: 0.3)

The best checkpoint (lowest val CER) is saved to:
    Playground_Mumbi/checkpoints/final_models/best_cnn.pt           (CNN, --train-fraction 1.0)
    Playground_Mumbi/checkpoints/training_fraction_ablation/best_cnn_{pct}pct.pt  (CNN, otherwise)
    Playground_Mumbi/checkpoints/final_models/best_cnn_lstm.pt      (CNN+LSTM)
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

from Playground_Kai.data_preprocess import (
    IN_FEATURES as PREP_IN_FEATURES,
    N_ELECTRODE_CHANNELS as PREP_N_CHANNELS,
)

from Playground_Mumbi.data_utils import get_dataloaders, get_dataloaders_biophys
from Playground_Mumbi.model import CNNLSTMModel
from scripts.logger import log_epoch, log_summary, make_run_id


# ---------------------------------------------------------------------------
# TDS model constants (from config/model/tds_conv_ctc.yaml)
# ---------------------------------------------------------------------------

_NUM_BANDS: int = 2
_ELECTRODE_CHANNELS: int = 16
_FREQ_BINS: int = 33              # n_fft // 2 + 1 = 64 // 2 + 1
_IN_FEATURES: int = 528           # _FREQ_BINS * _ELECTRODE_CHANNELS = 33 * 16
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


def build_cnn_lstm_model(args: argparse.Namespace) -> nn.Module:
    """Build the CNN+LSTM hybrid model from CLI args.

    Architecture:
        SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten
        → 1D CNN blocks (local temporal feature extraction)
        → BiLSTM (long-range sequential modelling)
        → Linear → LogSoftmax

    The CNN front-end shares the same SpectrogramNorm + MLP as the TDS baseline
    for a fair comparison.  The BiLSTM is bidirectional because we process
    fixed-length windows, so future context within the window is available.

    Args:
        args: Parsed CLI arguments.  The following fields are used:
            num_channels, cnn_channels, cnn_kernel, cnn_layers,
            lstm_hidden, lstm_layers, dropout.

    Returns:
        A :class:`CNNLSTMModel` instance.
    """
    if getattr(args, "biophys", False):
        in_features = PREP_IN_FEATURES        # 256 = 8 channels × 32 Mel bins
        electrode_channels = PREP_N_CHANNELS  # 8
    else:
        in_features = args.num_channels * _FREQ_BINS
        electrode_channels = args.num_channels
    return CNNLSTMModel(
        in_features=in_features,
        mlp_features=_MLP_FEATURES,
        num_bands=_NUM_BANDS,
        electrode_channels=electrode_channels,
        cnn_channels=args.cnn_channels,
        cnn_kernel=args.cnn_kernel,
        cnn_layers=args.cnn_layers,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        num_classes=charset().num_classes,
    )




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

def run_training(args: argparse.Namespace, train_fraction: float, notes: str,
                 model_name: str = "CNN", model: nn.Module | None = None) -> None:
    """Execute one complete train + test cycle for a given train_fraction.

    Args:
        args:           Parsed CLI arguments.
        train_fraction: Fraction of training windows to use (0.0, 1.0].
        notes:          Value written to the ``notes`` field in the run summary.
        model_name:     Model identifier used in logging (e.g. "CNN", "CNN_LSTM").
        model:          Model instance to train.  If None, builds the TDS CNN baseline.
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
    _recons_v3 = getattr(args, "recons_v3", False)
    _data_subdir = "89335547_recons_v3" if _recons_v3 else "89335547"
    _file_suffix = "_recons_v3" if _recons_v3 else ""
    if getattr(args, "biophys", False):
        loaders = get_dataloaders_biophys(
            data_root=args.data_root,
            config_path=args.config,
            window_length=args.window_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            test_window_length=args.window_length,
            train_fraction=train_fraction,
            data_subdir=_data_subdir,
            file_suffix=_file_suffix,
        )
    else:
        loaders = get_dataloaders(
            data_root=args.data_root,
            config_path=args.config,
            window_length=args.window_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            test_window_length=args.window_length,
            train_fraction=train_fraction,
            channel_indices=args.channel_indices,
            data_subdir=_data_subdir,
            file_suffix=_file_suffix,
        )
    print(
        f"  train batches : {len(loaders['train'])}"
        f" | val batches : {len(loaders['val'])}"
        f" | test batches : {len(loaders['test'])}"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if model is None:
        model = build_model()
    model = model.to(device)
    print(f"Model       : {model_name}")
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Logger run ID
    # ------------------------------------------------------------------
    _eff_channels = PREP_N_CHANNELS if getattr(args, "biophys", False) else args.num_channels
    _eff_hz = 1000 if getattr(args, "biophys", False) else 2000
    run_id = make_run_id(
        model=model_name,
        num_channels=_eff_channels,
        sampling_rate_hz=_eff_hz,
        train_fraction=train_fraction,
    )
    print(f"Run ID      : {run_id}")

    # ------------------------------------------------------------------
    # Loss / optimiser / LR schedule
    # ------------------------------------------------------------------
    # CTCLoss receives log-probabilities (LogSoftmax is part of the model).
    # zero_infinity=True prevents NaN loss for very short windows.
    criterion = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)

    # CNN baseline uses Adam (matches original TDS training setup).
    # CNN+LSTM uses AdamW — decouples weight decay from the adaptive LR update,
    # so the tuned weight_decay hyperparameter behaves as intended.
    if model_name == "CNN_LSTM":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
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
            model=model_name,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            val_cer=val_cer,
        )

        # Save checkpoint whenever val CER improves
        if val_cer < best_cer:
            best_cer = val_cer
            fraction_pct = int(round(train_fraction * 100))
            model_slug = model_name.lower().replace("_", "")  # "cnn" or "cnnlstm"
            if getattr(args, "biophys", False) and model_slug == "cnnlstm":
                model_slug = "cnnlstm_biophys"
            if train_fraction < 1.0:
                ckpt_dir = args.checkpoint_dir / "training_fraction_ablation"
                ckpt_path = ckpt_dir / f"best_{model_slug}_{fraction_pct}pct.pt"
            elif args.num_channels < 16:
                ckpt_dir = args.checkpoint_dir / "channel_ablation"
                ckpt_path = ckpt_dir / f"best_{model_slug}_{args.num_channels}ch.pt"
            else:
                ckpt_dir = args.checkpoint_dir / "final_models"
                ckpt_path = ckpt_dir / f"best_{model_slug}.pt"
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
    model_slug = model_name.lower().replace("_", "")
    if getattr(args, "biophys", False) and model_slug == "cnnlstm":
        model_slug = "cnnlstm_biophys"
    if train_fraction < 1.0:
        best_ckpt = args.checkpoint_dir / "training_fraction_ablation" / f"best_{model_slug}_{fraction_pct}pct.pt"
    elif args.num_channels < 16:
        best_ckpt = args.checkpoint_dir / "channel_ablation" / f"best_{model_slug}_{args.num_channels}ch.pt"
    else:
        best_ckpt = args.checkpoint_dir / "final_models" / f"best_{model_slug}.pt"
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
        model=model_name,
        epochs=epochs_completed,
        num_channels=_eff_channels,
        sampling_rate_hz=_eff_hz,
        train_fraction=train_fraction,
        input_type="biophys" if getattr(args, "biophys", False) else "spectrogram",
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
    # Model selection
    p.add_argument("--model", type=str, default="cnn",
                   choices=["cnn", "cnn_lstm"],
                   help="Which model to train (default: cnn)")
    p.add_argument("--biophys", action="store_true",
                   help="Use biophysics preprocessing pipeline (notch+bandpass+decimate+Mel). "
                        "Only applies to --model cnn_lstm.")
    p.add_argument("--recons-v3", action="store_true",
                   help="Use AE-reconstructed v3 data (data/89335547_recons_v3/).")
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
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (0 = main process, safest on Windows)")
    p.add_argument("--window-length", type=int, default=8000,
                   help="Raw EMG samples per training/test window (8000 = 4 s @ 2 kHz)")
    # Optimiser / scheduler
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Peak (post-warmup) learning rate for Adam / AdamW")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Weight decay for AdamW (CNN+LSTM only, ignored for CNN)")
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
    # Channel ablation flags
    p.add_argument("--num-channels", type=int, default=16,
                   help="Electrode channels per hand after selection (default: 16 = all)")
    p.add_argument("--channel-indices", type=int, nargs="+", default=None,
                   help="Electrode indices to keep per hand (e.g. 0 2 4 6 8 10 12 14). "
                        "When omitted all 16 channels are used.")
    p.add_argument("--notes", type=str, default=None,
                   help="Override the notes field written to the results CSV "
                        "(e.g. ablation_channels). Computed automatically when omitted.")
    # CNN+LSTM architecture flags (only used when --model cnn_lstm)
    p.add_argument("--from-hyperparams", type=Path, default=None,
                   help="Path to a YAML file saved by hyperparam_tuner.py. "
                        "Overrides --lr, --weight-decay, --cnn-channels, --cnn-kernel, "
                        "--cnn-layers, --lstm-hidden, --lstm-layers, and --dropout "
                        "with the tuned values. Any of those flags can still be "
                        "specified alongside to override individual fields.")
    p.add_argument("--cnn-channels", type=int, default=256,
                   help="Number of channels in each 1D CNN block")
    p.add_argument("--cnn-kernel", type=int, default=5,
                   help="Kernel size for 1D CNN blocks")
    p.add_argument("--cnn-layers", type=int, default=2,
                   help="Number of 1D CNN blocks")
    p.add_argument("--lstm-hidden", type=int, default=256,
                   help="BiLSTM hidden units per direction")
    p.add_argument("--lstm-layers", type=int, default=2,
                   help="Number of stacked BiLSTM layers")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="Dropout probability (used in CNN+LSTM only)")
    # Resume
    p.add_argument("--resume", type=Path, default=None,
                   help="Checkpoint .pt file to resume training from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_channel_indices(args: argparse.Namespace) -> None:
    """Auto-compute channel_indices from num_channels if not explicitly provided.

    Uses the same even-stride pattern as Ben's channel ablation YAML configs:
        16 ch/hand → None (all channels, no ChannelSelect applied)
         8 ch/hand → [0, 2, 4, 6, 8, 10, 12, 14]   (stride 2)
         4 ch/hand → [0, 4, 8, 12]                  (stride 4)
         2 ch/hand → [0, 8]                          (stride 8)
    """
    if args.channel_indices is not None or args.num_channels == 16:
        return
    stride = _ELECTRODE_CHANNELS // args.num_channels
    args.channel_indices = list(range(0, _ELECTRODE_CHANNELS, stride))


def main() -> None:
    args = parse_args()
    _resolve_channel_indices(args)

    # If --from-hyperparams is given, load the tuner YAML and patch args.
    # Explicit CLI flags always win — we only overwrite fields that are still
    # at their argparse defaults, so the user can mix-and-match freely.
    if args.from_hyperparams is not None:
        import yaml
        with open(args.from_hyperparams) as f:
            hp = yaml.safe_load(f)
        # Map YAML keys → args attribute names
        _HP_FIELDS = {
            "lr":           "lr",
            "weight_decay": "weight_decay",
            "cnn_channels": "cnn_channels",
            "cnn_kernel":   "cnn_kernel",
            "cnn_layers":   "cnn_layers",
            "lstm_hidden":  "lstm_hidden",
            "lstm_layers":  "lstm_layers",
            "dropout":      "dropout",
        }
        for yaml_key, arg_attr in _HP_FIELDS.items():
            if yaml_key in hp:
                setattr(args, arg_attr, hp[yaml_key])
        print(f"Loaded hyperparameters from: {args.from_hyperparams}")
        for yaml_key, arg_attr in _HP_FIELDS.items():
            if yaml_key in hp:
                print(f"  {arg_attr} = {getattr(args, arg_attr)}")

    if args.model == "cnn_lstm":
        # CNN+LSTM: single full training run (no fraction sweep)
        notes = args.notes or (
            ("arch_comparison_biophys" if args.biophys else "arch_comparison")
            if args.train_fraction == 1.0 else "ablation_train_fraction"
        )
        run_training(
            args,
            train_fraction=args.train_fraction,
            notes=notes,
            model_name="CNN_LSTM",
            model=build_cnn_lstm_model(args),
        )
    else:
        # TDS CNN baseline — optionally sweep all training fractions
        if args.run_all_fractions:
            print(f"--run-all-fractions: will train for fractions {_ALL_FRACTIONS}")
            for frac in _ALL_FRACTIONS:
                run_training(args, train_fraction=frac, notes="ablation_train_fraction",
                             model_name="CNN", model=build_model())
        else:
            notes = args.notes or (
                "arch_comparison" if args.train_fraction == 1.0 else "ablation_train_fraction"
            )
            run_training(args, train_fraction=args.train_fraction, notes=notes,
                         model_name="CNN", model=build_model())


if __name__ == "__main__":
    main()
