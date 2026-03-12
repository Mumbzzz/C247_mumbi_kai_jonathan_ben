"""CNN baseline training with the full biophysics EMG preprocessing pipeline.

Pipeline (implemented in Playground_Kai/data_preprocess.py):
    ToTensor → TemporalAlignmentJitter(120) → ForEach(RandomBandRotation)
    → ChannelSelector(8 even-indexed channels per wrist)
    → TemporalFilter (zero-phase 60 Hz notch + 4th-order Butterworth 20–450 Hz)
    → Decimator (2× → 1000 Hz)
    → MelSpectrogramTransform (n_fft=256, win=64, hop=8, 32-bin Mel 20–450 Hz, log10)
    → SpecAugment (train only)

Model: SpectrogramNorm → MultiBandRotationInvariantMLP → TDSConvEncoder → CTC

    in_features = 8 channels/wrist × 32 Mel bins = 256
    num_features (TDS operating dim) = 2 bands × mlp_features

Results are appended to the shared results/results_summary_CNN.csv and
results/results_curves_CNN.csv — existing rows are never overwritten.

Run from repo root:
    python Playground_Ben/scripts/train_biophysics_cnn.py
    python Playground_Ben/scripts/train_biophysics_cnn.py --epochs 150 --lr 1e-3
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[2]
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

from Playground_Kai.data_utils import get_dataloaders
from Playground_Kai.data_preprocess import (
    IN_FEATURES,
    N_ELECTRODE_CHANNELS,
    N_MELS,
)

from scripts.logger import log_epoch, log_summary, make_run_id

# ---------------------------------------------------------------------------
# Constants (fixed by the biophysics pipeline)
# ---------------------------------------------------------------------------

NUM_BANDS: int = 2                          # left + right wrist
FREQ_BINS: int = N_MELS                     # 32 Mel bins
ELECTRODE_CHANNELS: int = N_ELECTRODE_CHANNELS  # 8 per band (even-indexed)
# in_features = 8 * 32 = 256  (imported from data_preprocess as IN_FEATURES)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BiophysicsTDSConvCTC(nn.Module):
    """TDSConvCTC for the biophysics Mel-spectrogram pipeline.

    Accepts batches of shape (T, N, 2, 8, 32) and outputs
    (T', N, num_classes) log-softmax activations.

    Architecture mirrors TDSConvCTCModule:
        SpectrogramNorm → MultiBandRotationInvariantMLP
        → Flatten → TDSConvEncoder → Linear → LogSoftmax
    """

    def __init__(
        self,
        in_features: int = IN_FEATURES,          # 256 = 8 ch × 32 Mel bins
        mlp_features: int = 384,                 # per-band MLP output dim
        block_channels: int = 24,
        num_blocks: int = 4,
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        num_features = NUM_BANDS * mlp_features  # 768
        # Snap to nearest multiple of block_channels (TDSConvEncoder requirement)
        num_features = max(block_channels,
                           round(num_features / block_channels) * block_channels)

        self.model = nn.Sequential(
            SpectrogramNorm(channels=NUM_BANDS * ELECTRODE_CHANNELS),  # 16
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=[mlp_features],
                num_bands=NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=[block_channels] * num_blocks,
                kernel_width=kernel_width,
            ),
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine annealing
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
        input_lengths  = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        optimizer.zero_grad()
        emissions = model(inputs)

        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = (input_lengths - T_diff).clamp(min=1)

        loss = criterion(
            emissions,
            targets.transpose(0, 1),
            emission_lengths.to(device),
            target_lengths.to(device),
        )
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
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train TDSConvCTC with biophysics EMG preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root",       type=Path,  default=_ROOT / "data")
    p.add_argument("--config",          type=Path,  default=_ROOT / "config" / "user" / "single_user.yaml")
    p.add_argument("--checkpoint-dir",  type=Path,  default=Path(__file__).resolve().parents[1] / "checkpoints")
    p.add_argument("--epochs",          type=int,   default=150)
    p.add_argument("--batch-size",      type=int,   default=32)
    p.add_argument("--num-workers",     type=int,   default=0)
    p.add_argument("--window-length",   type=int,   default=8000,
                   help="Raw EMG samples per window (4 s at 2 kHz)")
    p.add_argument("--stride",          type=int,   default=None,
                   help="Window stride in raw samples. Defaults to window-length (no overlap).")
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight-decay",    type=float, default=1e-5)
    p.add_argument("--warmup-frac",     type=float, default=0.05)
    p.add_argument("--mlp-features",    type=int,   default=384,
                   help="Per-band MLP output dimension; TDS operating dim = 2 × this")
    p.add_argument("--block-channels",  type=int,   default=24)
    p.add_argument("--num-blocks",      type=int,   default=4)
    p.add_argument("--kernel-width",    type=int,   default=32)
    p.add_argument("--resume",          type=Path,  default=None,
                   help="Checkpoint .pt to resume from")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:      {device}")
    print(f"Data root:   {args.data_root}")
    print(f"Config:      {args.config}")
    print(f"in_features: {IN_FEATURES}  ({ELECTRODE_CHANNELS} ch × {FREQ_BINS} Mel bins)")

    # ── Data ──────────────────────────────────────────────────────────────────
    # preprocess=True enables the full biophysics pipeline in data_utils.py
    loaders = get_dataloaders(
        data_root=args.data_root,
        config_path=args.config,
        window_length=args.window_length,
        stride=args.stride,
        padding=(1800, 200),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocess=True,
        test_window_length=args.window_length,
    )
    print(
        f"Dataset  — train: {len(loaders['train'].dataset)} windows"
        f" | val: {len(loaders['val'].dataset)} windows"
        f" | test: {len(loaders['test'].dataset)} windows"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BiophysicsTDSConvCTC(
        in_features=IN_FEATURES,
        mlp_features=args.mlp_features,
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

    # ── Optimiser / scheduler ─────────────────────────────────────────────────
    optimizer    = torch.optim.AdamW(model.parameters(),
                                     lr=args.lr, weight_decay=args.weight_decay)
    total_steps  = args.epochs * len(loaders["train"])
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps),
    )
    criterion = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    decoder   = CTCGreedyDecoder()

    # ── Checkpointing ─────────────────────────────────────────────────────────
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt    = args.checkpoint_dir / "best_biophysics_cnn.pt"
    best_val_cer = float("inf")

    # ── Logging ───────────────────────────────────────────────────────────────
    run_id  = make_run_id("CNN", num_channels=ELECTRODE_CHANNELS,
                          sampling_rate_hz=1000, train_fraction=1.0)
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

        val_cer    = val_metrics["CER"]
        epoch_sec  = time.time() - t_epoch

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_CER={val_cer:.2f}% | "
            f"{epoch_sec:.0f}s"
        )

        log_epoch(
            run_id=run_id,
            model="CNN",
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            val_cer=val_cer,
        )

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
        num_channels=ELECTRODE_CHANNELS,
        sampling_rate_hz=1000,
        train_fraction=1.0,
        input_type="mel_spectrogram_biophysics",
        final_train_loss=train_loss,
        final_val_loss=val_loss,
        final_val_cer=best_val_cer,
        test_cer=test_cer,
        training_time_sec=training_sec,
        notes="biophysics_pipeline",
    )
    print(f"\nDone. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
