"""CNN baseline training on pre-computed AE latent vectors.

Uses emg_latent_ae_v2.hdf5 (27971 frames × 1024-dim float32, 32 ms/frame).
Data loading via Playground_Kai.data_utils.get_latent_dataloaders.

The front-end SpectrogramNorm + MultiBandRotationInvariantMLP is replaced by a
single nn.Linear(1024, num_features) projection; the TDSConvEncoder and CTC
decoder are kept identical to the raw-EMG baseline CNN.

Run from repo root:
    python Playground_Ben/scripts/train_latent_cnn.py
    python Playground_Ben/scripts/train_latent_cnn.py --epochs 150 --lr 3e-4
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import yaml
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
from emg2qwerty.modules import TDSConvEncoder

from Playground_Kai.data_utils import get_latent_dataloaders
from scripts.logger import log_epoch, log_summary, make_run_id

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

LATENT_DIM   = 1024   # fixed by the autoencoder
NUM_FEATURES = 768    # 2 bands × 384 (matches raw-EMG TDSConvCTC default)


class LatentTDSConvCTC(nn.Module):
    """TDSConvCTC adapted for 1024-dim latent-vector input.

    Replaces SpectrogramNorm + MultiBandMLP with a single Linear projection,
    then feeds into the standard TDSConvEncoder + CTC head.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        num_features: int = NUM_FEATURES,
        block_channels: list[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, num_features),     # (T, N, 1024) → (T, N, 768)
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
        return self.model(x)   # (T, N, num_classes)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine annealing (matches raw-EMG baseline)
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
        inputs         = batch["inputs"].to(device)        # (T, N, 1024)
        targets        = batch["targets"].to(device)
        input_lengths  = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad()
        emissions = model(inputs)                           # (T', N, num_classes)

        # TDSConvEncoder shrinks T by its receptive field; compute actual output lengths
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = (input_lengths - T_diff).clamp(min=1)

        loss = criterion(
            emissions,
            targets.transpose(0, 1),
            emission_lengths,
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
def evaluate(model, loader, device, decoder, debug=False):
    model.eval()
    total_loss = 0.0
    criterion  = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    cer_metric = CharacterErrorRates()
    n_valid_targets = 0

    for batch_idx, batch in enumerate(loader):
        inputs         = batch["inputs"].to(device)
        targets        = batch["targets"]
        input_lengths  = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        with torch.backends.cudnn.flags(enabled=False):
            emissions = model(inputs)

        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = (input_lengths - T_diff).clamp(min=1)

        if debug and batch_idx == 0:
            print(f"  [debug] input T={inputs.shape[0]}  emission T={emissions.shape[0]}"
                  f"  T_diff={T_diff}  target_lengths={target_lengths.tolist()}")

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
                continue   # skip windows with no keystrokes
            target = LabelData.from_labels(targets_np[:tgt_len, i])
            cer_metric.update(prediction=pred, target=target)
            n_valid_targets += tgt_len

    if debug:
        print(f"  [debug] total target chars across eval: {n_valid_targets}")

    if n_valid_targets == 0:
        return total_loss / max(len(loader), 1), {"CER": float("nan")}

    metrics = cer_metric.compute()
    return total_loss / max(len(loader), 1), metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    # Pre-parse --from-hyperparams so its values become defaults below.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--from-hyperparams", type=Path, default=None)
    _pre_args, _ = _pre.parse_known_args()
    _hp: dict = {}
    if _pre_args.from_hyperparams is not None:
        with open(_pre_args.from_hyperparams) as _f:
            _hp = yaml.safe_load(_f) or {}
        print(f"Loaded hyperparameters from {_pre_args.from_hyperparams}")

    p = argparse.ArgumentParser(
        description="Train CNN baseline on AE latent vectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hdf5-path", type=Path,
                   default=_ROOT / "data" / "emg_latent_ae_v2.hdf5")
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path(__file__).resolve().parents[1] / "checkpoints")
    p.add_argument("--epochs",       type=int,   default=150)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--num-workers",  type=int,   default=0)
    p.add_argument("--window-length",type=int,   default=_hp.get("window_length", 500),
                   help="Latent frames per window. TDSConvEncoder needs >124 frames "
                        "of headroom (4 blocks × kernel_width=32). Default 500 "
                        "gives 376 output frames.")
    p.add_argument("--stride",       type=int,   default=_hp.get("stride", 50),
                   help="Stride between windows in latent frames.")
    p.add_argument("--lr",           type=float, default=_hp.get("lr", 3e-4))
    p.add_argument("--weight-decay", type=float, default=_hp.get("weight_decay", 1e-5))
    p.add_argument("--warmup-frac",  type=float, default=0.05,
                   help="Fraction of total steps used for LR warmup")
    p.add_argument("--num-features", type=int,   default=_hp.get("num_features", NUM_FEATURES))
    p.add_argument("--block-channels",type=int,  default=_hp.get("block_channels", 24),
                   help="TDS block channels (same for all blocks)")
    p.add_argument("--num-blocks",   type=int,   default=_hp.get("num_blocks", 4),
                   help="Number of TDSConv blocks")
    p.add_argument("--kernel-width", type=int,   default=_hp.get("kernel_width", 32),
                   help="TDSConv temporal kernel width")
    p.add_argument("--from-hyperparams", type=Path, default=None,
                   help="YAML from hyperparam_tuner_cnn.py; sets lr, weight_decay, "
                        "num_features, block_channels, num_blocks, kernel_width")
    p.add_argument("--resume",       type=Path,  default=None,
                   help="Checkpoint .pt to resume from")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data:   {args.hdf5_path}")

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = get_latent_dataloaders(
        hdf5_path=args.hdf5_path,
        window_length=args.window_length,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LatentTDSConvCTC(
        num_features=args.num_features,
        block_channels=[args.block_channels] * args.num_blocks,
        kernel_width=args.kernel_width,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params/1e6:.2f} M")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")

    # ── Optimiser / scheduler ─────────────────────────────────────────────────
    optimizer  = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    total_steps   = args.epochs * len(loaders["train"])
    warmup_steps  = int(args.warmup_frac * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps),
    )
    criterion = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    decoder   = CTCGreedyDecoder()

    # ── Checkpointing ─────────────────────────────────────────────────────────
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = args.checkpoint_dir / "best_latent_cnn.pt"
    best_val_cer = float("inf")

    # ── Logging ───────────────────────────────────────────────────────────────
    run_id    = make_run_id("CNN", num_channels=16, sampling_rate_hz=62,
                            train_fraction=1.0)
    t_start   = time.time()
    print(f"\nRun ID: {run_id}")
    print(f"Starting training for {args.epochs} epochs...\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        t_epoch = time.time()

        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, scheduler
        )
        val_loss, val_metrics = evaluate(model, loaders["val"], device, decoder)

        val_cer = val_metrics["CER"]
        epoch_sec = time.time() - t_epoch

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_CER={val_cer:.2f}% | "
            f"{epoch_sec:.0f}s"
        )

        # Log to team CSV
        log_epoch(
            run_id=run_id,
            model="CNN",
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            val_cer=val_cer,
        )

        # Save best checkpoint
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
    test_cer = test_metrics["CER"]
    training_sec = time.time() - t_start

    print(f"\nTest CER: {test_cer:.2f}%")
    print(f"Training time: {training_sec/3600:.2f} hrs")

    log_summary(
        run_id=run_id,
        model="CNN",
        epochs=args.epochs,
        num_channels=16,
        sampling_rate_hz=62,
        train_fraction=1.0,
        input_type="latent_ae_v2",
        final_train_loss=train_loss,
        final_val_loss=val_loss,
        final_val_cer=best_val_cer,
        test_cer=test_cer,
        training_time_sec=training_sec,
        notes="latent_cnn_baseline",
    )  # note: sampling_rate_hz=62 represents 1/32ms latent frame rate
    print(f"\nDone. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
