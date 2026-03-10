"""Quick training test on a single processed HDF5 file using the Conformer model.

Always trains on data/emg_latent_ae.hdf5, loading hyperparameters from
Playground_Kai/checkpoints/best_hyperparams_conformer.yaml.

Run from the workspace root:
    python -m Playground_Kai.train_test
    python -m Playground_Kai.train_test --epochs 150
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
from torch.utils.data import DataLoader, random_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.decoder import CTCGreedyDecoder
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.transforms import Compose, LogSpectrogram, ToTensor

from Playground_Kai.model import ConformerEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HDF5_LATENT  = _ROOT / "data" / "emg_latent_ae.hdf5"
_HDF5_CONTROL = _ROOT / "data" / "2021-06-02-1622679967-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5"
_HYPERPARAMS  = Path(__file__).resolve().parent / "checkpoints" / "best_hyperparams_conformer.yaml"

# Raw LogSpectrogram pipeline (n_fft=64, hop=16 → 33 bins, 16 channels → in_features=528)
_TRANSFORM = Compose([ToTensor(), LogSpectrogram(n_fft=64, hop_length=16)])

# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay (step-level)
# ---------------------------------------------------------------------------

def _lr_lambda(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        inputs         = batch["inputs"].to(device)
        targets        = batch["targets"].to(device)
        input_lengths  = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        optimizer.zero_grad()
        emissions = model(inputs)
        loss = criterion(emissions, targets.transpose(0, 1), input_lengths, target_lengths)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, decoder) -> tuple[float, float]:
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
        for i, pred in enumerate(preds):
            tgt = LabelData.from_labels(targets.numpy()[: target_lengths[i], i])
            cer_metric.update(prediction=pred, target=tgt)
    metrics = cer_metric.compute()
    return total_loss / max(len(loader), 1), metrics.get("CER", float("nan"))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Quick Conformer test on a single HDF5 file")
    p.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    p.add_argument("--control", action="store_true",
                   help="Use the known-good control session instead of emg_latent_ae.hdf5")
    args = p.parse_args()

    # Load hyperparameters
    with open(_HYPERPARAMS) as f:
        hp = yaml.safe_load(f) or {}
    print(f"Hyperparameters loaded from {_HYPERPARAMS.name}:")
    for k in ("lr", "weight_decay", "d_model", "num_heads", "num_layers", "conv_kernel_size", "dropout"):
        if k in hp:
            print(f"  {k} = {hp[k]}")

    hdf5_file = _HDF5_CONTROL if args.control else _HDF5_LATENT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Data   : {hdf5_file}")
    print(f"Epochs : {args.epochs}")

    if not hdf5_file.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")

    # Build dataset from the single file and split 80/20 train/val
    full_dataset = WindowedEMGDataset(
        hdf5_path=hdf5_file,
        window_length=8000,   # 4 s @ 2 kHz
        stride=None,
        padding=(1800, 200),
        jitter=False,
        transform=_TRANSFORM,
    )
    n_total = len(full_dataset)
    n_train = max(1, int(0.8 * n_total))
    n_val   = n_total - n_train
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
    print(f"Windows: {n_total} total  ({n_train} train / {n_val} val)")

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        collate_fn=WindowedEMGDataset.collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        collate_fn=WindowedEMGDataset.collate,
    )

    # Model
    model = ConformerEncoder(
        in_features=528,        # 16 channels × 33 STFT bins (raw pipeline)
        mlp_features=(384,),
        d_model=int(hp.get("d_model", 256)),
        num_heads=int(hp.get("num_heads", 4)),
        num_layers=int(hp.get("num_layers", 4)),
        conv_kernel_size=int(hp.get("conv_kernel_size", 31)),
        dropout=float(hp.get("dropout", 0.1)),
        electrode_channels=16,
    ).to(device)
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hp.get("lr", 5e-4)),
        weight_decay=float(hp.get("weight_decay", 1e-2)),
    )
    steps_per_epoch = len(train_loader)
    total_steps  = args.epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch
    min_lr_ratio = 1e-5 / float(hp.get("lr", 5e-4))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps, min_lr_ratio),
    )
    decoder = CTCGreedyDecoder()

    print(f"\nTraining …\n")
    best_cer = float("inf")
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_cer = evaluate(model, val_loader, device, decoder)
        elapsed = time.perf_counter() - t0
        marker = " *" if val_cer < best_cer else ""
        if val_cer < best_cer:
            best_cer = val_cer
        print(
            f"Epoch {epoch+1:3d}/{args.epochs}"
            f" | train_loss={train_loss:.4f}"
            f" | val_loss={val_loss:.4f}"
            f" | val_CER={val_cer:.2f}%"
            f" | {elapsed:.1f}s{marker}"
        )

    print(f"\nDone. Best val CER: {best_cer:.2f}%")


if __name__ == "__main__":
    main()
