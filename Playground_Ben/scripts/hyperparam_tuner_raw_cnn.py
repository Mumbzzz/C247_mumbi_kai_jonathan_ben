"""Two-phase hyperparameter tuner for TDSConvCTC on raw EMG data.

Sweeps CNN-specific hyperparameters (mlp_features, block_channels, kernel_width,
num_blocks) plus optimiser settings (lr, weight_decay) using a subset of the
full session pool for fast proxy training.

Mirrors Playground_Kai/hyperparam_tuner.py in structure but targets the
TDSConvCTC architecture on the raw log-spectrogram pipeline.

Outputs are saved exclusively to Playground_Ben/checkpoints/ — Kai's results
are never touched.

Run from the repo root with the venv active:
    python Playground_Ben/scripts/hyperparam_tuner_raw_cnn.py
    python Playground_Ben/scripts/hyperparam_tuner_raw_cnn.py \\
        --coarse-trials 30 --coarse-epochs 6 --trial-sessions 3

After tuning, update the emg2qwerty Hydra config or pass overrides directly:
    python -m emg2qwerty.train \\
        user=single_user \\
        ++module.block_channels=[32,32,32,32] \\
        ++module.kernel_width=24 \\
        optimizer.lr=5e-4
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

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

from Playground_Kai.data_utils import build_loaders_from_paths, get_session_paths
from Playground_Ben.scripts.train_latent_cnn import _lr_lambda

_OOM_ERRORS: tuple[type[BaseException], ...] = (torch.cuda.OutOfMemoryError,)
if hasattr(torch, "AcceleratorError"):
    _OOM_ERRORS = _OOM_ERRORS + (torch.AcceleratorError,)

# Fixed by emg2qwerty default pipeline
FREQ_BINS:  int = 33   # n_fft=64 → n_fft//2+1=33
NUM_BANDS:  int = 2    # left + right wrist
ELECTRODE_CHANNELS: int = 16
IN_FEATURES: int = FREQ_BINS * ELECTRODE_CHANNELS   # 528


# ---------------------------------------------------------------------------
# Standalone TDSConvCTC model (no Lightning dependency)
# ---------------------------------------------------------------------------

class RawTDSConvCTC(nn.Module):
    """TDSConvCTC built from emg2qwerty modules, without Lightning.

    Accepts raw log-spectrogram batches of shape (T, N, 2, 16, 33) and
    outputs (T', N, num_classes) log-softmax activations.

    Pipeline mirrors TDSConvCTCModule exactly:
        SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten → TDSConvEncoder → Linear → LogSoftmax
    """

    def __init__(
        self,
        in_features: int = IN_FEATURES,
        mlp_features: int = 384,
        block_channels: int = 24,
        num_blocks: int = 4,
        kernel_width: int = 32,
    ) -> None:
        super().__init__()
        electrode_channels = in_features // FREQ_BINS
        num_features = NUM_BANDS * mlp_features

        # num_features must be divisible by block_channels for TDSConvEncoder
        block_ch = block_channels
        num_features = max(block_ch, round(num_features / block_ch) * block_ch)

        self.model = nn.Sequential(
            SpectrogramNorm(channels=NUM_BANDS * electrode_channels),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=[mlp_features],
                num_bands=NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=[block_ch] * num_blocks,
                kernel_width=kernel_width,
            ),
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Train / evaluate (with T_diff correction for TDS receptive field)
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
    n_valid = 0

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
# Search space
# ---------------------------------------------------------------------------

SEARCH_SPACE_CNN: dict[str, dict[str, Any]] = {
    "lr":             {"type": "log_uniform", "low": 3e-4,  "high": 3e-3},
    "weight_decay":   {"type": "log_uniform", "low": 1e-5,  "high": 1e-2},
    # mlp_features: per-band MLP output; num_features = 2 × mlp_features
    "mlp_features":   {"type": "choice",      "choices": [256, 384, 512]},
    # block_channels must divide num_features = 2×mlp_features → use values that divide 512,768,1024
    "block_channels": {"type": "choice",      "choices": [16, 32, 64]},
    "kernel_width":   {"type": "choice",      "choices": [16, 24, 32, 48]},
    "num_blocks":     {"type": "choice",      "choices": [2, 3, 4]},
}


def sample_config(rng: random.Random, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for name, spec in space.items():
        if spec["type"] == "log_uniform":
            config[name] = math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"])))
        elif spec["type"] == "uniform":
            config[name] = rng.uniform(spec["low"], spec["high"])
        elif spec["type"] == "choice":
            config[name] = rng.choice(spec["choices"])
    return config


def make_fine_search_space(
    base: dict[str, Any],
    original: dict[str, dict[str, Any]],
    shrink: float,
) -> dict[str, dict[str, Any]]:
    fine: dict[str, dict[str, Any]] = {}
    for name, spec in original.items():
        v = base[name]
        if spec["type"] == "log_uniform":
            lo = max(spec["low"],  v / shrink)
            hi = min(spec["high"], v * shrink)
            if lo >= hi:
                lo, hi = spec["low"], spec["high"]
            fine[name] = {"type": "log_uniform", "low": lo, "high": hi}
        elif spec["type"] == "uniform":
            half = (spec["high"] - spec["low"]) / 2.0 / shrink
            lo = max(spec["low"],  v - half)
            hi = min(spec["high"], v + half)
            if lo >= hi:
                lo, hi = spec["low"], spec["high"]
            fine[name] = {"type": "uniform", "low": lo, "high": hi}
        elif spec["type"] == "choice":
            fine[name] = {"type": "choice", "choices": [v]}
    return fine


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

TrialResult = tuple[str, int, dict[str, Any], float]


def run_trial(
    trial_idx: int,
    config: dict[str, Any],
    train_paths: list[Path],
    val_paths: list[Path],
    trial_epochs: int,
    window_length: int,
    batch_size: int,
    device: torch.device,
    timeout_secs: float = float("inf"),
    early_stopping_patience: int = 0,
) -> tuple[float, bool, int]:
    torch.manual_seed(trial_idx)

    loaders = build_loaders_from_paths(
        train_paths=train_paths,
        val_paths=val_paths,
        window_length=window_length,
        batch_size=batch_size,
        num_workers=0,
    )

    model = RawTDSConvCTC(
        mlp_features=int(config["mlp_features"]),
        block_channels=int(config["block_channels"]),
        num_blocks=int(config["num_blocks"]),
        kernel_width=int(config["kernel_width"]),
    ).to(device)

    criterion = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    steps_per_epoch = len(loaders["train"])
    total_steps  = trial_epochs * steps_per_epoch
    warmup_steps = max(1, trial_epochs // 5) * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps, min_lr_ratio=0.02),
    )

    decoder = CTCGreedyDecoder()
    best_val_cer = float("inf")
    timed_out = False
    epochs_run = 0
    no_improve_streak = 0
    t_start = time.perf_counter()

    try:
        for _ in range(trial_epochs):
            train_one_epoch(model, loaders["train"], optimizer, criterion, device, scheduler)
            _, val_metrics = evaluate(model, loaders["val"], device, decoder)
            val_cer = val_metrics["CER"]
            if math.isnan(val_cer):
                val_cer = float("inf")
            epochs_run += 1

            if val_cer < best_val_cer:
                best_val_cer = val_cer
                no_improve_streak = 0
            else:
                no_improve_streak += 1

            if early_stopping_patience > 0 and no_improve_streak >= early_stopping_patience:
                break
            if time.perf_counter() - t_start > timeout_secs:
                timed_out = True
                break

    except _OOM_ERRORS as e:
        if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e).lower():
            raise
        print("\n    [OOM — skipping config]", end="  ", flush=True)
        best_val_cer = float("inf")

    finally:
        del optimizer, scheduler, model
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return best_val_cer, timed_out, epochs_run


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def _print_config_inline(config: dict[str, Any]) -> None:
    print(
        f"  lr={config['lr']:.2e}"
        f"  mlp={config['mlp_features']}"
        f"  ch={config['block_channels']}"
        f"  blocks={config['num_blocks']}"
        f"  kw={config['kernel_width']}"
        f"  wd={config['weight_decay']:.2e}",
        end="  ",
        flush=True,
    )


def _run_phase(
    phase_label: str,
    num_trials: int,
    search_space: dict[str, dict[str, Any]],
    train_paths: list[Path],
    val_paths: list[Path],
    trial_epochs: int,
    window_length: int,
    batch_size: int,
    device: torch.device,
    timeout_secs: float,
    early_stopping_patience: int,
    rng: random.Random,
    trial_idx_offset: int,
) -> list[TrialResult]:
    results: list[TrialResult] = []
    for local_idx in range(num_trials):
        global_idx = trial_idx_offset + local_idx
        config = sample_config(rng, search_space)
        print(f"[{phase_label}] Trial {local_idx + 1:>3}/{num_trials}", end="", flush=True)
        _print_config_inline(config)

        t0 = time.perf_counter()
        try:
            val_cer, timed_out, epochs_run = run_trial(
                trial_idx=global_idx,
                config=config,
                train_paths=train_paths,
                val_paths=val_paths,
                trial_epochs=trial_epochs,
                window_length=window_length,
                batch_size=batch_size,
                device=device,
                timeout_secs=timeout_secs,
                early_stopping_patience=early_stopping_patience,
            )
        except KeyboardInterrupt:
            print(f"\n[Interrupted] Phase [{phase_label}] stopped after {len(results)} trials.")
            break

        elapsed = time.perf_counter() - t0
        tags: list[str] = []
        if math.isinf(val_cer):
            tags.append("OOM/nan — skipped")
        else:
            if timed_out:
                tags.append("timeout")
            if early_stopping_patience > 0 and epochs_run < trial_epochs and not timed_out:
                tags.append(f"early-stop ep={epochs_run}")
        tag_str = "  [" + ", ".join(tags) + "]" if tags else ""
        cer_str = "OOM" if math.isinf(val_cer) else f"{val_cer:.2f}%"
        print(f"-> val_CER={cer_str}  ({elapsed:.1f}s){tag_str}")
        results.append((phase_label, global_idx, config, val_cer))

    return results


def _print_results_table(results: list[TrialResult]) -> None:
    sorted_results = sorted(results, key=lambda x: x[3])
    col = 112
    print("\n" + "=" * col)
    print(
        f"{'Rank':>4}  {'Phase':>5}  {'Trial':>5}  {'val_CER':>8}  {'lr':>10}  "
        f"{'mlp':>5}  {'ch':>4}  {'blocks':>6}  {'kw':>4}  {'weight_decay':>12}"
    )
    print("-" * col)
    for rank, (phase, global_idx, cfg, cer) in enumerate(sorted_results, 1):
        cer_str = "   OOM  " if math.isinf(cer) else f"{cer:>7.2f}%"
        print(
            f"{rank:>4}  {phase:>5}  {global_idx + 1:>5}  {cer_str}  "
            f"{cfg['lr']:>10.2e}  {int(cfg['mlp_features']):>5}  "
            f"{int(cfg['block_channels']):>4}  {int(cfg['num_blocks']):>6}  "
            f"{int(cfg['kernel_width']):>4}  {cfg['weight_decay']:>12.2e}"
        )
    print("=" * col)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-phase hyperparameter tuner for TDSConvCTC on raw EMG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=_ROOT / "data")
    p.add_argument("--config", type=Path,
                   default=_ROOT / "config" / "user" / "single_user.yaml")
    p.add_argument("--output", type=Path, default=None,
                   help="YAML output (default: Playground_Ben/checkpoints/best_hyperparams_raw_cnn.yaml)")
    p.add_argument("--trial-sessions", type=int, default=5,
                   help="Training sessions per trial (first N from split YAML)")
    p.add_argument("--search-mode", choices=["two-phase", "coarse-only"], default="two-phase")
    p.add_argument("--coarse-trials",  type=int, default=20)
    p.add_argument("--coarse-epochs",  type=int, default=8,
                   help="Epochs per coarse proxy run")
    p.add_argument("--fine-trials",    type=int, default=10)
    p.add_argument("--fine-epochs",    type=int, default=15)
    p.add_argument("--fine-top-k",     type=int, default=3)
    p.add_argument("--fine-shrink",    type=float, default=3.0)
    p.add_argument("--confirm-epochs", type=int, default=0)
    p.add_argument("--trial-timeout",  type=float, default=300.0,
                   help="Per-trial wall-clock cap in seconds (0 = no limit)")
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--batch-size",     type=int, default=32)
    p.add_argument("--window-length",  type=int, default=8000,
                   help="Raw EMG samples per window (8000 = 4 s @ 2 kHz)")
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    timeout_secs = float("inf") if args.trial_timeout == 0 else args.trial_timeout
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output is None:
        args.output = (
            Path(__file__).resolve().parents[1]
            / "checkpoints"
            / "best_hyperparams_raw_cnn.yaml"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    session_paths = get_session_paths(data_root=args.data_root, config_path=args.config)
    all_train = session_paths["train"]
    val_paths  = session_paths["val"]

    if args.trial_sessions > len(all_train):
        raise ValueError(
            f"--trial-sessions={args.trial_sessions} > available sessions ({len(all_train)})"
        )
    trial_train = all_train[: args.trial_sessions]

    print(f"Device          : {device}")
    print(f"Model           : TDSConvCTC (raw log-spectrogram, 2000 Hz)")
    print(f"Data root       : {args.data_root}")
    print(f"Train sessions  : {len(trial_train)} / {len(all_train)}")
    print(f"Val sessions    : {len(val_paths)}")
    print(f"Window          : {args.window_length} samples ({args.window_length/2000:.1f}s @ 2kHz)")
    print(f"Search mode     : {args.search_mode}")
    if args.search_mode == "two-phase":
        top_k = min(args.fine_top_k, args.coarse_trials)
        print(f"Coarse          : {args.coarse_trials} trials × {args.coarse_epochs} epochs")
        print(f"Fine            : {top_k} anchors × {args.fine_trials} trials × {args.fine_epochs} epochs")
        print(f"Fine shrink     : {args.fine_shrink:.1f}x")
    else:
        print(f"Trials × epochs : {args.coarse_trials} × {args.coarse_epochs}")
    if timeout_secs != float("inf"):
        print(f"Trial timeout   : {timeout_secs:.0f}s")
    print(f"Output YAML     : {args.output}")
    print(f"Search params   : {list(SEARCH_SPACE_CNN.keys())}")
    print("Press Ctrl+C to stop — completed trials will still be ranked.\n")

    rng = random.Random(args.seed)
    all_results: list[TrialResult] = []
    total_start = time.perf_counter()

    # ── Coarse phase ──────────────────────────────────────────────────────────
    print("=" * 62)
    print(f"  PHASE C — Coarse search  ({args.coarse_trials} trials, {args.coarse_epochs} epochs each)")
    print("=" * 62)
    coarse_results = _run_phase(
        phase_label="C",
        num_trials=args.coarse_trials,
        search_space=SEARCH_SPACE_CNN,
        train_paths=trial_train,
        val_paths=val_paths,
        trial_epochs=args.coarse_epochs,
        window_length=args.window_length,
        batch_size=args.batch_size,
        device=device,
        timeout_secs=timeout_secs,
        early_stopping_patience=args.early_stopping_patience,
        rng=rng,
        trial_idx_offset=0,
    )
    all_results.extend(coarse_results)

    # ── Fine phase ────────────────────────────────────────────────────────────
    fine_results: list[TrialResult] = []
    if args.search_mode == "two-phase" and coarse_results:
        top_k = min(args.fine_top_k, len(coarse_results))
        top_k_coarse = sorted(coarse_results, key=lambda x: x[3])[:top_k]
        print(f"\n{'=' * 62}")
        print(f"  PHASE F — Fine search  ({top_k} anchors × {args.fine_trials} trials, {args.fine_epochs} epochs)")
        for rank, (_, gidx, cfg, cer) in enumerate(top_k_coarse, 1):
            print(f"    Anchor {rank}: trial {gidx + 1}  val_CER={cer:.2f}%")
        print("=" * 62)

        fine_offset = args.coarse_trials
        for anchor_idx, (_, _, anchor_cfg, anchor_cer) in enumerate(top_k_coarse):
            fine_space = make_fine_search_space(anchor_cfg, SEARCH_SPACE_CNN, args.fine_shrink)
            label = f"F{anchor_idx + 1}"
            print(f"\n  -- Anchor {anchor_idx + 1}/{top_k}  (coarse CER={anchor_cer:.2f}%) --")
            phase_results = _run_phase(
                phase_label=label,
                num_trials=args.fine_trials,
                search_space=fine_space,
                train_paths=trial_train,
                val_paths=val_paths,
                trial_epochs=args.fine_epochs,
                window_length=args.window_length,
                batch_size=args.batch_size,
                device=device,
                timeout_secs=timeout_secs,
                early_stopping_patience=args.early_stopping_patience,
                rng=rng,
                trial_idx_offset=fine_offset,
            )
            fine_results.extend(phase_results)
            fine_offset += args.fine_trials
            if len(phase_results) < args.fine_trials:
                break

        all_results.extend(fine_results)

    if not all_results:
        print("No trials completed.")
        return

    total_mins = (time.perf_counter() - total_start) / 60
    print(f"\nCompleted {len(all_results)} trials in {total_mins:.1f} min.")

    best_phase, best_global_idx, best_config, best_cer = min(all_results, key=lambda x: x[3])

    # ── Optional confirmation run ──────────────────────────────────────────────
    if args.confirm_epochs > 0:
        print(f"\n{'=' * 62}")
        print(f"  CONFIRMATION RUN  ({args.confirm_epochs} epochs)")
        print(f"  Best: phase={best_phase}  trial={best_global_idx + 1}  CER={best_cer:.2f}%")
        print("=" * 62)
        _print_config_inline(best_config)
        print()
        t0 = time.perf_counter()
        try:
            confirm_cer, _, confirm_ep = run_trial(
                trial_idx=10_000 + best_global_idx,
                config=best_config,
                train_paths=trial_train,
                val_paths=val_paths,
                trial_epochs=args.confirm_epochs,
                window_length=args.window_length,
                batch_size=args.batch_size,
                device=device,
                timeout_secs=timeout_secs,
            )
        except KeyboardInterrupt:
            confirm_cer, confirm_ep = best_cer, 0
        print(f"  Confirmation CER={confirm_cer:.2f}%  ({time.perf_counter()-t0:.1f}s, {confirm_ep} epochs)")
        best_cer = confirm_cer

    _print_results_table(all_results)
    cer_str = "OOM" if math.isinf(best_cer) else f"{best_cer:.2f}%"
    print(f"\nBest: phase={best_phase}  trial={best_global_idx + 1}  val_CER={cer_str}")

    if math.isinf(best_cer):
        print("All trials OOM/nan — try reducing --batch-size or model sizes.")
        return

    # ── Save YAML ─────────────────────────────────────────────────────────────
    output_data: dict[str, Any] = {
        "lr":             float(best_config["lr"]),
        "weight_decay":   float(best_config["weight_decay"]),
        "mlp_features":   int(best_config["mlp_features"]),
        "block_channels": int(best_config["block_channels"]),
        "num_blocks":     int(best_config["num_blocks"]),
        "kernel_width":   int(best_config["kernel_width"]),
        # Metadata
        "trial_val_cer":      round(float(best_cer), 4),
        "search_mode":        args.search_mode,
        "search_phase_found": best_phase,
        "num_coarse_trials":  len(coarse_results),
        "num_fine_trials":    len(fine_results),
        "coarse_epochs":      args.coarse_epochs,
        "fine_epochs":        args.fine_epochs if args.search_mode == "two-phase" else None,
        "fine_shrink":        args.fine_shrink if args.search_mode == "two-phase" else None,
        "tuned_at":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_data = {k: v for k, v in output_data.items() if v is not None}

    with open(args.output, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved to: {args.output}")
    print("\nTo train with the best config (example Hydra overrides):")
    bc = int(best_config["block_channels"])
    nb = int(best_config["num_blocks"])
    kw = int(best_config["kernel_width"])
    mf = int(best_config["mlp_features"])
    nf = NUM_BANDS * mf
    # Snap to multiple of block_channels as done in run_trial
    nf = max(bc, round(nf / bc) * bc)
    print(
        f"  python -m emg2qwerty.train user=single_user \\\n"
        f"      ++module.block_channels={[bc]*nb} \\\n"
        f"      ++module.kernel_width={kw} \\\n"
        f"      ++module.mlp_features=[{mf}] \\\n"
        f"      ++module.in_features={IN_FEATURES} \\\n"
        f"      optimizer.lr={best_config['lr']:.2e}"
    )


if __name__ == "__main__":
    main()
