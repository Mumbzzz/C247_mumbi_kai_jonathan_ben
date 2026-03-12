"""Hyperparameter tuner for CNN+LSTM on AE-reconstructed EMG v3 data.

Operates on ``_recons_v3.hdf5`` files (62.5 Hz, 32-dim input per frame).
Uses the same two-phase random search strategy as ``hyperparam_tuner_latent.py``.

Usage:
    python -m Playground_Mumbi.hyperparam_tuner_recons [options]

Key flags:
    --search-mode       two-phase | coarse-only          (default: two-phase)
    --coarse-trials     Configs evaluated in coarse phase (default: 20)
    --coarse-epochs     Epochs per coarse proxy run       (default: 8)
    --fine-trials       Configs per fine anchor           (default: 10)
    --fine-epochs       Epochs per fine proxy run         (default: 15)
    --fine-top-k        Top coarse configs as fine anchors (default: 3)
    --trial-sessions    Train sessions per trial          (default: 8)
    --trial-timeout     Per-trial wall-clock cap (s)      (default: 300)
    --early-stopping-patience  Stop after N non-improving epochs (default: 0)

After tuning, train with:
    python -m Playground_Mumbi.train_recons \\
        --from-hyperparams Playground_Mumbi/checkpoints/best_hyperparams_cnn_lstm_recons.yaml
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from emg2qwerty.charset import charset
from emg2qwerty.decoder import CTCGreedyDecoder
from torch.utils.data import DataLoader

from Playground_Mumbi.data_utils import ReconstructedEMGDataset
from Playground_Mumbi.model_latent import LatentCNNLSTMModel
from Playground_Mumbi.train_recons import _lr_lambda, evaluate, train_one_epoch

_OOM_ERRORS: tuple[type[BaseException], ...] = (torch.cuda.OutOfMemoryError,)
if hasattr(torch, "AcceleratorError"):
    _OOM_ERRORS = _OOM_ERRORS + (torch.AcceleratorError,)  # type: ignore[attr-defined]

RECONS_DIM: int = 32


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

SEARCH_SPACE_CNN_LSTM_RECONS: dict[str, dict[str, Any]] = {
    "lr":            {"type": "log_uniform", "low": 1e-4,  "high": 1e-2},
    "proj_features": {"type": "choice",      "choices": [64, 128, 256]},
    "cnn_channels":  {"type": "choice",      "choices": [64, 128, 256]},
    "cnn_kernel":    {"type": "choice",      "choices": [3, 5, 7]},
    "cnn_layers":    {"type": "choice",      "choices": [1, 2, 3]},
    "lstm_hidden":   {"type": "choice",      "choices": [128, 256, 384, 512]},
    "lstm_layers":   {"type": "choice",      "choices": [1, 2, 3]},
    "dropout":       {"type": "uniform",     "low": 0.1,  "high": 0.5},
    "weight_decay":  {"type": "log_uniform", "low": 1e-5, "high": 1e-1},
}


def sample_config(rng: random.Random, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for name, spec in search_space.items():
        if spec["type"] == "log_uniform":
            config[name] = math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"])))
        elif spec["type"] == "uniform":
            config[name] = rng.uniform(spec["low"], spec["high"])
        elif spec["type"] == "choice":
            config[name] = rng.choice(spec["choices"])
    return config


def make_fine_search_space(
    base_config: dict[str, Any],
    original_space: dict[str, dict[str, Any]],
    shrink_factor: float,
) -> dict[str, dict[str, Any]]:
    fine_space: dict[str, dict[str, Any]] = {}
    for name, spec in original_space.items():
        v = base_config[name]
        if spec["type"] == "log_uniform":
            new_low  = max(spec["low"],  v / shrink_factor)
            new_high = min(spec["high"], v * shrink_factor)
            if new_low >= new_high:
                new_low, new_high = spec["low"], spec["high"]
            fine_space[name] = {"type": "log_uniform", "low": new_low, "high": new_high}
        elif spec["type"] == "uniform":
            half     = (spec["high"] - spec["low"]) / 2.0 / shrink_factor
            new_low  = max(spec["low"],  v - half)
            new_high = min(spec["high"], v + half)
            if new_low >= new_high:
                new_low, new_high = spec["low"], spec["high"]
            fine_space[name] = {"type": "uniform", "low": new_low, "high": new_high}
        elif spec["type"] == "choice":
            fine_space[name] = {"type": "choice", "choices": [v]}
    return fine_space


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(
    trial_idx: int,
    config: dict[str, Any],
    data_dir: Path,
    config_path: Path,
    trial_sessions: int,
    trial_epochs: int,
    window_length: int,
    batch_size: int,
    device: torch.device,
    timeout_secs: float = float("inf"),
    early_stopping_patience: int = 0,
) -> tuple[float, bool, int]:
    """Run one proxy training trial. Returns ``(best_val_cer, timed_out, epochs_run)``."""
    import yaml as _yaml
    torch.manual_seed(trial_idx)

    with open(config_path) as _f:
        split_cfg = _yaml.safe_load(_f)

    train_sessions = [e["session"] for e in split_cfg["dataset"]["train"]][:trial_sessions]
    val_sessions   = [e["session"] for e in split_cfg["dataset"]["val"]]

    train_paths = [data_dir / f"{s}_recons_v3.hdf5" for s in train_sessions]
    val_paths   = [data_dir / f"{s}_recons_v3.hdf5" for s in val_sessions]

    collate = ReconstructedEMGDataset.collate
    train_ds = ReconstructedEMGDataset(train_paths, window_length=window_length,
                                       stride=window_length, jitter=True)
    val_ds   = ReconstructedEMGDataset(val_paths,   window_length=window_length,
                                       stride=window_length, jitter=False)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate),
    }

    effective_dropout = config["dropout"] if config["lstm_layers"] > 1 else 0.0
    model = LatentCNNLSTMModel(
        latent_dim=RECONS_DIM,
        proj_features=config["proj_features"],
        cnn_channels=config["cnn_channels"],
        cnn_kernel=config["cnn_kernel"],
        cnn_layers=config["cnn_layers"],
        lstm_hidden=config["lstm_hidden"],
        lstm_layers=config["lstm_layers"],
        dropout=effective_dropout,
        num_classes=charset().num_classes,
    ).to(device)

    criterion = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                   weight_decay=config["weight_decay"])

    steps_per_epoch = len(loaders["train"])
    total_steps  = trial_epochs * steps_per_epoch
    warmup_steps = max(1, trial_epochs // 5) * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps, 0.02),
    )

    decoder      = CTCGreedyDecoder()
    best_val_cer = float("inf")
    timed_out    = False
    epochs_run   = 0
    no_improve   = 0
    t_start      = time.perf_counter()

    try:
        for _ in range(trial_epochs):
            train_one_epoch(model, loaders["train"], optimizer, criterion, device, scheduler)
            _, val_metrics = evaluate(model, loaders["val"], device, decoder)
            val_cer = val_metrics["CER"]
            epochs_run += 1
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                no_improve   = 0
            else:
                no_improve += 1
            if early_stopping_patience > 0 and no_improve >= early_stopping_patience:
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

TrialResult = tuple[str, int, dict[str, Any], float]


def _print_config_inline(config: dict[str, Any]) -> None:
    print(
        f"  lr={config['lr']:.2e}"
        f"  proj={config['proj_features']}"
        f"  cnn_ch={config['cnn_channels']}"
        f"  cnn_k={config['cnn_kernel']}"
        f"  cnn_l={config['cnn_layers']}"
        f"  lstm_h={config['lstm_hidden']}"
        f"  lstm_l={config['lstm_layers']}"
        f"  do={config['dropout']:.3f}"
        f"  wd={config['weight_decay']:.2e}",
        end="  ", flush=True,
    )


def _run_phase(
    phase_label: str,
    num_trials: int,
    search_space: dict[str, dict[str, Any]],
    data_dir: Path,
    config_path: Path,
    trial_sessions: int,
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
        print(f"[{phase_label}] Trial {local_idx+1:>3}/{num_trials}", end="", flush=True)
        _print_config_inline(config)
        t0 = time.perf_counter()
        try:
            val_cer, timed_out, epochs_run = run_trial(
                trial_idx=global_idx, config=config,
                data_dir=data_dir, config_path=config_path,
                trial_sessions=trial_sessions, trial_epochs=trial_epochs,
                window_length=window_length, batch_size=batch_size,
                device=device, timeout_secs=timeout_secs,
                early_stopping_patience=early_stopping_patience,
            )
        except KeyboardInterrupt:
            print(f"\n[Interrupted] Phase [{phase_label}] stopped after {len(results)} trials.")
            break
        elapsed = time.perf_counter() - t0
        tags: list[str] = []
        if math.isinf(val_cer):
            tags.append("OOM — skipped")
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


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _print_results_table(results: list[TrialResult]) -> None:
    sorted_results = sorted(results, key=lambda x: x[3])
    col = 118
    print("\n" + "=" * col)
    print(
        f"{'Rank':>4}  {'Phase':>5}  {'Trial':>5}  {'val_CER':>8}  {'lr':>10}  "
        f"{'proj':>4}  {'cnn_ch':>6}  {'cnn_k':>5}  {'cnn_l':>5}  "
        f"{'lstm_h':>6}  {'lstm_l':>6}  {'dropout':>7}  {'weight_decay':>12}"
    )
    print("-" * col)
    for rank, (phase, global_idx, cfg, cer) in enumerate(sorted_results, 1):
        cer_str = "   OOM  " if math.isinf(cer) else f"{cer:>7.2f}%"
        print(
            f"{rank:>4}  {phase:>5}  {global_idx+1:>5}  {cer_str}  "
            f"{cfg['lr']:>10.2e}  {cfg['proj_features']:>4}  {cfg['cnn_channels']:>6}  "
            f"{cfg['cnn_kernel']:>5}  {cfg['cnn_layers']:>5}  {cfg['lstm_hidden']:>6}  "
            f"{cfg['lstm_layers']:>6}  {cfg['dropout']:>7.3f}  {cfg['weight_decay']:>12.2e}"
        )
    print("=" * col)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-phase hyperparameter tuner for CNN+LSTM on recons_v3 data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path,
                   default=_ROOT / "data" / "89335547_recons_v3",
                   help="Directory containing _recons_v3.hdf5 session files")
    p.add_argument("--recons-config", type=Path,
                   default=_ROOT / "config" / "user" / "single_user.yaml",
                   help="YAML defining train/val/test split")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--search-mode", choices=["two-phase", "coarse-only"], default="two-phase")
    p.add_argument("--coarse-trials", type=int, default=None)
    p.add_argument("--coarse-epochs", type=int, default=None)
    p.add_argument("--num-trials",  type=int, default=20)
    p.add_argument("--trial-epochs", type=int, default=8)
    p.add_argument("--fine-trials", type=int, default=10)
    p.add_argument("--fine-epochs", type=int, default=15)
    p.add_argument("--fine-top-k",  type=int, default=3)
    p.add_argument("--fine-shrink", type=float, default=3.0)
    p.add_argument("--confirm-epochs", type=int, default=0)
    p.add_argument("--trial-sessions", type=int, default=8)
    p.add_argument("--trial-timeout",  type=float, default=300.0)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--window-length", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    coarse_trials = args.coarse_trials if args.coarse_trials is not None else args.num_trials
    coarse_epochs = args.coarse_epochs if args.coarse_epochs is not None else args.trial_epochs
    timeout_secs  = float("inf") if args.trial_timeout == 0 else args.trial_timeout

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output is None:
        args.output = (
            Path(__file__).resolve().parent / "checkpoints" / "best_hyperparams_cnn_lstm_recons.yaml"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device            : {device}")
    print(f"Model             : CNN+LSTM (recons_v3)")
    print(f"Pipeline          : recons_v3 (32-dim @ 62.5 Hz)")
    print(f"Data dir          : {args.data_dir}")
    print(f"Recons config     : {args.recons_config}")
    print(f"Trial sessions    : {args.trial_sessions}")
    print(f"Window length     : {args.window_length} frames (≈ {args.window_length / 62.5:.1f}s)")
    print(f"Search mode       : {args.search_mode}")
    if args.search_mode == "two-phase":
        top_k_display = min(args.fine_top_k, coarse_trials)
        print(f"Coarse            : {coarse_trials} trials × {coarse_epochs} epochs")
        print(f"Fine              : {top_k_display} anchors × {args.fine_trials} trials × {args.fine_epochs} epochs")
        print(f"Fine shrink       : {args.fine_shrink:.1f}x")
    else:
        print(f"Trials × epochs   : {coarse_trials} × {coarse_epochs}")
    if timeout_secs != float("inf"):
        print(f"Trial timeout     : {timeout_secs:.0f}s")
    if args.early_stopping_patience > 0:
        print(f"Early stopping    : patience={args.early_stopping_patience}")
    if args.confirm_epochs > 0:
        print(f"Confirmation run  : {args.confirm_epochs} epochs on overall best")
    print(f"Search params     : {list(SEARCH_SPACE_CNN_LSTM_RECONS.keys())}")
    print("Press Ctrl+C to stop early — completed trials will still be ranked.\n")

    rng = random.Random(args.seed)
    all_results: list[TrialResult] = []
    total_start = time.perf_counter()

    # Coarse phase
    print("=" * 62)
    print(f"  PHASE C — Coarse search  ({coarse_trials} trials, {coarse_epochs} epochs each)")
    print("=" * 62)
    coarse_results = _run_phase(
        phase_label="C", num_trials=coarse_trials,
        search_space=SEARCH_SPACE_CNN_LSTM_RECONS,
        data_dir=args.data_dir, config_path=args.recons_config,
        trial_sessions=args.trial_sessions, trial_epochs=coarse_epochs,
        window_length=args.window_length, batch_size=args.batch_size,
        device=device, timeout_secs=timeout_secs,
        early_stopping_patience=args.early_stopping_patience,
        rng=rng, trial_idx_offset=0,
    )
    all_results.extend(coarse_results)

    # Fine phase
    fine_results: list[TrialResult] = []
    if args.search_mode == "two-phase" and coarse_results:
        top_k = min(args.fine_top_k, len(coarse_results))
        top_k_coarse = sorted(coarse_results, key=lambda x: x[3])[:top_k]
        print(f"\n{'='*62}")
        print(f"  PHASE F — Fine search  ({top_k} anchors × {args.fine_trials} trials, {args.fine_epochs} epochs each)")
        print(f"  Anchoring on top-{top_k} coarse configs:")
        for rank, (_, gidx, cfg, cer) in enumerate(top_k_coarse, 1):
            print(f"    Anchor {rank}: trial {gidx+1}  val_CER={cer:.2f}%")
        print("=" * 62)

        fine_trial_offset = coarse_trials
        for anchor_idx, (_, _, anchor_cfg, anchor_cer) in enumerate(top_k_coarse):
            fine_space = make_fine_search_space(anchor_cfg, SEARCH_SPACE_CNN_LSTM_RECONS, args.fine_shrink)
            label = f"F{anchor_idx+1}"
            print(f"\n  -- Anchor {anchor_idx+1}/{top_k}  (coarse CER={anchor_cer:.2f}%) --")
            phase_results = _run_phase(
                phase_label=label, num_trials=args.fine_trials,
                search_space=fine_space,
                data_dir=args.data_dir, config_path=args.recons_config,
                trial_sessions=args.trial_sessions, trial_epochs=args.fine_epochs,
                window_length=args.window_length, batch_size=args.batch_size,
                device=device, timeout_secs=timeout_secs,
                early_stopping_patience=args.early_stopping_patience,
                rng=rng, trial_idx_offset=fine_trial_offset,
            )
            fine_results.extend(phase_results)
            fine_trial_offset += args.fine_trials
            if len(phase_results) < args.fine_trials:
                break
        all_results.extend(fine_results)

    if not all_results:
        print("No trials completed — nothing to save.")
        return

    total_mins = (time.perf_counter() - total_start) / 60
    print(f"\nCompleted {len(all_results)} trials in {total_mins:.1f} min.")

    best_phase, best_global_idx, best_config, best_cer = min(all_results, key=lambda x: x[3])

    # Confirmation run
    if args.confirm_epochs > 0:
        print(f"\n{'='*62}")
        print(f"  CONFIRMATION RUN  ({args.confirm_epochs} epochs on overall best)")
        print(f"  Best so far: phase={best_phase}  trial={best_global_idx+1}  CER={best_cer:.2f}%")
        print("=" * 62)
        _print_config_inline(best_config)
        print()
        t0 = time.perf_counter()
        try:
            confirm_cer, _, _ = run_trial(
                trial_idx=10_000 + best_global_idx, config=best_config,
                data_dir=args.data_dir, config_path=args.recons_config,
                trial_sessions=args.trial_sessions, trial_epochs=args.confirm_epochs,
                window_length=args.window_length, batch_size=args.batch_size,
                device=device, timeout_secs=timeout_secs,
                early_stopping_patience=args.early_stopping_patience,
            )
        except KeyboardInterrupt:
            confirm_cer = best_cer
        print(f"  Confirmation CER={confirm_cer:.2f}%  ({time.perf_counter()-t0:.1f}s)")
        best_cer = confirm_cer

    _print_results_table(all_results)
    best_cer_str = "OOM" if math.isinf(best_cer) else f"{best_cer:.2f}%"
    print(f"\nBest config: phase={best_phase}  trial={best_global_idx+1}  val_CER={best_cer_str}")

    if math.isinf(best_cer):
        print("All trials OOM — no YAML saved.")
        return

    output_data: dict[str, Any] = {
        "lr":            float(best_config["lr"]),
        "weight_decay":  float(best_config["weight_decay"]),
        "proj_features": int(best_config["proj_features"]),
        "cnn_channels":  int(best_config["cnn_channels"]),
        "cnn_kernel":    int(best_config["cnn_kernel"]),
        "cnn_layers":    int(best_config["cnn_layers"]),
        "lstm_hidden":   int(best_config["lstm_hidden"]),
        "lstm_layers":   int(best_config["lstm_layers"]),
        "dropout":       float(best_config["dropout"]),
        "trial_val_cer":      round(float(best_cer), 4),
        "data_dir":           str(args.data_dir),
        "recons_config":      str(args.recons_config),
        "num_trial_sessions": args.trial_sessions,
        "window_length":      args.window_length,
        "search_mode":        args.search_mode,
        "search_phase_found": best_phase,
        "num_coarse_trials":  len(coarse_results),
        "num_fine_trials":    len(fine_results),
        "coarse_epochs":      coarse_epochs,
        "fine_epochs":        args.fine_epochs if args.search_mode == "two-phase" else None,
        "fine_shrink":        args.fine_shrink if args.search_mode == "two-phase" else None,
        "tuned_at":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_data = {k: v for k, v in output_data.items() if v is not None}

    with open(args.output, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"\nBest hyperparameters saved to: {args.output}")
    print("\nTo train with these hyperparameters run:")
    print(f"  python -m Playground_Mumbi.train_recons --from-hyperparams {args.output}")


if __name__ == "__main__":
    main()
