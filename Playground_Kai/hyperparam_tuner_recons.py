"""Hyperparameter tuner for the recons EMG-to-keystroke RNN/Conformer model.

Uses reconstructed EMG data from ``data_recons/*_recons_v3.hdf5`` (62.5 Hz).
Identical two-phase search strategy as ``hyperparam_tuner.py`` but defaults
window length to 500 frames (≈ 8 s at 62.5 Hz) and writes YAML / checkpoint
files with ``_recons_raw`` naming so they do not collide with vanilla outputs.

Supports two search modes:

  two-phase (default)
    Phase C (coarse): broad random search over the full search space using a
    short proxy training run per trial.
    Phase F (fine):   for each of the top-K coarse configs, a tighter random
    search is launched around that config.

  coarse-only
    Classic single-phase random search.

Usage:
    python -m Playground_Kai.hyperparam_tuner_recons [options]

Key flags (all have defaults):
    --search-mode       two-phase | coarse-only                 (default: two-phase)
    --coarse-trials     Configs evaluated in coarse phase       (default: 20)
    --coarse-epochs     Epochs per coarse proxy run             (default: 8)
    --fine-trials       Configs evaluated per fine anchor       (default: 10)
    --fine-epochs       Epochs per fine proxy run               (default: 15)
    --fine-top-k        Top coarse configs used as fine anchors (default: 3)
    --fine-shrink       Log-scale shrink factor for fine bounds (default: 3.0)
    --confirm-epochs    Re-run best config with N more epochs   (default: 0 = off)
    --early-stopping-patience  Stop trial after N non-improving epochs
                                                                (default: 0 = off)
    --trial-sessions    Training sessions per trial (<=16)      (default: 5)
    --trial-timeout     Per-trial wall-clock cap in seconds     (default: 180s)

After tuning, launch full training with the best config:
    python -m Playground_Kai.train_recons --model {rnn,conformer} \\
        --from-hyperparams Playground_Kai/checkpoints/best_hyperparams_{model}_recons_raw.yaml
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

from Playground_Kai.data_utils import build_recons_loaders_from_paths, get_recons_session_paths
from Playground_Kai.model import ReconsRNNEncoder, ReconsConformerEncoder

_OOM_ERRORS: tuple[type[BaseException], ...] = (torch.cuda.OutOfMemoryError,)
if hasattr(torch, "AcceleratorError"):
    _OOM_ERRORS = _OOM_ERRORS + (torch.AcceleratorError,)  # type: ignore[attr-defined]
from Playground_Kai.train_recons import _lr_lambda, evaluate, train_one_epoch


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

SEARCH_SPACE_RNN: dict[str, dict[str, Any]] = {
    "lr":           {"type": "log_uniform", "low": 1e-4,  "high": 1e-3},
    "proj_dim":     {"type": "choice",      "choices": [64, 128, 256]},
    "hidden_size":  {"type": "choice",      "choices": [256, 512, 768]},
    "num_layers":   {"type": "choice",      "choices": [1, 2, 3]},
    "dropout":      {"type": "uniform",     "low": 0.0,   "high": 0.4},
    "weight_decay": {"type": "log_uniform", "low": 1e-3,  "high": 1e-1},
}

SEARCH_SPACE_CONFORMER: dict[str, dict[str, Any]] = {
    "lr":              {"type": "log_uniform", "low": 1e-4,  "high": 1e-3},
    "proj_dim":        {"type": "choice",      "choices": [64, 128, 256]},
    "d_model":         {"type": "choice",      "choices": [128, 192, 256, 384]},
    "num_heads":       {"type": "choice",      "choices": [4, 8]},
    "num_layers":      {"type": "choice",      "choices": [2, 4, 6]},
    "conv_kernel_size":{"type": "choice",      "choices": [15, 31]},
    "dropout":         {"type": "uniform",     "low": 0.0,   "high": 0.4},
    "weight_decay":    {"type": "log_uniform", "low": 1e-3,  "high": 1e-1},
}


def sample_config(rng: random.Random, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sample one random configuration from *search_space*."""
    config: dict[str, Any] = {}
    for name, spec in search_space.items():
        if spec["type"] == "log_uniform":
            log_val = rng.uniform(math.log(spec["low"]), math.log(spec["high"]))
            config[name] = math.exp(log_val)
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
    """Build a narrowed search space centred on *base_config*."""
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
    train_paths: list[Path],
    val_paths: list[Path],
    trial_epochs: int,
    window_length: int,
    batch_size: int,
    device: torch.device,
    model_type: str = "rnn",
    timeout_secs: float = float("inf"),
    early_stopping_patience: int = 0,
) -> tuple[float, bool, int]:
    """Run one proxy training trial on recons data.

    Returns:
        ``(best_val_cer, timed_out, epochs_run)``
    """
    torch.manual_seed(trial_idx)

    loaders = build_recons_loaders_from_paths(
        train_paths=train_paths,
        val_paths=val_paths,
        window_length=window_length,
        batch_size=batch_size,
    )

    if model_type == "conformer":
        model = ReconsConformerEncoder(
            in_channels=16,
            proj_dim=config["proj_dim"],
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            conv_kernel_size=config["conv_kernel_size"],
            dropout=config["dropout"],
        ).to(device)
    else:
        model = ReconsRNNEncoder(
            in_channels=16,
            proj_dim=config["proj_dim"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(device)

    criterion = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    steps_per_epoch = len(loaders["train"])
    total_steps = trial_epochs * steps_per_epoch
    warmup_steps = max(1, trial_epochs // 5) * steps_per_epoch
    min_lr_ratio = 0.02

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps, min_lr_ratio),
    )

    decoder = CTCGreedyDecoder()
    best_val_cer = float("inf")
    timed_out = False
    epochs_run = 0
    no_improve_streak = 0
    t_trial_start = time.perf_counter()

    try:
        for _ in range(trial_epochs):
            train_one_epoch(model, loaders["train"], optimizer, criterion, device, scheduler)
            _, val_metrics = evaluate(model, loaders["val"], device, decoder)
            val_cer = val_metrics["CER"]
            epochs_run += 1
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                no_improve_streak = 0
            else:
                no_improve_streak += 1

            if early_stopping_patience > 0 and no_improve_streak >= early_stopping_patience:
                break
            if time.perf_counter() - t_trial_start > timeout_secs:
                timed_out = True
                break

    except _OOM_ERRORS as e:
        if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e).lower():
            raise
        print("\n    [OOM — skipping config]", end="  ", flush=True)
        best_val_cer = float("inf")

    finally:
        del optimizer
        del scheduler
        del model
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


def _print_config_inline(config: dict[str, Any], model_type: str) -> None:
    if model_type == "conformer":
        print(
            f"  lr={config['lr']:.2e}"
            f"  proj={config['proj_dim']}"
            f"  d={config['d_model']}"
            f"  h={config['num_heads']}"
            f"  layers={config['num_layers']}"
            f"  k={config['conv_kernel_size']}"
            f"  do={config['dropout']:.3f}"
            f"  wd={config['weight_decay']:.2e}",
            end="  ",
            flush=True,
        )
    else:
        print(
            f"  lr={config['lr']:.2e}"
            f"  proj={config['proj_dim']}"
            f"  hidden={config['hidden_size']}"
            f"  layers={config['num_layers']}"
            f"  dropout={config['dropout']:.3f}"
            f"  wd={config['weight_decay']:.2e}",
            end="  ",
            flush=True,
        )


def _run_phase(
    phase_label: str,
    num_trials: int,
    search_space: dict[str, dict[str, Any]],
    trial_train_paths: list[Path],
    val_paths: list[Path],
    trial_epochs: int,
    window_length: int,
    batch_size: int,
    device: torch.device,
    model_type: str,
    timeout_secs: float,
    early_stopping_patience: int,
    rng: random.Random,
    trial_idx_offset: int,
) -> list[TrialResult]:
    """Run *num_trials* proxy trials and return all results."""
    results: list[TrialResult] = []
    for local_idx in range(num_trials):
        global_idx = trial_idx_offset + local_idx
        config = sample_config(rng, search_space)
        print(
            f"[{phase_label}] Trial {local_idx + 1:>3}/{num_trials}",
            end="",
            flush=True,
        )
        _print_config_inline(config, model_type)

        t0 = time.perf_counter()
        try:
            val_cer, timed_out, epochs_run = run_trial(
                trial_idx=global_idx,
                config=config,
                train_paths=trial_train_paths,
                val_paths=val_paths,
                trial_epochs=trial_epochs,
                window_length=window_length,
                batch_size=batch_size,
                device=device,
                model_type=model_type,
                timeout_secs=timeout_secs,
                early_stopping_patience=early_stopping_patience,
            )
        except KeyboardInterrupt:
            print(
                f"\n[Interrupted] Phase [{phase_label}] stopped after "
                f"{len(results)} completed trials."
            )
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

def _print_results_table(results: list[TrialResult], model_type: str) -> None:
    sorted_results = sorted(results, key=lambda x: x[3])
    if model_type == "conformer":
        col = 121
        print("\n" + "=" * col)
        print(
            f"{'Rank':>4}  {'Phase':>5}  {'Trial':>5}  {'val_CER':>8}  {'lr':>10}  "
            f"{'proj':>4}  {'d_model':>7}  {'heads':>5}  {'layers':>6}  "
            f"{'conv_k':>6}  {'dropout':>7}  {'weight_decay':>12}"
        )
        print("-" * col)
        for rank, (phase, global_idx, cfg, cer) in enumerate(sorted_results, 1):
            cer_str = "   OOM  " if math.isinf(cer) else f"{cer:>7.2f}%"
            print(
                f"{rank:>4}  {phase:>5}  {global_idx + 1:>5}  {cer_str}  "
                f"{cfg['lr']:>10.2e}  {cfg['proj_dim']:>4}  {cfg['d_model']:>7}  {cfg['num_heads']:>5}  "
                f"{cfg['num_layers']:>6}  {cfg['conv_kernel_size']:>6}  "
                f"{cfg['dropout']:>7.3f}  {cfg['weight_decay']:>12.2e}"
            )
        print("=" * col)
    else:
        col = 109
        print("\n" + "=" * col)
        print(
            f"{'Rank':>4}  {'Phase':>5}  {'Trial':>5}  {'val_CER':>8}  {'lr':>10}  "
            f"{'proj':>4}  {'hidden':>6}  {'layers':>6}  {'dropout':>7}  {'weight_decay':>12}"
        )
        print("-" * col)
        for rank, (phase, global_idx, cfg, cer) in enumerate(sorted_results, 1):
            cer_str = "   OOM  " if math.isinf(cer) else f"{cer:>7.2f}%"
            print(
                f"{rank:>4}  {phase:>5}  {global_idx + 1:>5}  {cer_str}  "
                f"{cfg['lr']:>10.2e}  {cfg['proj_dim']:>4}  {cfg['hidden_size']:>6}  "
                f"{cfg['num_layers']:>6}  {cfg['dropout']:>7.3f}  "
                f"{cfg['weight_decay']:>12.2e}"
            )
        print("=" * col)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-phase hyperparameter tuner for the recons EMG RNN/Conformer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Architecture ---
    p.add_argument("--model", choices=["rnn", "conformer"], default="rnn",
                   help="Model architecture to tune")
    # --- Data ---
    p.add_argument("--data-root", type=Path, default=_ROOT / "data_recons",
                   help="Directory containing *_recons_v3.hdf5 session files")
    p.add_argument("--config", type=Path,
                   default=_ROOT / "config" / "user" / "single_user.yaml",
                   help="Path to the train/val/test split YAML")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write the best hyperparameters YAML "
                        "(default: checkpoints/best_hyperparams_{model}_recons.yaml)")
    p.add_argument("--trial-sessions", type=int, default=5,
                   help="Training sessions per trial (first N from split YAML, max 16)")
    # --- Search mode ---
    p.add_argument("--search-mode", choices=["two-phase", "coarse-only"],
                   default="two-phase",
                   help="two-phase: coarse then fine; coarse-only: classic random search")
    # --- Coarse phase ---
    p.add_argument("--coarse-trials", type=int, default=None,
                   help="Configs evaluated in coarse phase (overrides --num-trials)")
    p.add_argument("--coarse-epochs", type=int, default=None,
                   help="Epochs per coarse proxy run (overrides --trial-epochs)")
    p.add_argument("--num-trials", type=int, default=20,
                   help="Alias for --coarse-trials (backward compat, default: 20)")
    p.add_argument("--trial-epochs", type=int, default=8,
                   help="Alias for --coarse-epochs (backward compat, default: 8)")
    # --- Fine phase ---
    p.add_argument("--fine-trials", type=int, default=10,
                   help="Configs evaluated per fine-phase anchor")
    p.add_argument("--fine-epochs", type=int, default=15,
                   help="Training epochs per fine proxy run")
    p.add_argument("--fine-top-k", type=int, default=3,
                   help="Number of top coarse configs used as fine-phase anchors")
    p.add_argument("--fine-shrink", type=float, default=3.0,
                   help="Shrink factor for continuous param bounds in fine phase.")
    # --- Confirmation run ---
    p.add_argument("--confirm-epochs", type=int, default=0,
                   help="If >0, re-run the overall best config with this many epochs. 0 = off.")
    # --- Per-trial settings ---
    p.add_argument("--trial-timeout", type=float, default=180.0,
                   help="Per-trial wall-clock cap in seconds. Set 0 to disable.")
    p.add_argument("--early-stopping-patience", type=int, default=0,
                   help="Stop a trial after this many consecutive non-improving "
                        "validation epochs. 0 = off.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--window-length", type=int, default=500,
                   help="Recons EMG samples per training window (500 = 8 s @ 62.5 Hz)")
    p.add_argument("--seed", type=int, default=42,
                   help="Global random seed for reproducibility")
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
        _yaml_stem = f"best_hyperparams_{args.model}_recons_raw"
        args.output = Path(__file__).resolve().parent / "checkpoints" / f"{_yaml_stem}.yaml"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    session_paths = get_recons_session_paths(data_root=args.data_root, config_path=args.config)
    all_train_paths = session_paths["train"]
    val_paths       = session_paths["val"]

    if args.trial_sessions > len(all_train_paths):
        raise ValueError(
            f"--trial-sessions={args.trial_sessions} exceeds available "
            f"train sessions ({len(all_train_paths)})"
        )
    trial_train_paths = all_train_paths[: args.trial_sessions]
    active_space = SEARCH_SPACE_CONFORMER if args.model == "conformer" else SEARCH_SPACE_RNN

    # ---- Startup banner ----
    print(f"Device            : {device}")
    print(f"Model             : {args.model}")
    print(f"Pipeline          : recons-raw (data_recons/, raw channels 16ch×2, 62.5 Hz)")
    print(f"Search mode       : {args.search_mode}")
    print(f"Train sessions    : {len(trial_train_paths)} / {len(all_train_paths)}")
    print(f"Val sessions      : {len(val_paths)}")
    if args.search_mode == "two-phase":
        top_k_display = min(args.fine_top_k, coarse_trials)
        print(f"Coarse            : {coarse_trials} trials × {coarse_epochs} epochs")
        print(
            f"Fine              : {top_k_display} anchors "
            f"× {args.fine_trials} trials × {args.fine_epochs} epochs"
        )
        print(f"Fine shrink       : {args.fine_shrink:.1f}x (continuous params)")
    else:
        print(f"Trials × epochs   : {coarse_trials} × {coarse_epochs}")
    if timeout_secs != float("inf"):
        print(f"Trial timeout     : {timeout_secs:.0f}s (checked between epochs)")
    else:
        print(f"Trial timeout     : none")
    if args.early_stopping_patience > 0:
        print(f"Early stopping    : patience={args.early_stopping_patience}")
    if args.confirm_epochs > 0:
        print(f"Confirmation run  : {args.confirm_epochs} epochs on overall best")
    print(f"Search params     : {list(active_space.keys())}")
    print("Press Ctrl+C to stop early — completed trials will still be ranked.\n")

    rng = random.Random(args.seed)
    all_results: list[TrialResult] = []
    total_start = time.perf_counter()

    # ---- Coarse phase ----
    print("=" * 62)
    print(f"  PHASE C — Coarse search  ({coarse_trials} trials, {coarse_epochs} epochs each)")
    print("=" * 62)
    coarse_results = _run_phase(
        phase_label="C",
        num_trials=coarse_trials,
        search_space=active_space,
        trial_train_paths=trial_train_paths,
        val_paths=val_paths,
        trial_epochs=coarse_epochs,
        window_length=args.window_length,
        batch_size=args.batch_size,
        device=device,
        model_type=args.model,
        timeout_secs=timeout_secs,
        early_stopping_patience=args.early_stopping_patience,
        rng=rng,
        trial_idx_offset=0,
    )
    all_results.extend(coarse_results)

    # ---- Fine phase (two-phase mode only) ----
    fine_results: list[TrialResult] = []
    if args.search_mode == "two-phase" and coarse_results:
        top_k = min(args.fine_top_k, len(coarse_results))
        top_k_coarse = sorted(coarse_results, key=lambda x: x[3])[:top_k]
        print(f"\n{'=' * 62}")
        print(
            f"  PHASE F — Fine search  "
            f"({top_k} anchors × {args.fine_trials} trials, {args.fine_epochs} epochs each)"
        )
        print(f"  Anchoring on top-{top_k} coarse configs:")
        for rank, (_, gidx, cfg, cer) in enumerate(top_k_coarse, 1):
            print(f"    Anchor {rank}: trial {gidx + 1}  val_CER={cer:.2f}%")
        print("=" * 62)

        fine_trial_offset = coarse_trials
        for anchor_idx, (_, _, anchor_cfg, anchor_cer) in enumerate(top_k_coarse):
            fine_space = make_fine_search_space(anchor_cfg, active_space, args.fine_shrink)
            label = f"F{anchor_idx + 1}"
            print(f"\n  -- Anchor {anchor_idx + 1}/{top_k}  (coarse CER={anchor_cer:.2f}%) --")
            phase_results = _run_phase(
                phase_label=label,
                num_trials=args.fine_trials,
                search_space=fine_space,
                trial_train_paths=trial_train_paths,
                val_paths=val_paths,
                trial_epochs=args.fine_epochs,
                window_length=args.window_length,
                batch_size=args.batch_size,
                device=device,
                model_type=args.model,
                timeout_secs=timeout_secs,
                early_stopping_patience=args.early_stopping_patience,
                rng=rng,
                trial_idx_offset=fine_trial_offset,
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

    # ---- Confirmation run ----
    best_phase, best_global_idx, best_config, best_cer = min(all_results, key=lambda x: x[3])
    if args.confirm_epochs > 0:
        print(f"\n{'=' * 62}")
        print(f"  CONFIRMATION RUN  ({args.confirm_epochs} epochs on overall best)")
        print(
            f"  Best so far: phase={best_phase}  trial={best_global_idx + 1}"
            f"  CER={best_cer:.2f}%"
        )
        print("=" * 62)
        _print_config_inline(best_config, args.model)
        print()
        confirm_seed = 10_000 + best_global_idx
        t0 = time.perf_counter()
        try:
            confirm_cer, _, confirm_epochs_run = run_trial(
                trial_idx=confirm_seed,
                config=best_config,
                train_paths=trial_train_paths,
                val_paths=val_paths,
                trial_epochs=args.confirm_epochs,
                window_length=args.window_length,
                batch_size=args.batch_size,
                device=device,
                model_type=args.model,
                timeout_secs=timeout_secs,
                early_stopping_patience=args.early_stopping_patience,
            )
        except KeyboardInterrupt:
            confirm_cer = best_cer
            confirm_epochs_run = 0
        elapsed = time.perf_counter() - t0
        print(
            f"  Confirmation CER={confirm_cer:.2f}%  "
            f"({elapsed:.1f}s, {confirm_epochs_run} epochs)"
        )
        best_cer = confirm_cer

    # ---- Results table ----
    _print_results_table(all_results, args.model)
    best_cer_str = "OOM" if math.isinf(best_cer) else f"{best_cer:.2f}%"
    print(
        f"\nBest config: phase={best_phase}  trial={best_global_idx + 1}"
        f"  val_CER={best_cer_str}"
    )

    if math.isinf(best_cer):
        print("All trials ran out of GPU memory — no YAML saved.")
        return

    # ---- Save YAML ----
    if args.model == "conformer":
        model_hp: dict[str, Any] = {
            "proj_dim":         int(best_config["proj_dim"]),
            "d_model":          int(best_config["d_model"]),
            "num_heads":        int(best_config["num_heads"]),
            "num_layers":       int(best_config["num_layers"]),
            "conv_kernel_size": int(best_config["conv_kernel_size"]),
            "dropout":          float(best_config["dropout"]),
        }
    else:
        model_hp = {
            "proj_dim":    int(best_config["proj_dim"]),
            "hidden_size": int(best_config["hidden_size"]),
            "num_layers":  int(best_config["num_layers"]),
            "dropout":     float(best_config["dropout"]),
        }

    output_data: dict[str, Any] = {
        "lr":           float(best_config["lr"]),
        "weight_decay": float(best_config["weight_decay"]),
        **model_hp,
        # Metadata
        "trial_val_cer":      round(float(best_cer), 4),
        "pipeline":           "recons-raw",
        "search_mode":        args.search_mode,
        "search_phase_found": best_phase,
        "num_coarse_trials":  len(coarse_results),
        "num_fine_trials":    len(fine_results),
        "coarse_epochs":      coarse_epochs,
        "fine_epochs":        args.fine_epochs if args.search_mode == "two-phase" else None,
        "num_trial_sessions": args.trial_sessions,
        "fine_shrink":        args.fine_shrink if args.search_mode == "two-phase" else None,
        "tuned_at":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_data = {k: v for k, v in output_data.items() if v is not None}

    with open(args.output, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"\nBest hyperparameters saved to: {args.output}")
    print("\nTo train with these hyperparameters run:")
    print(
        f"  .venv\\Scripts\\python.exe -m Playground_Kai.train_recons "
        f"--model {args.model} --from-hyperparams {args.output}"
    )


if __name__ == "__main__":
    main()
