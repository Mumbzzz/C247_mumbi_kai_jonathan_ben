"""Hyperparameter tuner for the EMG-to-keystroke RNN model.

Uses random search over a defined search space.  Each trial runs a short
proxy training on the first N training sessions, evaluates on the fixed
validation session, and records the best val CER achieved during that trial.
After all trials the best configuration is saved to best_hyperparams.yaml,
which can be passed directly to train.py via --from-hyperparams.

Usage:
    python -m Playground_Kai.hyperparam_tuner [options]

Key flags (all have defaults):
    --num-trials        Random configurations to evaluate     (default: 50)
    --trial-epochs      Training epochs per proxy run         (default: 10)
    --trial-sessions    Training sessions per trial (≤16)     (default: 5)
    --output            Path to write best_hyperparams.yaml

After tuning, launch full training with the best config:
    python -m Playground_Kai.train --from-hyperparams Playground_Kai/checkpoints/best_hyperparams.yaml
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

from Playground_Kai.data_utils import build_loaders_from_paths, get_session_paths
from Playground_Kai.model import RNNEncoder, ConformerEncoder
from Playground_Kai.train import _lr_lambda, evaluate, train_one_epoch


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

# Each entry is either:
#   {"type": "log_uniform", "low": <float>, "high": <float>}  — sample exp(U[log(low), log(high)])
#   {"type": "uniform",     "low": <float>, "high": <float>}  — sample U[low, high]
#   {"type": "choice",      "choices": [<val>, ...]}          — sample uniformly from list
SEARCH_SPACE_RNN: dict[str, dict[str, Any]] = {
    "lr":           {"type": "log_uniform", "low": 1e-4,  "high": 1e-3},
    "hidden_size":  {"type": "choice",      "choices": [256, 384, 512, 768]},
    "num_layers":   {"type": "choice",      "choices": [1, 2, 3]},
    "dropout":      {"type": "uniform",     "low": 0.1,   "high": 0.5},
    "weight_decay": {"type": "log_uniform", "low": 1e-3,  "high": 1e-1},
}

# d_model choices are all divisible by both 4 and 8 so any num_heads value works.
SEARCH_SPACE_CONFORMER: dict[str, dict[str, Any]] = {
    "lr":              {"type": "log_uniform", "low": 1e-4,  "high": 1e-3},
    "d_model":         {"type": "choice",      "choices": [128, 192, 256, 384]},
    "num_heads":       {"type": "choice",      "choices": [4, 8]},
    "num_layers":      {"type": "choice",      "choices": [2, 4, 6]},
    "conv_kernel_size":{"type": "choice",      "choices": [15, 31]},
    "dropout":         {"type": "uniform",     "low": 0.1,   "high": 0.4},
    "weight_decay":    {"type": "log_uniform", "low": 1e-3,  "high": 1e-1},
}


def sample_config(rng: random.Random, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sample one random configuration from the given search space."""
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
) -> tuple[float, bool]:
    """Run one proxy training trial and return the best val CER achieved.

    Args:
        trial_idx: Used as the random seed for reproducibility.
        config: Hyperparameter dict sampled from the active search space.
        train_paths: HDF5 session paths for the proxy training set.
        val_paths: HDF5 session paths for validation (always the full val split).
        trial_epochs: Number of training epochs for this proxy run.
        window_length: Raw EMG samples per window.
        batch_size: Batch size.
        device: Torch device.
        model_type: ``"rnn"`` or ``"conformer"``.
        timeout_secs: Wall-clock budget in seconds.  Checked **between epochs**
            — not mid-epoch, since PyTorch CUDA kernels cannot be interrupted.
            If cumulative time exceeds this after an epoch, training stops early
            and the best CER seen so far is returned.  Default: no timeout.

    Returns:
        Tuple of (best_val_cer, timed_out) where timed_out is True if the trial
        was stopped early by the timeout.
    """
    torch.manual_seed(trial_idx)

    loaders = build_loaders_from_paths(
        train_paths=train_paths,
        val_paths=val_paths,
        window_length=window_length,
        batch_size=batch_size,
    )

    if model_type == "conformer":
        model = ConformerEncoder(
            in_features=528,
            mlp_features=(384,),
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            conv_kernel_size=config["conv_kernel_size"],
            dropout=config["dropout"],
        ).to(device)
    else:
        # Dropout has no effect with a single-layer LSTM
        effective_dropout = config["dropout"] if config["num_layers"] > 1 else 0.0
        model = RNNEncoder(
            in_features=528,
            mlp_features=(384,),
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=effective_dropout,
        ).to(device)

    criterion = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    steps_per_epoch = len(loaders["train"])
    total_steps = trial_epochs * steps_per_epoch
    # Warmup fraction matches full training (~10% of steps), scaled to trial length
    warmup_steps = max(1, trial_epochs // 5) * steps_per_epoch
    # Fix min_lr at 2% of peak for all trials
    min_lr_ratio = 0.02

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps, min_lr_ratio),
    )

    decoder = CTCGreedyDecoder()
    best_val_cer = float("inf")
    timed_out = False
    t_trial_start = time.perf_counter()

    for _ in range(trial_epochs):
        train_one_epoch(model, loaders["train"], optimizer, criterion, device, scheduler)
        _, val_metrics = evaluate(model, loaders["val"], device, decoder)
        val_cer = val_metrics["CER"]
        if val_cer < best_val_cer:
            best_val_cer = val_cer
        # Timeout is checked between epochs — cannot interrupt mid-epoch safely.
        if time.perf_counter() - t_trial_start > timeout_secs:
            timed_out = True
            break

    # Release GPU memory before the next trial
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best_val_cer, timed_out


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _print_results_table(results: list[tuple[int, dict, float]], model_type: str) -> None:
    sorted_results = sorted(results, key=lambda x: x[2])
    if model_type == "conformer":
        col = 102
        print("\n" + "=" * col)
        print(
            f"{'Rank':>4}  {'Trial':>5}  {'val_CER':>8}  {'lr':>10}  "
            f"{'d_model':>7}  {'heads':>5}  {'layers':>6}  "
            f"{'conv_k':>6}  {'dropout':>7}  {'weight_decay':>12}"
        )
        print("-" * col)
        for rank, (trial_idx, cfg, cer) in enumerate(sorted_results, 1):
            print(
                f"{rank:>4}  {trial_idx + 1:>5}  {cer:>7.2f}%  "
                f"{cfg['lr']:>10.2e}  {cfg['d_model']:>7}  {cfg['num_heads']:>5}  "
                f"{cfg['num_layers']:>6}  {cfg['conv_kernel_size']:>6}  "
                f"{cfg['dropout']:>7.3f}  {cfg['weight_decay']:>12.2e}"
            )
        print("=" * col)
    else:
        col = 90
        print("\n" + "=" * col)
        print(
            f"{'Rank':>4}  {'Trial':>5}  {'val_CER':>8}  {'lr':>10}  "
            f"{'hidden':>6}  {'layers':>6}  {'dropout':>7}  {'weight_decay':>12}"
        )
        print("-" * col)
        for rank, (trial_idx, cfg, cer) in enumerate(sorted_results, 1):
            print(
                f"{rank:>4}  {trial_idx + 1:>5}  {cer:>7.2f}%  "
                f"{cfg['lr']:>10.2e}  {cfg['hidden_size']:>6}  "
                f"{cfg['num_layers']:>6}  {cfg['dropout']:>7.3f}  "
                f"{cfg['weight_decay']:>12.2e}"
            )
        print("=" * col)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Random-search hyperparameter tuner for the EMG RNN/Conformer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=["rnn", "conformer"], default="rnn",
                   help="Model architecture to tune")
    p.add_argument("--data-root", type=Path, default=_ROOT / "data",
                   help="Directory containing *.hdf5 session files")
    p.add_argument("--config", type=Path,
                   default=_ROOT / "config" / "user" / "single_user.yaml",
                   help="Path to the train/val/test split YAML")
    p.add_argument("--output", type=Path, default=None,
                   help="Path to write the best hyperparameters YAML "
                        "(default: checkpoints/best_hyperparams_{model}.yaml)")
    p.add_argument("--num-trials", type=int, default=50,
                   help="Number of random configurations to evaluate")
    p.add_argument("--trial-epochs", type=int, default=10,
                   help="Training epochs per proxy run")
    p.add_argument("--trial-sessions", type=int, default=5,
                   help="Number of training sessions to use per trial (first N from split YAML)")
    p.add_argument("--trial-timeout", type=float, default=180.0,
                   help="Per-trial wall-clock timeout in seconds.  Checked between "
                        "epochs — trials that exceed this are stopped early and "
                        "scored on their best CER so far.  Set 0 to disable.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--window-length", type=int, default=8000,
                   help="Raw EMG samples per training window (8000 = 4 s @ 2 kHz)")
    p.add_argument("--seed", type=int, default=42,
                   help="Global random seed for reproducibility")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device            : {device}")
    print(f"Model             : {args.model}")

    if args.output is None:
        args.output = (
            Path(__file__).resolve().parent / "checkpoints" / f"best_hyperparams_{args.model}.yaml"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Parse session splits
    session_paths = get_session_paths(data_root=args.data_root, config_path=args.config)
    all_train_paths = session_paths["train"]
    val_paths       = session_paths["val"]

    if args.trial_sessions > len(all_train_paths):
        raise ValueError(
            f"--trial-sessions={args.trial_sessions} exceeds available "
            f"train sessions ({len(all_train_paths)})"
        )
    # Always use the first N sessions (chronological order from YAML)
    # so that trial results are directly comparable across runs.
    trial_train_paths = all_train_paths[: args.trial_sessions]

    active_space = SEARCH_SPACE_CONFORMER if args.model == "conformer" else SEARCH_SPACE_RNN
    print(f"Train sessions    : {len(trial_train_paths)} / {len(all_train_paths)}")
    print(f"Val sessions      : {len(val_paths)}")
    print(f"Trials × epochs   : {args.num_trials} × {args.trial_epochs}")
    timeout_secs = float("inf") if args.trial_timeout == 0 else args.trial_timeout
    if timeout_secs != float("inf"):
        print(f"Trial timeout     : {timeout_secs:.0f}s (checked between epochs)")
    else:
        print(f"Trial timeout     : none")
    print(f"Search params     : {list(active_space.keys())}")
    print("Press Ctrl+C to stop early — completed trials will still be ranked.\n")

    rng = random.Random(args.seed)
    results: list[tuple[int, dict, float]] = []
    total_start = time.perf_counter()

    for trial_idx in range(args.num_trials):
        config = sample_config(rng, active_space)

        if args.model == "conformer":
            print(
                f"Trial {trial_idx + 1:>3}/{args.num_trials}"
                f"  lr={config['lr']:.2e}"
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
                f"Trial {trial_idx + 1:>3}/{args.num_trials}"
                f"  lr={config['lr']:.2e}"
                f"  hidden={config['hidden_size']}"
                f"  layers={config['num_layers']}"
                f"  dropout={config['dropout']:.3f}"
                f"  wd={config['weight_decay']:.2e}",
                end="  ",
                flush=True,
            )

        t0 = time.perf_counter()
        try:
            val_cer, timed_out = run_trial(
                trial_idx=trial_idx,
                config=config,
                train_paths=trial_train_paths,
                val_paths=val_paths,
                trial_epochs=args.trial_epochs,
                window_length=args.window_length,
                batch_size=args.batch_size,
                device=device,
                model_type=args.model,
                timeout_secs=timeout_secs,
            )
        except KeyboardInterrupt:
            print(f"\n[Interrupted] Stopping after {len(results)} completed trials.")
            break

        elapsed = time.perf_counter() - t0
        timeout_tag = "  [timeout]" if timed_out else ""
        print(f"-> val_CER={val_cer:.2f}%  ({elapsed:.1f}s){timeout_tag}")
        results.append((trial_idx, config, val_cer))

    if not results:
        print("No trials completed — nothing to save.")
        return

    total_mins = (time.perf_counter() - total_start) / 60
    print(f"\nCompleted {len(results)} / {args.num_trials} trials in {total_mins:.1f} min.")

    _print_results_table(results, args.model)

    _, best_config, best_cer = min(results, key=lambda x: x[2])
    best_trial_display = results.index(min(results, key=lambda x: x[2])) + 1
    print(f"\nBest config: trial {best_trial_display}  val_CER={best_cer:.2f}%")

    # Save as YAML — use str(Path) output path for human-readable comment
    if args.model == "conformer":
        model_hp: dict[str, Any] = {
            "d_model":          int(best_config["d_model"]),
            "num_heads":        int(best_config["num_heads"]),
            "num_layers":       int(best_config["num_layers"]),
            "conv_kernel_size": int(best_config["conv_kernel_size"]),
            "dropout":          float(best_config["dropout"]),
        }
    else:
        model_hp = {
            "hidden_size": int(best_config["hidden_size"]),
            "num_layers":  int(best_config["num_layers"]),
            "dropout":     float(best_config["dropout"]),
        }
    output_data: dict[str, Any] = {
        "lr":           float(best_config["lr"]),
        "weight_decay": float(best_config["weight_decay"]),
        **model_hp,
        # Metadata (ignored by train.py, useful for auditing)
        "trial_val_cer":      round(float(best_cer), 4),
        "num_trial_epochs":   args.trial_epochs,
        "num_trial_sessions": args.trial_sessions,
        "num_trials_run":     len(results),
        "tuned_at":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(args.output, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"\nBest hyperparameters saved to: {args.output}")
    print("\nTo train with these hyperparameters run:")
    print(f"  .venv\\Scripts\\python.exe -m Playground_Kai.train --model {args.model} --from-hyperparams {args.output}")


if __name__ == "__main__":
    main()
