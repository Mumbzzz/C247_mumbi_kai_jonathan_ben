"""
Plotting utilities for the emg2qwerty project.

Four plot types are available as subcommands:

  training_curves  -- loss and CER over epochs from a TensorBoard log dir
  emg_signal       -- raw sEMG channels for a short time window with keystroke markers
  compare          -- bar chart comparing final val/test CER across multiple runs
  channel_ablation -- line chart of CER vs. number of electrode channels used

Examples
--------
# Training curves from the most recent run
python scripts/plot_results.py training_curves \\
    --log_dir logs/2026-03-04/13-49-32

# EMG signal visualization (first 2.5 s of the val session)
python scripts/plot_results.py emg_signal \\
    --hdf5 data/2021-06-04-1622862148-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5 \\
    --duration 2.5

# Architecture comparison (add one --log_dir / --label pair per run)
python scripts/plot_results.py compare \\
    --log_dirs logs/run_tds logs/run_rnn \\
    --labels "TDS-Conv" "RNN"

# Channel ablation (supply your own channel counts and matching CER values)
python scripts/plot_results.py channel_ablation \\
    --channels 2 4 8 16 \\
    --val_cer  85 65 45 32 \\
    --test_cer 88 68 48 34
"""

import argparse
import glob
import sys
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# UCLA styling
# ---------------------------------------------------------------------------

UCLA_COLORS = [
    '#2774AE',  # UCLA Blue
    '#FFD100',  # UCLA Gold
    '#003B5C',  # Darkest Blue
    '#005587',  # Darker Blue
    '#8BB8EE',  # Lighter Blue
    '#FFB81C',  # Darkest Gold
]

GREY_BG = "#F2F2F2"  # UCLA Gray 5%

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.major.size': 0,
    'ytick.major.size': 0,
    'axes.linewidth': 0,
})


# ---------------------------------------------------------------------------
# Helper: read scalar series from a TensorBoard event file
# ---------------------------------------------------------------------------

def _read_tb_scalars(log_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Return {tag: [(step, value), ...]} from all event files under log_dir."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
        SCALARS,
    )

    # Search recursively for event files (lightning puts them under version_0/)
    pattern = str(log_dir / "**" / "events.out.tfevents.*")
    event_files = sorted(glob.glob(pattern, recursive=True))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {log_dir}")

    merged: dict[str, list[tuple[int, float]]] = {}
    for ef in event_files:
        ea = EventAccumulator(ef, size_guidance={SCALARS: 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            events = [(e.step, e.value) for e in ea.Scalars(tag)]
            if tag not in merged:
                merged[tag] = events
            else:
                merged[tag].extend(events)

    # Sort each series by step
    for tag in merged:
        merged[tag].sort(key=lambda x: x[0])

    return merged


def _steps_values(series: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    steps = np.array([s for s, _ in series])
    values = np.array([v for _, v in series])
    return steps, values


def _to_epochs(
    steps: np.ndarray,
    epoch_series: list[tuple[int, float]] | None,
) -> np.ndarray:
    """Convert step indices to epoch numbers using the logged 'epoch' scalar.
    Falls back to steps if epoch data is unavailable."""
    if not epoch_series:
        return steps
    ep_steps = np.array([s for s, _ in epoch_series])
    ep_vals  = np.array([v for _, v in epoch_series])
    # Interpolate epoch value at each requested step
    return np.interp(steps, ep_steps, ep_vals)


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(log_dir: Path, out_path: Path | None) -> None:
    print(f"Reading TensorBoard logs from {log_dir} ...")
    scalars = _read_tb_scalars(log_dir)

    available = set(scalars.keys())
    print(f"  Available tags: {sorted(available)}")

    epoch_series = scalars.get("epoch")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Curves", fontsize=14)

    # -- Loss --
    ax = axes[0]
    for tag, label, color in [
        ("train/loss", "Train loss", UCLA_COLORS[0]),
        ("val/loss",   "Val loss",   UCLA_COLORS[1]),
    ]:
        if tag in scalars:
            steps, vals = _steps_values(scalars[tag])
            epochs = _to_epochs(steps, epoch_series)
            ax.plot(epochs, vals, label=label, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CTC Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- CER --
    ax = axes[1]
    for tag, label, color in [
        ("train/CER", "Train CER", UCLA_COLORS[0]),
        ("val/CER",   "Val CER",   UCLA_COLORS[1]),
        ("test/CER",  "Test CER",  UCLA_COLORS[2]),
    ]:
        if tag in scalars:
            steps, vals = _steps_values(scalars[tag])
            epochs = _to_epochs(steps, epoch_series)
            ax.plot(epochs, vals, label=label, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CER (%)")
    ax.set_title("Character Error Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, out_path, "training_curves.png")


# ---------------------------------------------------------------------------
# 2. EMG signal visualization
# ---------------------------------------------------------------------------

def plot_emg_signal(hdf5_path: Path, duration: float, out_path: Path | None) -> None:
    print(f"Loading EMG data from {hdf5_path} ...")

    with h5py.File(hdf5_path, "r") as f:
        grp = f["emg2qwerty"]
        ts_ds = grp["timeseries"]

        # Pull the first `duration` seconds worth of samples
        timestamps = ts_ds["time"][:]
        t0 = timestamps[0]
        t1 = t0 + duration
        mask = timestamps <= t1
        n = int(mask.sum())

        emg_left  = ts_ds["emg_left"][:n]   # (T, 16)
        emg_right = ts_ds["emg_right"][:n]  # (T, 16)
        times = timestamps[:n] - t0          # relative seconds

        # Keystrokes in this window
        import json
        keystrokes_raw = json.loads(grp.attrs.get("keystrokes", "[]"))
        keystrokes = [
            k for k in keystrokes_raw
            if t0 <= k["start"] <= t1
        ]

    n_channels = emg_left.shape[1]  # 16

    # Normalise each channel for display (subtract mean, divide by global std)
    def _normalise(emg):
        emg = emg - emg.mean(axis=0, keepdims=True)
        scale = emg.std() or 1.0
        return emg / scale

    left_norm  = _normalise(emg_left)
    right_norm = _normalise(emg_right)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("sEMG Signal Visualization", fontsize=14)

    spacing = 4.0  # vertical spacing between channels

    for ax, emg_norm, hand, color in [
        (axes[0], right_norm, "Right hand", "tab:blue"),
        (axes[1], left_norm,  "Left hand",  "tab:orange"),
    ]:
        for ch in range(n_channels):
            offset = ch * spacing
            ax.plot(times, emg_norm[:, ch] + offset, color=color, lw=0.5, alpha=0.8)

        # Keystroke markers (convert absolute unix time -> relative seconds)
        for ks in keystrokes:
            t_rel = ks["start"] - t0
            if not (0 <= t_rel <= duration):
                continue
            ax.axvline(t_rel, color="black", lw=0.8, alpha=0.5, linestyle="--")
            key_label = ks.get("key", "")
            if key_label.startswith("'") and key_label.endswith("'"):
                key_label = key_label[1:-1]
            elif key_label == "Key.space":
                key_label = "␣"
            ax.annotate(
                key_label, xy=(t_rel, ax.get_ylim()[1]),
                xytext=(t_rel, (n_channels - 1) * spacing + spacing * 0.8),
                ha="center", va="bottom", fontsize=7, color="black",
            )

        ax.set_ylabel(hand)
        ax.set_yticks(np.arange(n_channels) * spacing)
        ax.set_yticklabels([f"ch{i+1}" for i in range(n_channels)], fontsize=7)
        ax.grid(axis="x", alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    _save_or_show(fig, out_path, "emg_signal.png")


# ---------------------------------------------------------------------------
# 3. Architecture comparison
# ---------------------------------------------------------------------------

def plot_compare(
    log_dirs: list[Path],
    labels: list[str],
    out_path: Path | None,
) -> None:
    if len(log_dirs) != len(labels):
        raise ValueError("--log_dirs and --labels must have the same length")

    val_cers, test_cers = [], []
    for ld in log_dirs:
        try:
            scalars = _read_tb_scalars(ld)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            val_cers.append(float("nan"))
            test_cers.append(float("nan"))
            continue

        def _last(tag):
            if tag in scalars:
                return scalars[tag][-1][1]
            return float("nan")

        val_cers.append(_last("val/CER"))
        test_cers.append(_last("test/CER"))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 5))
    bars_val  = ax.bar(x - width / 2, val_cers,  width, label="Val CER",  color=UCLA_COLORS[0])
    bars_test = ax.bar(x + width / 2, test_cers, width, label="Test CER", color=UCLA_COLORS[1])

    # Value labels on bars
    for bars in (bars_val, bars_test):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CER (%)")
    ax.set_title("Architecture Comparison — Character Error Rate")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    plt.tight_layout()
    _save_or_show(fig, out_path, "architecture_comparison.png")


# ---------------------------------------------------------------------------
# 4. Channel ablation
# ---------------------------------------------------------------------------

def plot_channel_ablation(
    channels: list[int],
    val_cer: list[float],
    test_cer: list[float],
    out_path: Path | None,
    labels: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(channels, val_cer,  "o-", label="Val CER",  color=UCLA_COLORS[0])
    ax.plot(channels, test_cer, "s-", label="Test CER", color=UCLA_COLORS[1])

    ax.set_xlabel("Number of electrode channels per hand")
    ax.set_ylabel("CER (%)")
    ax.set_title("Channel Ablation — CER vs. Electrode Channels")
    ax.set_xticks(channels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, color="#CCCCCC")
    plt.tight_layout()
    _save_or_show(fig, out_path, "channel_ablation.png")


def _best_val_cer(scalars: dict) -> float:
    if "val/CER" in scalars:
        return min(v for _, v in scalars["val/CER"])
    return float("nan")


def _last_test_cer(scalars: dict) -> float:
    if "test/CER" in scalars:
        return scalars["test/CER"][-1][1]
    return float("nan")


def plot_channel_ablation_from_dirs(
    log_dirs: list[Path],
    channels: list[int],
    labels: list[str],
    out_path: Path | None,
) -> None:
    val_cers, test_cers = [], []
    for ld in log_dirs:
        scalars = _read_tb_scalars(ld)
        val_cers.append(_best_val_cer(scalars))
        test_cers.append(_last_test_cer(scalars))
    plot_channel_ablation(channels, val_cers, test_cers, out_path, labels=labels)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, out_path: Path | None, default_name: str) -> None:
    if out_path is None:
        out_path = Path(default_name)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def plot_training_curves_overlay(
    log_dirs: list[Path],
    labels: list[str],
    out_path: Path | None,
) -> None:
    """Overlay val/CER training curves from multiple runs on one figure."""
    if len(log_dirs) != len(labels):
        raise ValueError("--log_dirs and --labels must have the same length")

    colors = UCLA_COLORS

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Training Curve Comparison", fontsize=13)

    for idx, (ld, label) in enumerate(zip(log_dirs, labels)):
        try:
            scalars = _read_tb_scalars(ld)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

        color = colors[idx % len(colors)]
        epoch_series = scalars.get("epoch")

        # Loss
        if "val/loss" in scalars:
            steps, vals = _steps_values(scalars["val/loss"])
            epochs = _to_epochs(steps, epoch_series)
            axes[0].plot(epochs, vals, label=label, color=color)
        # Val CER
        if "val/CER" in scalars:
            steps, vals = _steps_values(scalars["val/CER"])
            epochs = _to_epochs(steps, epoch_series)
            axes[1].plot(epochs, vals, label=label, color=color)
            # Mark final value
            axes[1].scatter(epochs[-1], vals[-1], color=color, s=50, zorder=5)
            axes[1].annotate(
                f"{vals[-1]:.1f}%",
                xy=(epochs[-1], vals[-1]),
                xytext=(8, 0), textcoords="offset points",
                fontsize=7.5, color=color, va="center",
            )

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val Loss")
    axes[0].set_title("Validation Loss"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val CER (%)")
    axes[1].set_title("Validation CER"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, out_path, "training_curves_overlay.png")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- training_curves ---
    p = sub.add_parser("training_curves", help="Plot loss and CER over training.")
    p.add_argument(
        "--log_dir", type=Path, required=True,
        help="Path to a Hydra run directory (e.g. logs/2026-03-04/13-49-32).",
    )
    p.add_argument("--out", type=Path, default=None, help="Output image path.")

    # --- emg_signal ---
    p = sub.add_parser("emg_signal", help="Visualize raw sEMG signals.")
    p.add_argument("--hdf5", type=Path, required=True, help="Path to an HDF5 session file.")
    p.add_argument(
        "--duration", type=float, default=2.5,
        help="How many seconds to display (default: 2.5).",
    )
    p.add_argument("--out", type=Path, default=None, help="Output image path.")

    # --- overlay ---
    p = sub.add_parser("overlay", help="Overlay val/CER training curves from multiple runs.")
    p.add_argument("--log_dirs", type=Path, nargs="+", required=True)
    p.add_argument("--labels",   type=str,  nargs="+", required=True)
    p.add_argument("--out", type=Path, default=None)

    # --- compare ---
    p = sub.add_parser("compare", help="Compare CER across multiple runs.")
    p.add_argument(
        "--log_dirs", type=Path, nargs="+", required=True,
        help="One Hydra run directory per architecture.",
    )
    p.add_argument(
        "--labels", type=str, nargs="+", required=True,
        help="Display label for each run (same order as --log_dirs).",
    )
    p.add_argument("--out", type=Path, default=None, help="Output image path.")

    # --- channel_ablation ---
    p = sub.add_parser("channel_ablation", help="Plot CER vs. number of channels.")
    p.add_argument(
        "--channels", type=int, nargs="+", required=True,
        help="Electrode channel counts (e.g. 2 4 8 16).",
    )
    p.add_argument(
        "--log_dirs", type=Path, nargs="+", default=None,
        help="Hydra run dirs (one per channel count). If given, CER is read from TensorBoard logs.",
    )
    p.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help="Display labels for each point.",
    )
    p.add_argument(
        "--val_cer", type=float, nargs="+", default=None,
        help="Validation CER for each channel count (required if --log_dirs not given).",
    )
    p.add_argument(
        "--test_cer", type=float, nargs="+", default=None,
        help="Test CER for each channel count (required if --log_dirs not given).",
    )
    p.add_argument("--out", type=Path, default=None, help="Output image path.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "overlay":
        plot_training_curves_overlay(args.log_dirs, args.labels, args.out)

    elif args.command == "training_curves":
        plot_training_curves(args.log_dir, args.out)

    elif args.command == "emg_signal":
        plot_emg_signal(args.hdf5, args.duration, args.out)

    elif args.command == "compare":
        plot_compare(args.log_dirs, args.labels, args.out)

    elif args.command == "channel_ablation":
        if args.log_dirs is not None:
            labels = args.labels or [str(c) + "ch" for c in args.channels]
            plot_channel_ablation_from_dirs(args.log_dirs, args.channels, labels, args.out)
        else:
            if not (len(args.channels) == len(args.val_cer) == len(args.test_cer)):
                print("Error: --channels, --val_cer, and --test_cer must all have the same length.")
                sys.exit(1)
            plot_channel_ablation(args.channels, args.val_cer, args.test_cer, args.out, args.labels)


if __name__ == "__main__":
    main()
