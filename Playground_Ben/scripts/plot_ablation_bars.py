"""
Bar plots + summary tables for both ablation studies.

Saves (new filenames — does not overwrite existing plots):
  Playground_Ben/plots/channel_ablation_bar.png
  Playground_Ben/plots/sampling_ablation_bar.png
  Playground_Ben/plots/ablation_combined_bar.png

Also prints formatted tables and writes:
  Playground_Ben/results/channel_ablation_table.csv
  Playground_Ben/results/sampling_ablation_table.csv

Run from the repo root with the venv active:
    python Playground_Ben/scripts/plot_ablation_bars.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parents[2]
PLAYGROUND = REPO_ROOT / "Playground_Ben"
EMG_LOGS   = Path("/home/benforbes/emg2qwerty/logs")

sys.path.insert(0, str(REPO_ROOT))
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator, SCALARS,
)

# ── UCLA styling ───────────────────────────────────────────────────────────────
UCLA = {
    "blue":       "#2774AE",
    "gold":       "#FFD100",
    "dark_blue":  "#003B5C",
    "mid_blue":   "#005587",
    "light_blue": "#8BB8EE",
    "dark_gold":  "#FFB81C",
}

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman"],
    "font.size":          12,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "axes.labelsize":     12,
    "axes.labelweight":   "bold",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
})

OUT_DIR     = PLAYGROUND / "plots"
RESULTS_DIR = PLAYGROUND / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── TensorBoard helpers ────────────────────────────────────────────────────────

def load_scalars(log_dir: Path) -> tuple[dict, float]:
    tb_dir = log_dir / "lightning_logs" / "version_0"
    event_files = sorted(glob.glob(str(tb_dir / "events.out.tfevents.*")))
    if not event_files:
        raise FileNotFoundError(f"No events in {tb_dir}")
    merged: dict = {}
    wall_times: list[float] = []
    for ef in event_files:
        ea = EventAccumulator(ef, size_guidance={SCALARS: 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            wall_times.extend(e.wall_time for e in events)
            merged.setdefault(tag, [])
            merged[tag] = sorted(merged[tag] + list(events), key=lambda e: e.step)
    training_sec = (max(wall_times) - min(wall_times)) if wall_times else 0.0
    return merged, training_sec


def last_value(scalars: dict, tag: str) -> float:
    events = scalars.get(tag, [])
    return events[-1].value if events else float("nan")


def best_value(scalars: dict, tag: str, minimize: bool = True) -> float:
    events = scalars.get(tag, [])
    vals = [e.value for e in events if not np.isnan(e.value)]
    if not vals:
        return float("nan")
    return min(vals) if minimize else max(vals)


# ── run-finder helpers (mirrors existing plot scripts) ─────────────────────────

def _find_temporal_run(factor: int) -> Path | None:
    for d in sorted((REPO_ROOT / "logs").glob("*/*"), reverse=True):
        hydra = d / "hydra_configs" / "hydra.yaml"
        if not hydra.exists():
            continue
        text = hydra.read_text()
        if f"temporal_downsample_{factor}" not in text:
            continue
        if factor > 2 and "window_length" not in text:
            continue
        ckpts = [c for c in (d / "checkpoints").glob("*.ckpt") if c.name != "last.ckpt"]
        if not ckpts:
            continue
        return d
    return None


def _find_channel_run(transform_tag: str) -> Path | None:
    candidates = []
    for root in [EMG_LOGS, REPO_ROOT / "logs"]:
        if not root.exists():
            continue
        for d in sorted(root.glob("*/*"), reverse=True):
            hydra = d / "hydra_configs" / "hydra.yaml"
            if not hydra.exists():
                continue
            if transform_tag not in hydra.read_text():
                continue
            train_log = d / "emg2qwerty.log"
            if train_log.exists() and transform_tag != "log_spectrogram":
                log_text = train_log.read_text()
                if "TemporalDownsample" in log_text or "BandpassFilter" in log_text:
                    continue
            ckpts = [c for c in (d / "checkpoints").glob("*.ckpt") if c.name != "last.ckpt"]
            if not ckpts:
                continue
            best_epoch = max(
                int(c.stem.split("epoch=")[1].split("-")[0]) for c in ckpts
            )
            if best_epoch < 10:
                continue
            candidates.append((best_epoch, d))
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


# ── collect channel ablation data ─────────────────────────────────────────────

CHANNEL_CONFIGS = {
    16: "log_spectrogram",
    8:  "channel_stride2_2000hz",
    4:  "channel_stride4_2000hz",
    2:  "channel_stride8_2000hz",
}

print("=" * 60)
print("CHANNEL ABLATION")
print("=" * 60)

ch_data = []
for num_ch, tag in CHANNEL_CONFIGS.items():
    log_dir = _find_channel_run(tag)
    print(f"  {num_ch:2d} ch/hand [{tag}]: ", end="")
    if log_dir is None:
        print("NO RUN FOUND — skipped")
        continue
    try:
        scalars, training_sec = load_scalars(log_dir)
    except FileNotFoundError as e:
        print(f"SKIP ({e})")
        continue
    entry = {
        "num_ch":       num_ch,
        "label":        f"{num_ch} ch/hand",
        "val_cer":      best_value(scalars, "val/CER", minimize=True),
        "test_cer":     last_value(scalars, "test/CER"),
        "val_loss":     best_value(scalars, "val/loss", minimize=True),
        "training_sec": training_sec,
    }
    ch_data.append(entry)
    print(f"val_CER={entry['val_cer']:.2f}%  test_CER={entry['test_cer']:.2f}%")

ch_data.sort(key=lambda d: d["num_ch"])

# ── collect temporal/sampling ablation data ────────────────────────────────────

TEMPORAL_RUNS = [
    (2000, EMG_LOGS / "2026-03-04/17-06-49", "2000 Hz\n(baseline)"),
    (1000, _find_temporal_run(2),              "1000 Hz\n(×2)"),
    ( 500, _find_temporal_run(4),              "500 Hz\n(×4)"),
    ( 250, _find_temporal_run(8),              "250 Hz\n(×8)"),
    ( 125, _find_temporal_run(16),             "125 Hz\n(×16)"),
]

print()
print("=" * 60)
print("TEMPORAL / SAMPLING ABLATION")
print("=" * 60)

sp_data = []
for hz, log_dir, label in TEMPORAL_RUNS:
    print(f"  {hz:4d} Hz: ", end="")
    if log_dir is None or not log_dir.exists():
        print("NO RUN FOUND — skipped")
        continue
    try:
        scalars, training_sec = load_scalars(log_dir)
    except FileNotFoundError as e:
        print(f"SKIP ({e})")
        continue
    entry = {
        "hz":           hz,
        "label":        label,
        "label_short":  f"{hz} Hz",
        "val_cer":      best_value(scalars, "val/CER", minimize=True),
        "test_cer":     last_value(scalars, "test/CER"),
        "val_loss":     best_value(scalars, "val/loss", minimize=True),
        "training_sec": training_sec,
    }
    sp_data.append(entry)
    print(f"val_CER={entry['val_cer']:.2f}%  test_CER={entry['test_cer']:.2f}%")


# ── print tables ───────────────────────────────────────────────────────────────

def _fmt(v):
    return f"{v:.2f}%" if not np.isnan(v) else "N/A"

def _fmt_time(sec):
    if np.isnan(sec) or sec == 0:
        return "N/A"
    h, m = divmod(int(sec), 3600)
    m //= 60
    return f"{h}h {m:02d}m" if h else f"{m}m"


print()
print("=" * 60)
print("TABLE: Channel Ablation (2000 Hz raw EMG)")
print("=" * 60)
col_w = [12, 12, 12, 12]
header = f"{'Channels':>{col_w[0]}}  {'Val CER':>{col_w[1]}}  {'Test CER':>{col_w[2]}}  {'Train Time':>{col_w[3]}}"
print(header)
print("-" * (sum(col_w) + 6))
for d in ch_data:
    print(
        f"{d['label']:>{col_w[0]}}  "
        f"{_fmt(d['val_cer']):>{col_w[1]}}  "
        f"{_fmt(d['test_cer']):>{col_w[2]}}  "
        f"{_fmt_time(d['training_sec']):>{col_w[3]}}"
    )

print()
print("=" * 60)
print("TABLE: Temporal Downsampling Ablation")
print("=" * 60)
header = f"{'Sample Rate':>{col_w[0]}}  {'Val CER':>{col_w[1]}}  {'Test CER':>{col_w[2]}}  {'Train Time':>{col_w[3]}}"
print(header)
print("-" * (sum(col_w) + 6))
for d in sp_data:
    print(
        f"{d['label_short']:>{col_w[0]}}  "
        f"{_fmt(d['val_cer']):>{col_w[1]}}  "
        f"{_fmt(d['test_cer']):>{col_w[2]}}  "
        f"{_fmt_time(d['training_sec']):>{col_w[3]}}"
    )


# ── save CSV tables ────────────────────────────────────────────────────────────

ch_csv = RESULTS_DIR / "channel_ablation_table.csv"
with open(ch_csv, "w") as f:
    f.write("channels_per_hand,val_cer_pct,test_cer_pct,training_time_sec\n")
    for d in ch_data:
        f.write(f"{d['num_ch']},{d['val_cer']:.4f},{d['test_cer']:.4f},{d['training_sec']:.0f}\n")
print(f"\nSaved → {ch_csv}")

sp_csv = RESULTS_DIR / "sampling_ablation_table.csv"
with open(sp_csv, "w") as f:
    f.write("sample_rate_hz,val_cer_pct,test_cer_pct,training_time_sec\n")
    for d in sp_data:
        f.write(f"{d['hz']},{d['val_cer']:.4f},{d['test_cer']:.4f},{d['training_sec']:.0f}\n")
print(f"Saved → {sp_csv}")


# ── bar plot helper ────────────────────────────────────────────────────────────

def grouped_bar(ax, labels, val_cers, test_cers, xlabel, title):
    x = np.arange(len(labels))
    w = 0.35
    bars_val  = ax.bar(x - w/2, val_cers,  w, label="Val CER",  color=UCLA["blue"],
                       edgecolor="white", linewidth=0.5)
    bars_test = ax.bar(x + w/2, test_cers, w, label="Test CER", color=UCLA["gold"],
                       edgecolor="white", linewidth=0.5)

    # value labels on top of each bar
    for bar in list(bars_val) + list(bars_test):
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.4,
            f"{h:.1f}%",
            ha="center", va="bottom", fontsize=8.5, color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Character Error Rate (%)")
    ax.set_title(title)
    valid = [v for v in list(val_cers) + list(test_cers) if not np.isnan(v)]
    ax.set_ylim(0, max(valid) * 1.25 if valid else 110)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)


# ── figure 1: channel ablation bar plot ───────────────────────────────────────

if ch_data:
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped_bar(
        ax,
        labels    = [d["label"] for d in ch_data],
        val_cers  = [d["val_cer"]  for d in ch_data],
        test_cers = [d["test_cer"] for d in ch_data],
        xlabel    = "Electrode Channels per Hand",
        title     = "CER vs. Electrode Channel Count  (2000 Hz)",
    )
    plt.tight_layout()
    out = OUT_DIR / "channel_ablation_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close(fig)


# ── figure 2: sampling ablation bar plot ──────────────────────────────────────

if sp_data:
    fig, ax = plt.subplots(figsize=(10, 5))
    grouped_bar(
        ax,
        labels    = [d["label_short"] for d in sp_data],
        val_cers  = [d["val_cer"]  for d in sp_data],
        test_cers = [d["test_cer"] for d in sp_data],
        xlabel    = "Sampling Rate",
        title     = "CER vs. EMG Sampling Rate",
    )
    plt.tight_layout()
    out = OUT_DIR / "sampling_ablation_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ── figure 3: combined side-by-side bar plot ──────────────────────────────────

if ch_data and sp_data:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Ablation Study Results — Character Error Rate", fontsize=14)

    grouped_bar(
        ax1,
        labels    = [d["label"] for d in ch_data],
        val_cers  = [d["val_cer"]  for d in ch_data],
        test_cers = [d["test_cer"] for d in ch_data],
        xlabel    = "Electrode Channels per Hand",
        title     = "Channel Count Ablation  (2000 Hz)",
    )
    grouped_bar(
        ax2,
        labels    = [d["label_short"] for d in sp_data],
        val_cers  = [d["val_cer"]  for d in sp_data],
        test_cers = [d["test_cer"] for d in sp_data],
        xlabel    = "Sampling Rate",
        title     = "Temporal Downsampling Ablation",
    )

    plt.tight_layout()
    out = OUT_DIR / "ablation_combined_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)

print(f"\nAll done. Plots in {OUT_DIR}/")
