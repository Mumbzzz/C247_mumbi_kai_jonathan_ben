"""
Plots comparing model performance across temporal downsampling rates:
2000, 1000, 500, 250, 125 Hz.

Generates three figures saved to Playground_Ben/plots/:
  1. sampling_ablation_cer.png    — bar chart: val & test CER vs sample rate
  2. sampling_ablation_curves.png — val/CER training curves overlaid for all 5 rates
  3. sampling_ablation_loss.png   — val loss training curves overlaid for all 5 rates

Also logs the 2000 Hz baseline into results/results_summary_CNN.csv if not present.

Run from the repo root with the venv active:
    python Playground_Ben/scripts/plot_sampling_ablation.py
"""

import sys
import glob
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT   = Path(__file__).resolve().parents[2]
PLAYGROUND  = REPO_ROOT / "Playground_Ben"
EMG_LOGS    = Path("/home/benforbes/emg2qwerty/logs")

sys.path.insert(0, str(REPO_ROOT))
from scripts.logger import log_summary, make_run_id
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator, SCALARS,
)

# ── UCLA styling ──────────────────────────────────────────────────────────────
UCLA = {
    "blue":        "#2774AE",
    "gold":        "#FFD100",
    "dark_blue":   "#003B5C",
    "mid_blue":    "#005587",
    "light_blue":  "#8BB8EE",
    "dark_gold":   "#FFB81C",
}
PALETTE = [
    UCLA["dark_blue"], UCLA["mid_blue"], UCLA["blue"],
    UCLA["light_blue"], UCLA["gold"],
]

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# ── run registry ──────────────────────────────────────────────────────────────
# Auto-detects the most recent completed run for each downsampling factor.

def _find_best_run(factor: int) -> Path | None:
    for d in sorted((REPO_ROOT / "logs").glob("*/*"), reverse=True):
        hydra = d / "hydra_configs" / "hydra.yaml"
        if not hydra.exists():
            continue
        text = hydra.read_text()
        if f"temporal_downsample_{factor}" not in text:
            continue
        if factor > 2 and "window_length" not in text:
            continue
        ckpts = [c for c in (d / "checkpoints").glob("*.ckpt")
                 if c.name != "last.ckpt"]
        if not ckpts:
            continue
        return d
    return None

# (sample_rate_hz, log_dir, label)
RUNS = [
    (2000, EMG_LOGS / "2026-03-04/17-06-49", "2000 Hz\n(baseline)"),
    (1000, _find_best_run(2),                 "1000 Hz\n(2×)"),
    ( 500, _find_best_run(4),                 "500 Hz\n(4×)"),
    ( 250, _find_best_run(8),                 "250 Hz\n(8×)"),
    ( 125, _find_best_run(16),                "125 Hz\n(16×)"),
]

OUT_DIR = PLAYGROUND / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── TensorBoard helpers ───────────────────────────────────────────────────────

def load_scalars(log_dir: Path) -> dict:
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
    return merged, (min(wall_times), max(wall_times)) if wall_times else (0, 0)


def epoch_series(scalars: dict, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (epochs, values) arrays aligned by step→epoch mapping."""
    step_to_epoch = {e.step: int(e.value) for e in scalars.get("epoch", [])}
    pairs = []
    for e in scalars.get(tag, []):
        ep = step_to_epoch.get(e.step)
        if ep is not None:
            pairs.append((ep, e.value))
    if not pairs:
        return np.array([]), np.array([])
    pairs.sort()
    eps, vals = zip(*pairs)
    return np.array(eps), np.array(vals)


def last_value(scalars: dict, tag: str) -> float:
    events = scalars.get(tag, [])
    return events[-1].value if events else float("nan")


# ── collect data ──────────────────────────────────────────────────────────────

data = []   # list of dicts, one per run

for hz, log_dir, label in RUNS:
    print(f"Loading {hz} Hz from {log_dir.relative_to(log_dir.parents[3]) if log_dir.parents[3].exists() else log_dir} ...")
    try:
        scalars, (t0, t1) = load_scalars(log_dir)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        continue

    val_ep,  val_cer_vals  = epoch_series(scalars, "val/CER")
    loss_ep, val_loss_vals = epoch_series(scalars, "val/loss")

    entry = {
        "hz":           hz,
        "label":        label,
        "val_cer":      last_value(scalars, "val/CER"),
        "test_cer":     last_value(scalars, "test/CER"),
        "final_train_loss": last_value(scalars, "train/loss"),
        "final_val_loss":   last_value(scalars, "val/loss"),
        "val_cer_curve":  (val_ep,  val_cer_vals),
        "val_loss_curve": (loss_ep, val_loss_vals),
        "training_sec":  t1 - t0,
        "scalars":       scalars,
    }
    data.append(entry)
    print(f"  val_CER={entry['val_cer']:.2f}%  test_CER={entry['test_cer']:.2f}%")

# ── log 2000 Hz baseline to CSV if not already there ─────────────────────────
results_summary = REPO_ROOT / "results" / "results_summary_CNN.csv"
baseline = next((d for d in data if d["hz"] == 2000), None)

if baseline:
    existing_ids = set()
    if results_summary.exists():
        with open(results_summary) as f:
            for line in f:
                existing_ids.add(line.split(",")[0])

    run_id = make_run_id("CNN", 16, 2000, 1.0, timestamp="20260304_170649")
    if run_id not in existing_ids:
        log_summary(
            run_id=run_id,
            model="CNN",
            epochs=150,
            num_channels=16,
            sampling_rate_hz=2000,
            train_fraction=1.0,
            input_type="spectrogram",
            final_train_loss=baseline["final_train_loss"],
            final_val_loss=baseline["final_val_loss"],
            final_val_cer=baseline["val_cer"],
            test_cer=baseline["test_cer"],
            training_time_sec=baseline["training_sec"],
            notes="ablation_sampling_rate",
        )
        print(f"\nLogged 2000 Hz baseline: {run_id}")

# ── figure 1: CER line plot ───────────────────────────────────────────────────

hz_vals   = [d["hz"]       for d in data]
val_cers  = [d["val_cer"]  for d in data]
test_cers = [d["test_cer"] for d in data]

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(hz_vals, val_cers,  "o-", color=UCLA["blue"], lw=2,   label="Val CER",  markersize=7)
ax.plot(hz_vals, test_cers, "s-", color=UCLA["gold"], lw=2,   label="Test CER", markersize=7)


ax.set_xscale("log", base=2)
ax.set_xticks(hz_vals)
ax.set_xticklabels([f"{hz} Hz" for hz in hz_vals])
ax.invert_xaxis()   # left = highest rate (best), right = most downsampled
ax.set_xlabel("Sampling Rate")
ax.set_ylabel("Character Error Rate (%)")
ax.set_title("CER vs. EMG Sampling Rate")
ax.set_ylim(0, max(max(val_cers), max(test_cers)) * 1.2)
ax.legend()
ax.grid(True, alpha=0.25, linestyle="--")

plt.tight_layout()
out = OUT_DIR / "sampling_ablation_cer.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.close(fig)

# ── figure 2: val/CER training curves ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Validation CER over Training — Sampling Rate Ablation")

for d, color in zip(data, PALETTE):
    ep, vals = d["val_cer_curve"]
    if len(ep) == 0:
        continue
    # Skip NaN-dominated curves (125 Hz) — still plot but dashed
    nan_frac = np.isnan(vals).mean()
    ls = "--" if nan_frac > 0.5 else "-"
    lw = 1.2 if nan_frac > 0.5 else 1.6
    ax.plot(ep, vals, label=d["label"].replace("\n", " "), color=color,
            lw=lw, linestyle=ls, alpha=0.9)

ax.set_xlabel("Epoch")
ax.set_ylabel("Val CER (%)")
ax.set_ylim(bottom=0)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.2, linestyle="--")

plt.tight_layout()
out = OUT_DIR / "sampling_ablation_curves.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)

# ── figure 3: val loss training curves ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Validation Loss over Training — Sampling Rate Ablation")

for d, color in zip(data, PALETTE):
    ep, vals = d["val_loss_curve"]
    if len(ep) == 0:
        continue
    nan_frac = np.isnan(vals).mean()
    ls = "--" if nan_frac > 0.5 else "-"
    lw = 1.2 if nan_frac > 0.5 else 1.6
    ax.plot(ep, vals, label=d["label"].replace("\n", " "), color=color,
            lw=lw, linestyle=ls, alpha=0.9)

ax.set_xlabel("Epoch")
ax.set_ylabel("Val CTC Loss")
ax.set_ylim(bottom=0)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.2, linestyle="--")

plt.tight_layout()
out = OUT_DIR / "sampling_ablation_loss.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)

# ── figure 4: CER + training speedup combined ────────────────────────────────

baseline_sec = next(d["training_sec"] for d in data if d["hz"] == 2000)
speedups = [(1 - d["training_sec"] / baseline_sec) * 100 for d in data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Sampling Rate Ablation — CER vs. Training Speed", fontsize=14)

# Left: CER
ax1.plot(hz_vals, val_cers,  "o-", color=UCLA["blue"], lw=2, label="Val CER",  markersize=7)
ax1.plot(hz_vals, test_cers, "s-", color=UCLA["gold"], lw=2, label="Test CER", markersize=7)
ax1.set_xscale("log", base=2)
ax1.set_xticks(hz_vals)
ax1.set_xticklabels([f"{hz} Hz" for hz in hz_vals])
ax1.invert_xaxis()
ax1.set_xlabel("Sampling Rate")
ax1.set_ylabel("CER (%)")
ax1.set_title("Character Error Rate")
ax1.set_ylim(0, max(max(val_cers), max(test_cers)) * 1.2)
ax1.legend()
ax1.grid(True, alpha=0.25, linestyle="--")

# Right: speedup — exclude the 2000 Hz baseline (0% by definition)
speed_data = [(hz, sp) for hz, sp in zip(hz_vals, speedups) if hz != 2000]
sp_hz, sp_vals = zip(*speed_data) if speed_data else ([], [])
ax2.plot(sp_hz, sp_vals, "o-", color=UCLA["dark_blue"], lw=2, markersize=7)
ax2.set_xscale("log", base=2)
ax2.set_xticks(list(sp_hz))
ax2.set_xticklabels([f"{hz} Hz" for hz in sp_hz])
ax2.invert_xaxis()
ax2.set_xlabel("Sampling Rate")
ax2.set_ylabel("Training Time Reduction vs. 2000 Hz (%)")
ax2.set_title("Training Speed Improvement vs. 2000 Hz Baseline")
ax2.set_ylim(0, 105)
ax2.grid(True, alpha=0.25, linestyle="--")

plt.tight_layout()
out = OUT_DIR / "sampling_ablation_speed.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)

print(f"\nAll plots saved to {OUT_DIR}/")
