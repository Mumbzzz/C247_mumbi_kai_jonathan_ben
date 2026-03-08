"""
Plots comparing model performance across electrode channel counts:
16, 8, 4, 2 channels per hand (all at 2000 Hz).

Generates three figures saved to Playground_Ben/plots/:
  1. channel_ablation_cer.png    — line plot: val & test CER vs channel count
  2. channel_ablation_curves.png — val/CER training curves overlaid for all 4 counts
  3. channel_ablation_loss.png   — val loss training curves overlaid for all 4 counts

Run from the repo root with the venv active:
    python Playground_Ben/scripts/plot_channel_ablation.py
"""

import sys
import glob
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

# ── UCLA styling ──────────────────────────────────────────────────────────────
UCLA = {
    "blue":       "#2774AE",
    "gold":       "#FFD100",
    "dark_blue":  "#003B5C",
    "mid_blue":   "#005587",
    "light_blue": "#8BB8EE",
    "dark_gold":  "#FFB81C",
}
PALETTE = [UCLA["dark_blue"], UCLA["blue"], UCLA["light_blue"], UCLA["gold"]]

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
# All runs at 2000 Hz raw data, varying channel count only.
CHANNEL_CONFIGS = {
    16: "log_spectrogram",
    8:  "channel_stride2",
    4:  "channel_stride4",
    2:  "channel_stride8",
}

def _find_best_run(transform_tag: str) -> Path | None:
    """Return the run with the most completed epochs matching transform_tag,
    scanning both the external emg2qwerty logs and the repo logs.
    Excludes runs that used temporal downsampling (1000 Hz runs) by checking
    the full expanded config written to emg2qwerty.log.
    """
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
            # Check the full expanded config in emg2qwerty.log for temporal downsample
            train_log = d / "emg2qwerty.log"
            if train_log.exists() and transform_tag != "log_spectrogram":
                log_text = train_log.read_text()
                if "TemporalDownsample" in log_text or "BandpassFilter" in log_text:
                    continue
            ckpts = [c for c in (d / "checkpoints").glob("*.ckpt")
                     if c.name != "last.ckpt"]
            if not ckpts:
                continue
            best_epoch = max(
                int(c.stem.split("epoch=")[1].split("-")[0]) for c in ckpts
            )
            if best_epoch < 10:
                continue
            candidates.append((best_epoch, d))
    return max(candidates, key=lambda x: x[0])[1] if candidates else None

OUT_DIR = PLAYGROUND / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── TensorBoard helpers ───────────────────────────────────────────────────────

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


def epoch_series(scalars: dict, tag: str) -> tuple[np.ndarray, np.ndarray]:
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

data = []

for num_ch, transform_tag in CHANNEL_CONFIGS.items():
    log_dir = _find_best_run(transform_tag)
    print(f"Loading {num_ch} ch/hand [{transform_tag}] ...")
    if log_dir is None:
        print(f"  SKIP: no run found")
        continue
    try:
        scalars, training_sec = load_scalars(log_dir)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        continue

    val_ep,  val_cer_vals  = epoch_series(scalars, "val/CER")
    loss_ep, val_loss_vals = epoch_series(scalars, "val/loss")

    entry = {
        "num_ch":         num_ch,
        "label":          f"{num_ch} ch/hand",
        "val_cer":        last_value(scalars, "val/CER"),
        "test_cer":       last_value(scalars, "test/CER"),
        "val_cer_curve":  (val_ep, val_cer_vals),
        "val_loss_curve": (loss_ep, val_loss_vals),
        "training_sec":   training_sec,
    }
    data.append(entry)
    print(f"  val_CER={entry['val_cer']:.2f}%  test_CER={entry['test_cer']:.2f}%")

data.sort(key=lambda d: d["num_ch"])

# ── figure 1: CER line plot ───────────────────────────────────────────────────

ch_vals   = [d["num_ch"]   for d in data]
val_cers  = [d["val_cer"]  for d in data]
test_cers = [d["test_cer"] for d in data]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(ch_vals, val_cers,  "o-", color=UCLA["blue"], lw=2, label="Val CER",  markersize=7)
ax.plot(ch_vals, test_cers, "s-", color=UCLA["gold"], lw=2, label="Test CER", markersize=7)

for ch, vc, tc in zip(ch_vals, val_cers, test_cers):
    if not np.isnan(vc):
        ax.annotate(f"{vc:.1f}%", (ch, vc), textcoords="offset points",
                    xytext=(0, 8),   ha="center", fontsize=8.5, color=UCLA["blue"])
    if not np.isnan(tc):
        ax.annotate(f"{tc:.1f}%", (ch, tc), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8.5, color=UCLA["dark_gold"])

ax.set_xticks(ch_vals)
ax.set_xticklabels([f"{c} ch/hand" for c in ch_vals])
ax.set_xlabel("Electrode Channels per Hand")
ax.set_ylabel("Character Error Rate (%)")
ax.set_title("CER vs. Electrode Channel Count  (2000 Hz)")
valid_cers = [v for v in val_cers + test_cers if not np.isnan(v)]
ax.set_ylim(0, max(valid_cers) * 1.25)
ax.legend()
ax.grid(True, alpha=0.25, linestyle="--")

plt.tight_layout()
out = OUT_DIR / "channel_ablation_cer.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.close(fig)

# ── figure 2: val/CER training curves ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Validation CER over Training — Channel Count Ablation")

for d, color in zip(data, PALETTE):
    ep, vals = d["val_cer_curve"]
    if len(ep) == 0:
        continue
    ax.plot(ep, vals, label=d["label"], color=color, lw=1.6, alpha=0.9)

ax.set_xlabel("Epoch")
ax.set_ylabel("Val CER (%)")
ax.set_ylim(bottom=0)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.2, linestyle="--")

plt.tight_layout()
out = OUT_DIR / "channel_ablation_curves.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)

# ── figure 3: val loss training curves ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Validation Loss over Training — Channel Count Ablation")

for d, color in zip(data, PALETTE):
    ep, vals = d["val_loss_curve"]
    if len(ep) == 0:
        continue
    ax.plot(ep, vals, label=d["label"], color=color, lw=1.6, alpha=0.9)

ax.set_xlabel("Epoch")
ax.set_ylabel("Val CTC Loss")
ax.set_ylim(bottom=0)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.2, linestyle="--")

plt.tight_layout()
out = OUT_DIR / "channel_ablation_loss.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)

# ── figure 4: CER + training speedup combined ────────────────────────────────
# Speedup relative to 16 ch/hand baseline; drop 16 ch from speed plot

baseline_sec = next(d["training_sec"] for d in data if d["num_ch"] == 16)
speedups = [(1 - d["training_sec"] / baseline_sec) * 100 for d in data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Channel Count Ablation — CER vs. Training Speed  (2000 Hz)", fontsize=14)

# Left: CER
ax1.plot(ch_vals, val_cers,  "o-", color=UCLA["blue"], lw=2, label="Val CER",  markersize=7)
ax1.plot(ch_vals, test_cers, "s-", color=UCLA["gold"], lw=2, label="Test CER", markersize=7)
for ch, vc, tc in zip(ch_vals, val_cers, test_cers):
    if not np.isnan(vc):
        ax1.annotate(f"{vc:.1f}%", (ch, vc), textcoords="offset points",
                     xytext=(0, 8),   ha="center", fontsize=8, color=UCLA["blue"])
    if not np.isnan(tc):
        ax1.annotate(f"{tc:.1f}%", (ch, tc), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color=UCLA["dark_gold"])
ax1.set_xticks(ch_vals)
ax1.set_xticklabels([f"{c} ch/hand" for c in ch_vals])
ax1.set_xlabel("Electrode Channels per Hand")
ax1.set_ylabel("CER (%)")
ax1.set_title("Character Error Rate")
ax1.set_ylim(0, max(v for v in val_cers + test_cers if not np.isnan(v)) * 1.25)
ax1.legend()
ax1.grid(True, alpha=0.25, linestyle="--")

# Right: speedup — exclude 16 ch baseline (0% by definition)
sp_data = [(ch, sp) for ch, sp in zip(ch_vals, speedups) if ch != 16]
sp_ch, sp_vals = zip(*sp_data) if sp_data else ([], [])
ax2.plot(sp_ch, sp_vals, "o-", color=UCLA["dark_blue"], lw=2, markersize=7)
for ch, sp in zip(sp_ch, sp_vals):
    ax2.annotate(f"{sp:.0f}%", (ch, sp), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=8.5, color=UCLA["dark_blue"])
ax2.set_xticks(list(sp_ch))
ax2.set_xticklabels([f"{c} ch/hand" for c in sp_ch])
ax2.set_xlabel("Electrode Channels per Hand")
ax2.set_ylabel("Training Time Reduction vs. 16 ch/hand (%)")
ax2.set_title("Training Speed Improvement vs. 16 ch/hand Baseline")
ax2.set_ylim(0, 105)
ax2.grid(True, alpha=0.25, linestyle="--")

plt.tight_layout()
out = OUT_DIR / "channel_ablation_speed.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)

print(f"\nAll plots saved to {OUT_DIR}/")
