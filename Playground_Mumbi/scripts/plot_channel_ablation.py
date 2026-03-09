"""
Plots CNN+LSTM performance across electrode channel counts:
16, 8, 4, 2 channels per hand (all at 2000 Hz, 100% training data).

Reads from:
    results/results_summary_CNN_LSTM.csv
    results/results_curves_CNN_LSTM.csv

Generates four figures saved to Playground_Mumbi/plots/:
    1. channel_ablation_cer.png    — val & test CER vs channel count
    2. channel_ablation_curves.png — val CER training curves for all 4 counts
    3. channel_ablation_loss.png   — val loss training curves for all 4 counts
    4. channel_ablation_speed.png  — CER + training time reduction side by side

Run from the repo root:
    python Playground_Mumbi/scripts/plot_channel_ablation.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parents[2]
PLAYGROUND = REPO_ROOT / "Playground_Mumbi"
OUT_DIR    = PLAYGROUND / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

# ── UCLA styling (matching Ben's plots) ───────────────────────────────────────
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

# ── Load CSVs ─────────────────────────────────────────────────────────────────

SUMMARY_CSV = REPO_ROOT / "results" / "results_summary_CNN_LSTM.csv"
CURVES_CSV  = REPO_ROOT / "results" / "results_curves_CNN_LSTM.csv"

if not SUMMARY_CSV.exists():
    print(f"ERROR: {SUMMARY_CSV} not found. Run the channel ablation first.")
    sys.exit(1)

# Read summary: filter for channel ablation runs at 2000 Hz, 100% data
# For each num_channels keep only the latest run (run_id is lexicographically
# sortable because the timestamp is the suffix).
best_runs: dict[int, dict] = {}
with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("notes") != "ablation_channels":
            continue
        if float(row.get("sampling_rate_hz", 0)) != 2000:
            continue
        if float(row.get("train_fraction", 0)) != 1.0:
            continue
        num_ch = int(row["num_channels"])
        # Keep the run with the lexicographically largest run_id (latest timestamp)
        if num_ch not in best_runs or row["run_id"] > best_runs[num_ch]["run_id"]:
            best_runs[num_ch] = row

if not best_runs:
    print("No ablation_channels runs found in summary CSV. Run the ablation first.")
    sys.exit(1)

print(f"Found {len(best_runs)} channel configurations: {sorted(best_runs)}")

# Read curves: group by run_id
curves_by_run: dict[str, list[dict]] = defaultdict(list)
if CURVES_CSV.exists():
    with open(CURVES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            curves_by_run[row["run_id"]].append(row)

# ── Build data list ───────────────────────────────────────────────────────────

data = []
for num_ch in sorted(best_runs):
    row = best_runs[num_ch]
    run_id = row["run_id"]

    val_cer  = float(row["final_val_cer"])
    test_cer = float(row["test_cer"])
    train_sec = float(row["training_time_sec"])

    # Build epoch curves from the curves CSV
    epoch_rows = sorted(curves_by_run.get(run_id, []), key=lambda r: int(r["epoch"]))
    if epoch_rows:
        epochs        = np.array([int(r["epoch"])      for r in epoch_rows])
        val_cer_curve = np.array([float(r["val_cer"])  for r in epoch_rows])
        val_loss_curve = np.array([float(r["val_loss"]) for r in epoch_rows])
    else:
        epochs = val_cer_curve = val_loss_curve = np.array([])

    data.append({
        "num_ch":          num_ch,
        "label":           f"{num_ch} ch/hand",
        "val_cer":         val_cer,
        "test_cer":        test_cer,
        "training_sec":    train_sec,
        "val_cer_curve":   (epochs, val_cer_curve),
        "val_loss_curve":  (epochs, val_loss_curve),
    })
    print(f"  {num_ch} ch/hand | val_CER={val_cer:.2f}%  test_CER={test_cer:.2f}%"
          f"  time={train_sec/3600:.2f}h  run_id={run_id}")

# ── Figure 1: CER line plot ───────────────────────────────────────────────────

ch_vals   = [d["num_ch"]   for d in data]
val_cers  = [d["val_cer"]  for d in data]
test_cers = [d["test_cer"] for d in data]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ch_vals, val_cers,  "o-", color=UCLA["blue"], lw=2, label="Val CER",  markersize=7)
ax.plot(ch_vals, test_cers, "s-", color=UCLA["gold"], lw=2, label="Test CER", markersize=7)
ax.set_xticks(ch_vals)
ax.set_xticklabels([f"{c} ch/hand" for c in ch_vals])
ax.set_xlabel("Electrode Channels per Hand")
ax.set_ylabel("Character Error Rate (%)")
ax.set_title("CNN+LSTM: CER vs. Electrode Channel Count  (2000 Hz)")
valid_cers = [v for v in val_cers + test_cers if not np.isnan(v)]
ax.set_ylim(0, max(valid_cers) * 1.25)
ax.legend()
ax.grid(True, alpha=0.25, linestyle="--")
plt.tight_layout()
out = OUT_DIR / "channel_ablation_cer.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.close(fig)

# ── Figure 2: val CER training curves ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("CNN+LSTM: Validation CER over Training — Channel Count Ablation")
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

# ── Figure 3: val loss training curves ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("CNN+LSTM: Validation Loss over Training — Channel Count Ablation")
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

# ── Figure 4: CER + training speedup combined ────────────────────────────────

baseline = next((d for d in data if d["num_ch"] == 16), None)
if baseline is not None:
    baseline_sec = baseline["training_sec"]
    speedups = [(1 - d["training_sec"] / baseline_sec) * 100 for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("CNN+LSTM Channel Count Ablation — CER vs. Training Speed  (2000 Hz)",
                 fontsize=14)

    ax1.plot(ch_vals, val_cers,  "o-", color=UCLA["blue"], lw=2, label="Val CER",  markersize=7)
    ax1.plot(ch_vals, test_cers, "s-", color=UCLA["gold"], lw=2, label="Test CER", markersize=7)
    ax1.set_xticks(ch_vals)
    ax1.set_xticklabels([f"{c} ch/hand" for c in ch_vals])
    ax1.set_xlabel("Electrode Channels per Hand")
    ax1.set_ylabel("CER (%)")
    ax1.set_title("Character Error Rate")
    ax1.set_ylim(0, max(v for v in val_cers + test_cers if not np.isnan(v)) * 1.25)
    ax1.legend()
    ax1.grid(True, alpha=0.25, linestyle="--")

    sp_data = [(ch, sp) for ch, sp in zip(ch_vals, speedups) if ch != 16]
    if sp_data:
        sp_ch, sp_vals = zip(*sp_data)
        ax2.plot(sp_ch, sp_vals, "o-", color=UCLA["dark_blue"], lw=2, markersize=7)
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
else:
    print("Skipping speed plot: no 16 ch/hand run found for baseline.")

print(f"\nAll plots saved to {OUT_DIR}/")
