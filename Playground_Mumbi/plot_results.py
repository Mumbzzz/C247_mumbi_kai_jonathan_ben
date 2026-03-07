"""Generate result plots for training fraction ablation experiments.

Run from the workspace root with:
    python -m Playground_Mumbi.plot_results
    python -m Playground_Mumbi.plot_results --model RNN
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

UCLA_COLORS = ['#2D68C4', '#FFD100', '#003B5C', '#8BB8E8', '#00A5E0', '#005587']

FRACTIONS_ORDERED = [0.10, 0.25, 0.50, 0.75, 1.00]
FRACTION_LABELS   = {0.10: "10%", 0.25: "25%", 0.50: "50%", 0.75: "75%", 1.00: "100%"}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "training_fraction_ablation_plots")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def _load_summary(model: str) -> pd.DataFrame | None:
    path = os.path.join(RESULTS_DIR, f"results_summary_{model}.csv")
    if not os.path.exists(path):
        print(f"[plot_results] summary file not found: {path}")
        return None
    df = pd.read_csv(path)
    # Keep only ablation_train_fraction rows if that column/value exists
    if "notes" in df.columns and "ablation_train_fraction" in df["notes"].values:
        df = df[df["notes"] == "ablation_train_fraction"].copy()
    # Keep best (lowest final_val_cer) run per train_fraction
    df = (
        df.sort_values("final_val_cer")
          .groupby("train_fraction", as_index=False)
          .first()
    )
    df = df.sort_values("train_fraction").reset_index(drop=True)
    return df


def _load_curves(model: str, run_ids: list[str] | None = None) -> pd.DataFrame | None:
    path = os.path.join(RESULTS_DIR, f"results_curves_{model}.csv")
    if not os.path.exists(path):
        print(f"[plot_results] curves file not found: {path}")
        return None
    df = pd.read_csv(path)
    if run_ids is not None:
        df = df[df["run_id"].isin(run_ids)].copy()
    return df


def _fraction_label(frac: float) -> str:
    return FRACTION_LABELS.get(round(frac, 2), f"{int(round(frac * 100))}%")


def _find_elbow(fractions: list[float], cers: list[float], threshold: float = 3.0) -> int | None:
    """Return index of the first fraction where marginal CER drop falls below threshold%."""
    for i in range(1, len(fractions)):
        improvement = cers[i - 1] - cers[i]
        if improvement < threshold:
            return i
    return None


# ---------------------------------------------------------------------------
# Plot 1: Val CER and Test CER vs Training Fraction
# ---------------------------------------------------------------------------

def plot_cer_vs_fraction(summary: pd.DataFrame, model: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    fracs  = summary["train_fraction"].tolist()
    labels = [_fraction_label(f) for f in fracs]
    x      = list(range(len(fracs)))

    val_cers  = summary["final_val_cer"].tolist()
    test_cers = summary["test_cer"].tolist()

    ax.plot(x, val_cers,  marker="o", color=UCLA_COLORS[0], linewidth=2,
            markersize=7, label="Val CER",  zorder=3)
    ax.plot(x, test_cers, marker="s", color=UCLA_COLORS[4], linewidth=2,
            markersize=7, label="Test CER", zorder=3)

    # Annotate points
    for i, (v, t) in enumerate(zip(val_cers, test_cers)):
        ax.annotate(f"{v:.1f}", (i, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=UCLA_COLORS[0])
        ax.annotate(f"{t:.1f}", (i, t), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=UCLA_COLORS[4])

    # Elbow point
    elbow_idx = _find_elbow(fracs, val_cers, threshold=3.0)
    if elbow_idx is not None:
        ax.axvline(x=elbow_idx, color=UCLA_COLORS[1], linestyle="--",
                   linewidth=1.5, label="Elbow (<3% marginal gain)", zorder=2)
        ax.annotate(
            f"Elbow\n{labels[elbow_idx]}",
            xy=(elbow_idx, val_cers[elbow_idx]),
            xytext=(elbow_idx + 0.15, val_cers[elbow_idx] + (max(val_cers) - min(val_cers)) * 0.1),
            fontsize=8, color=UCLA_COLORS[2],
            arrowprops=dict(arrowstyle="->", color=UCLA_COLORS[2], lw=1.2),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    _apply_style(ax,
                 title=f"Val CER and Test CER vs Training Fraction ({model})",
                 xlabel="Training Fraction",
                 ylabel="CER (%)")
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"cer_vs_fraction_{model}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot_results] saved {out}")


# ---------------------------------------------------------------------------
# Plot 2: Training Curves – Val CER vs Epoch by Training Fraction
# ---------------------------------------------------------------------------

def plot_training_curves(curves: pd.DataFrame, summary: pd.DataFrame, model: str) -> None:
    # Map run_id -> train_fraction using summary
    run_to_frac: dict[str, float] = dict(
        zip(summary["run_id"], summary["train_fraction"])
    )
    curves = curves[curves["run_id"].isin(run_to_frac)].copy()
    curves["train_fraction"] = curves["run_id"].map(run_to_frac)

    # Sort fractions for consistent color assignment
    present_fracs = sorted(curves["train_fraction"].unique())
    color_map = {f: UCLA_COLORS[i % len(UCLA_COLORS)] for i, f in enumerate(present_fracs)}

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for frac in present_fracs:
        subset = curves[curves["train_fraction"] == frac].sort_values("epoch")
        label  = _fraction_label(frac)
        ax.plot(subset["epoch"].to_numpy(), subset["val_cer"].to_numpy(),
                marker="o", markersize=4, linewidth=2,
                color=color_map[frac], label=label, zorder=3)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    _apply_style(ax,
                 title=f"Val CER vs Epoch by Training Fraction ({model})",
                 xlabel="Epoch",
                 ylabel="Val CER (%)")
    ax.legend(title="Train Fraction", fontsize=9, title_fontsize=9, framealpha=0.9)
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"training_curves_{model}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot_results] saved {out}")


# ---------------------------------------------------------------------------
# Plot 3: Normalized Training Time vs Training Fraction
# ---------------------------------------------------------------------------

def plot_training_time(summary: pd.DataFrame, model: str) -> None:
    fracs  = summary["train_fraction"].tolist()
    labels = [_fraction_label(f) for f in fracs]
    times  = summary["training_time_sec"].tolist()

    # Normalize to full-dataset (fraction == 1.0) if present, else to max
    if 1.0 in fracs:
        baseline = times[fracs.index(1.0)]
    else:
        baseline = max(times)
    norm_times = [t / baseline for t in times]

    x = list(range(len(fracs)))
    colors = [UCLA_COLORS[i % len(UCLA_COLORS)] for i in range(len(fracs))]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(x, norm_times, color=colors, edgecolor="white", linewidth=0.8, zorder=3)

    # Label bars
    for bar, val in zip(bars, norm_times):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{val:.2f}x",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Note
    ax.text(0.98, 0.97,
            "Normalized to full dataset\ntraining time",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(norm_times) * 1.18)
    _apply_style(ax,
                 title=f"Normalized Training Time vs Training Fraction ({model})",
                 xlabel="Training Fraction",
                 ylabel="Normalized Training Time")
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"training_time_{model}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot_results] saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training fraction ablation plots.")
    parser.add_argument("--model", default="CNN",
                        help="Model name suffix used in CSV filenames (default: CNN)")
    args = parser.parse_args()
    model = args.model

    summary = _load_summary(model)
    if summary is None or summary.empty:
        print("[plot_results] No summary data available — exiting.")
        return

    # Plot 1
    plot_cer_vs_fraction(summary, model)

    # Plot 2 + 3 need curves / time columns
    curves = _load_curves(model, run_ids=summary["run_id"].tolist())
    if curves is not None and not curves.empty:
        plot_training_curves(curves, summary, model)
    else:
        print("[plot_results] No curves data available — skipping training curves plot.")

    if "training_time_sec" in summary.columns:
        plot_training_time(summary, model)
    else:
        print("[plot_results] training_time_sec column missing — skipping time plot.")

    print("[plot_results] Done.")


if __name__ == "__main__":
    main()
