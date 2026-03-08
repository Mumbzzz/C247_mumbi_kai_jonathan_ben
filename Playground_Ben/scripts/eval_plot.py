"""
Visualize model predictions on individual windows from the val/test session.

For each window this script produces a three-panel figure:
  1. Raw sEMG signal (both hands, all 16 channels)
  2. CTC emission probability heatmap (time × character class)
  3. Ground-truth vs predicted text with character-level diff colouring

Usage
-----
python scripts/eval_plot.py \\
    --checkpoint logs/2026-03-04/13-49-32/checkpoints/epoch=126-step=15240.ckpt \\
    --hdf5       data/2021-06-04-1622862148-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5 \\
    --n_examples 6 \\
    --out_dir    plots/eval

Optional
--------
--window_length   Number of raw EMG samples per window (default: 8000 = 4 s)
--start_offset    Skip this many seconds from the session start (default: 5)
--stride          Step between consecutive windows in seconds (default: 6)
--channel_indices Space-separated channel indices to use (default: all 16)
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── constants ────────────────────────────────────────────────────────────────
FS_RAW   = 2000          # raw EMG sample rate (Hz)
HOP      = 16            # LogSpectrogram hop_length  → 125 Hz after STFT
FS_SPEC  = FS_RAW / HOP  # spectrogram frame rate
N_FFT    = 64
FREQ_BINS = N_FFT // 2 + 1   # 33

CMAP_16  = plt.colormaps.get_cmap("tab20").resampled(16)


# ── load model ───────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, channel_indices: list[int]):
    """Load TDSConvCTCModule from checkpoint.
    If channel_indices is not all-16, patch in_features accordingly."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from emg2qwerty.lightning import TDSConvCTCModule

    n_ch = len(channel_indices)
    in_features = FREQ_BINS * n_ch

    # Load with map_location so it works without GPU
    module = TDSConvCTCModule.load_from_checkpoint(
        str(ckpt_path),
        map_location="cpu",
        strict=False,           # allow in_features mismatch for ablation ckpts
    )
    module.eval()
    return module


# ── transforms ───────────────────────────────────────────────────────────────

def build_transform(channel_indices: list[int]):
    """Minimal val-time transform: ToTensor → LogSpectrogram [→ ChannelSelect]."""
    import torchaudio
    from emg2qwerty.transforms import ToTensor, LogSpectrogram, ChannelSelect, Compose

    steps = [ToTensor(fields=["emg_left", "emg_right"]),
             LogSpectrogram(n_fft=N_FFT, hop_length=HOP)]
    if sorted(channel_indices) != list(range(16)):
        steps.append(ChannelSelect(indices=channel_indices))
    return Compose(steps)


# ── load a window ─────────────────────────────────────────────────────────────

def load_window_raw(hdf5_path: Path, start_sample: int, window_length: int):
    """Return raw EMG and label data for a single window."""
    from emg2qwerty.data import EMGSessionData, LabelData

    with EMGSessionData(hdf5_path) as session:
        end_sample = min(start_sample + window_length, len(session))
        window = session[start_sample:end_sample]
        timestamps = window[EMGSessionData.TIMESTAMPS]
        start_t, end_t = timestamps[0], timestamps[-1]
        label_data = session.ground_truth(start_t, end_t)
        emg_left  = window[EMGSessionData.EMG_LEFT]
        emg_right = window[EMGSessionData.EMG_RIGHT]
    return window, emg_left, emg_right, timestamps, label_data


# ── run inference ─────────────────────────────────────────────────────────────

def run_inference(module, window_np, transform):
    """Apply transform and run forward pass. Returns (emissions_np, prediction)."""
    from emg2qwerty.decoder import CTCGreedyDecoder

    with torch.no_grad():
        x = transform(window_np)          # (T, bands=2, C, freq)
        x = x.unsqueeze(1)                # add batch dim → (T, 1, bands, C, freq)
        emissions = module(x)             # (T, 1, num_classes) log-softmax
        emissions_np = emissions[:, 0, :].cpu().numpy()   # (T, num_classes)

    decoder = CTCGreedyDecoder()
    prediction = decoder.decode(
        emissions=emissions_np,
        timestamps=np.arange(len(emissions_np)),
    )
    return emissions_np, prediction


# ── diff colouring helper ─────────────────────────────────────────────────────

def char_diff_colors(pred_text: str, gt_text: str):
    """
    Align pred and gt with Levenshtein edit-ops and return per-character colours
    for pred (green=correct, red=wrong/insert) and gt (green=correct, blue=delete).
    """
    import Levenshtein
    ops = Levenshtein.editops(pred_text, gt_text)

    pred_colors = ["#2e7d32"] * len(pred_text)   # default: correct (dark green)
    gt_colors   = ["#2e7d32"] * len(gt_text)

    for op, i, j in ops:
        if op == "replace":
            pred_colors[i] = "#c62828"   # red  – substitution in pred
            gt_colors[j]   = "#1565c0"  # blue – what it should have been
        elif op == "insert":             # char in gt, missing from pred
            gt_colors[j]   = "#1565c0"
        elif op == "delete":             # extra char in pred
            pred_colors[i] = "#c62828"

    return pred_colors, gt_colors


# ── per-example figure ────────────────────────────────────────────────────────

def plot_example(
    idx: int,
    emg_left: np.ndarray,
    emg_right: np.ndarray,
    timestamps: np.ndarray,
    emissions_np: np.ndarray,
    prediction,
    ground_truth,
    channel_indices: list[int],
    out_path: Path,
) -> None:
    from emg2qwerty.charset import charset as get_charset
    cs = get_charset()

    t_rel = timestamps - timestamps[0]         # relative seconds
    n_ch  = emg_left.shape[1]
    spacing = 5.0

    # Character labels on y-axis of emission heatmap
    char_labels = [cs.label_to_char(i) for i in range(cs.num_classes - 1)]
    char_labels.append("∅")   # blank / null class

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Example {idx+1}  |  GT: \"{str(ground_truth)[:60]}\"",
        fontsize=11, y=0.99,
    )

    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.8, 3.5, 0.8],
        width_ratios=[1, 1],
        hspace=0.42, wspace=0.08,
    )

    # ── Panel 1a/1b: raw EMG ────────────────────────────────────────────────
    for col, (emg, hand) in enumerate([(emg_right, "Right hand"), (emg_left, "Left hand")]):
        ax = fig.add_subplot(gs[0, col])
        emg_c = emg - emg.mean(axis=0)
        scale = emg_c.std() or 1.0
        emg_n = emg_c / scale
        for ch in range(n_ch):
            ax.plot(t_rel, emg_n[:, ch] + ch * spacing,
                    color=CMAP_16(ch), lw=0.5, alpha=0.85, rasterized=True)
        ax.set_ylabel(hand, fontsize=9)
        ax.set_yticks([])
        ax.set_xlim(t_rel[0], t_rel[-1])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
        if col == 0:
            ax.set_title("Raw sEMG", fontsize=10)

    # ── Panel 2: emission heatmap (full width) ───────────────────────────────
    ax_em = fig.add_subplot(gs[1, :])
    probs = np.exp(emissions_np).T    # (num_classes, T)
    T_frames = probs.shape[1]
    duration  = t_rel[-1]

    im = ax_em.imshow(
        probs,
        aspect="auto",
        origin="lower",
        extent=[0, duration, -0.5, cs.num_classes - 0.5],
        cmap="hot",
        vmin=0, vmax=0.6,
        interpolation="nearest",
        rasterized=True,
    )
    fig.colorbar(im, ax=ax_em, fraction=0.015, pad=0.01, label="P(class)")

    # y-axis: character labels (every other one to avoid overlap)
    tick_step = 2
    ticks     = list(range(0, cs.num_classes, tick_step))
    ax_em.set_yticks(ticks)
    ax_em.set_yticklabels([char_labels[i] for i in ticks], fontsize=5.5)
    ax_em.set_xlabel("Time (s)", fontsize=9)
    ax_em.set_title("CTC Emission Probabilities  (brighter = higher confidence)", fontsize=10)

    # Overlay ground truth keystroke lines
    if ground_truth.timestamps is not None:
        t0 = timestamps[0]
        for ch_t in ground_truth.timestamps:
            tr = ch_t - t0
            if 0 <= tr <= duration:
                ax_em.axvline(tr, color="cyan", lw=0.6, alpha=0.55)

    # ── Panel 3: text diff ───────────────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")

    pred_str = str(prediction)
    gt_str   = str(ground_truth)
    pred_colors, gt_colors = char_diff_colors(pred_str, gt_str)

    import Levenshtein
    cer = Levenshtein.distance(pred_str, gt_str) / max(len(gt_str), 1) * 100

    def render_coloured_text(ax, text, colors, y, prefix, fontsize=9.5):
        ax.text(0.0, y, prefix, transform=ax.transAxes,
                fontsize=fontsize, va="center", color="black", fontweight="bold")
        x = 0.09
        for ch, col in zip(text, colors):
            display = ch if ch.isprintable() and ch != " " else "·" if ch == " " else "↵"
            t = ax.text(x, y, display, transform=ax.transAxes,
                        fontsize=fontsize, va="center", color=col,
                        fontfamily="monospace")
            x += 0.012
            if x > 0.98:
                break  # truncate if too long

    render_coloured_text(ax_txt, gt_str,   gt_colors,   0.72, "GT:  ")
    render_coloured_text(ax_txt, pred_str, pred_colors, 0.28, "Pred:")

    # Legend + CER
    patches = [
        mpatches.Patch(color="#2e7d32", label="Correct"),
        mpatches.Patch(color="#c62828", label="Insertion / Substitution (pred)"),
        mpatches.Patch(color="#1565c0", label="Deletion / Substitution (GT)"),
    ]
    ax_txt.legend(handles=patches, loc="lower right",
                  fontsize=7.5, framealpha=0.7,
                  bbox_to_anchor=(1.0, -0.15))
    ax_txt.text(0.0, -0.05,
                f"CER = {cer:.1f}%  |  GT length = {len(gt_str)}  |  "
                f"Pred length = {len(pred_str)}",
                transform=ax_txt.transAxes, fontsize=9, color="#555555")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hdf5",       type=Path, required=True)
    parser.add_argument("--n_examples", type=int,  default=6)
    parser.add_argument("--out_dir",    type=Path, default=Path("plots/eval"))
    parser.add_argument("--window_length", type=int, default=8000,
                        help="Window size in raw samples (default: 8000 = 4 s)")
    parser.add_argument("--start_offset", type=float, default=5.0,
                        help="Skip this many seconds from session start (default: 5)")
    parser.add_argument("--stride", type=float, default=6.0,
                        help="Seconds between consecutive windows (default: 6)")
    parser.add_argument("--channel_indices", type=int, nargs="+",
                        default=list(range(16)),
                        help="Which electrode channels to use (default: all 16)")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint} ...")
    module = load_model(args.checkpoint, args.channel_indices)
    transform = build_transform(args.channel_indices)

    print(f"Generating {args.n_examples} examples from {args.hdf5.name} ...")
    start_sample = int(args.start_offset * FS_RAW)
    stride_samples = int(args.stride * FS_RAW)

    for i in range(args.n_examples):
        s = start_sample + i * stride_samples
        window_np, emg_l, emg_r, timestamps, gt = load_window_raw(
            args.hdf5, s, args.window_length
        )
        if len(window_np) < args.window_length // 2:
            print(f"  Window {i+1}: too short, skipping.")
            continue

        emissions_np, prediction = run_inference(module, window_np, transform)

        out_path = args.out_dir / f"example_{i+1:02d}.png"
        plot_example(
            idx=i,
            emg_left=emg_l,
            emg_right=emg_r,
            timestamps=timestamps,
            emissions_np=emissions_np,
            prediction=prediction,
            ground_truth=gt,
            channel_indices=args.channel_indices,
            out_path=out_path,
        )

    print(f"\nDone. Plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
