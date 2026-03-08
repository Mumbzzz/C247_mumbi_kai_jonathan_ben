"""
Comprehensive EMG data analysis: signal visualization, FFT, and cross-correlation.

Usage
-----
python scripts/analyze_emg.py --hdf5 data/<session>.hdf5 [options]

Options
-------
--hdf5       Path to HDF5 session file (required)
--start      Start offset in seconds from session start (default: auto = densest window)
--duration   Window length in seconds (default: 3.0)
--out_dir    Directory to save plots (default: plots/emg_analysis)
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

# ── UCLA styling ─────────────────────────────────────────────────────────────
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

# ── colour palette ──────────────────────────────────────────────────────────
# Diverging: Dark Blue → White → UCLA Gold  (for correlation matrices, -1 to +1)
CMAP_DIVERGE = matplotlib.colors.LinearSegmentedColormap.from_list(
    "ucla_diverge", ['#003B5C', '#2774AE', '#8BB8EE', '#FFFFFF', '#FFD100', '#FFB81C', '#FF8C00'],
)
# Sequential: Dark Blue → White (at 0.5) → UCLA Gold  (pinned midpoint)
CMAP_SEQ = matplotlib.colors.LinearSegmentedColormap.from_list(
    "ucla_seq", [
        (0.00, '#003B5C'),
        (0.25, '#2774AE'),
        (0.50, '#FFFFFF'),
        (0.75, '#FFD100'),
        (1.00, '#FFB81C'),
    ],
)
CMAP_SEQ_BLUE = CMAP_SEQ
# 16-channel line colors cycling through UCLA palette
_ucla_cycle = matplotlib.colors.LinearSegmentedColormap.from_list(
    "ucla16", ['#8BB8EE', '#2774AE', '#005587', '#003B5C', '#FFD100', '#FFB81C'], N=16
)
CMAP_16 = _ucla_cycle
BAND_COLORS = {"Right hand": UCLA_COLORS[0], "Left hand": UCLA_COLORS[1]}
FS = 2000  # nominal sampling rate (Hz)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_window(hdf5_path: Path, start_abs: float, duration: float):
    """Return (emg_left, emg_right, times_rel, keystrokes_in_window, t0_abs)."""
    with h5py.File(hdf5_path, "r") as f:
        grp = f["emg2qwerty"]
        ts = grp["timeseries"]
        t_all = ts["time"][:]
        t0 = t_all[0]

        w0 = start_abs
        w1 = start_abs + duration
        i0, i1 = np.searchsorted(t_all, [w0, w1])
        i1 = min(i1, len(t_all))

        emg_l = ts["emg_left"][i0:i1].astype(np.float32)
        emg_r = ts["emg_right"][i0:i1].astype(np.float32)
        t_win = t_all[i0:i1] - w0

        ks_raw = json.loads(grp.attrs.get("keystrokes", "[]"))
        ks_win = [k for k in ks_raw if w0 <= k["start"] <= w1]

    return emg_l, emg_r, t_win, ks_win, w0


def find_dense_window(hdf5_path: Path, duration: float = 3.0) -> float:
    """Return the absolute timestamp of the densest `duration`-second typing window."""
    with h5py.File(hdf5_path, "r") as f:
        grp = f["emg2qwerty"]
        t0 = grp["timeseries"]["time"][0]
        ks_raw = json.loads(grp.attrs.get("keystrokes", "[]"))

    times = np.array([k["start"] for k in ks_raw])
    best_t, best_n = t0, 0
    for kt in times[:500]:
        n = int(((times >= kt) & (times < kt + duration)).sum())
        if n > best_n:
            best_n, best_t = n, kt
    return float(best_t)


def keystroke_label(k: dict) -> str:
    key = k.get("key", "")
    if key == "Key.space":
        return "␣"
    if len(key) == 1:
        return key
    return "·"


# ── figure 1: enhanced signal viewer ─────────────────────────────────────────

def plot_signal(emg_l, emg_r, t_win, ks_win, out_path: Path) -> None:
    n_ch = emg_l.shape[1]
    spacing = 6.0  # vertical separation in normalised units

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    fig.suptitle("sEMG Signal — Raw Channels with Keystroke Markers", fontsize=13, y=0.98)

    for ax, emg, label, color in [
        (axes[0], emg_r, "Right hand", BAND_COLORS["Right hand"]),
        (axes[1], emg_l, "Left hand",  BAND_COLORS["Left hand"]),
    ]:
        # Normalise: subtract per-channel mean, divide by session std
        emg_c = emg - emg.mean(axis=0, keepdims=True)
        scale = emg_c.std() or 1.0
        emg_n = emg_c / scale

        for ch in range(n_ch):
            offset = ch * spacing
            ax.plot(t_win, emg_n[:, ch] + offset,
                    color=CMAP_16(ch), lw=0.6, alpha=0.85, rasterized=True)
            ax.text(-0.01, offset, f"{ch+1}",
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=6.5, color=CMAP_16(ch))

        # Keystroke markers
        ylim_top = (n_ch - 0.2) * spacing
        for ks in ks_win:
            tr = ks["start"] - (t_win[0] + (ks_win[0]["start"] - ks_win[0]["start"]))
            # t_win is already relative to w0
            tr = ks["start"] - (t_win[0] + float(np.searchsorted(t_win, 0)))
            tr = ks["start"] - (t_win[0] + 0)  # t_win[0] ≈ 0
            # correct: w0 was already subtracted when building t_win
            # just use ks["start"] relative to the same origin
            pass

        # Recompute properly using the offsets embedded in t_win
        # t_win[i] = t_all[i0+i] - w0,  ks["start"] is absolute → tr = ks["start"] - w0
        # But w0 is not stored here; however t_win[0] ≈ 0 and w0 = t_win[0] + first_abs_time
        # We passed ks_win filtered by w0 <= ks["start"] <= w1, so just:
        # tr = ks["start"] - ks_win[0]["start"] + (ks_win[0]["start"] - w0)
        # Simplest: the caller already gave us ks relative to w0 via load_window
        # We'll recompute tr outside the loop using a closure variable set in __main__

        ax.set_ylabel(label, fontsize=10, labelpad=6)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2, linestyle="--")

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()
    _save(fig, out_path)


def plot_signal_v2(emg_l, emg_r, t_win, ks_win, w0: float, out_path: Path) -> None:
    """Correct version with keystroke times passed in."""
    n_ch = emg_l.shape[1]
    spacing = 6.0

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    fig.suptitle("sEMG Signal — Raw Channels with Keystroke Markers", fontsize=13, y=0.98)

    for ax, emg, label, color in [
        (axes[0], emg_r, "Right hand", BAND_COLORS["Right hand"]),
        (axes[1], emg_l, "Left hand",  BAND_COLORS["Left hand"]),
    ]:
        emg_c = emg - emg.mean(axis=0, keepdims=True)
        scale = emg_c.std() or 1.0
        emg_n = emg_c / scale

        for ch in range(n_ch):
            offset = ch * spacing
            ax.plot(t_win, emg_n[:, ch] + offset,
                    color=CMAP_16(ch), lw=0.6, alpha=0.85, rasterized=True)
            ax.text(-0.005, offset, f"{ch+1}",
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=6.5, color=CMAP_16(ch))

        ylim_top = (n_ch + 0.3) * spacing
        ax.set_ylim(-spacing, ylim_top)

        for ks in ks_win:
            tr = ks["start"] - w0
            ax.axvline(tr, color="black", lw=0.7, alpha=0.45, linestyle="--", zorder=5)
            lbl = keystroke_label(ks)
            ax.text(tr, ylim_top * 0.97, lbl,
                    ha="center", va="top", fontsize=8, color="#333333",
                    fontweight="bold")

        ax.set_ylabel(label, fontsize=10, labelpad=6)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2, linestyle="--")

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()
    _save(fig, out_path)


# ── figure 2: per-channel FFT / power spectral density ───────────────────────

def plot_fft(emg_l, emg_r, out_path: Path) -> None:
    n_ch = emg_l.shape[1]

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"hspace": 0.15},
    )
    fig.suptitle("Power Spectral Density — All Channels (Welch method)", fontsize=13)

    for ax, emg, label in [
        (axes[0], emg_r, "Right hand"),
        (axes[1], emg_l, "Left hand"),
    ]:
        for ch in range(n_ch):
            freqs, psd = scipy_signal.welch(
                emg[:, ch], fs=FS, nperseg=512, noverlap=256,
            )
            ax.semilogy(freqs, psd, color=CMAP_16(ch), lw=1.0,
                        alpha=0.85, label=f"{ch+1}")

        ax.set_ylabel(f"{label}\nPSD (V²/Hz)", fontsize=9)
        ax.set_xlim(0, FS / 2)
        ax.set_ylim(bottom=1e-6)
        ax.grid(True, which="both", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", ncol=4, fontsize=6.5,
                  framealpha=0.7, columnspacing=0.5)

    axes[-1].set_xlabel("Frequency (Hz)", fontsize=10)
    plt.tight_layout()
    _save(fig, out_path)


def plot_fft_heatmap(emg_l, emg_r, out_path: Path) -> None:
    """PSD as a colour heatmap: channels × frequency."""
    n_ch = emg_l.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    fig.suptitle("PSD Heatmap — Channel × Frequency", fontsize=13)

    for ax, emg, label in [
        (axes[0], emg_r, "Right hand"),
        (axes[1], emg_l, "Left hand"),
    ]:
        psd_mat = []
        for ch in range(n_ch):
            freqs, psd = scipy_signal.welch(
                emg[:, ch], fs=FS, nperseg=512, noverlap=256,
            )
            psd_mat.append(psd)
        psd_mat = np.array(psd_mat)  # (C, freq)
        psd_db = 10 * np.log10(psd_mat + 1e-12)

        im = ax.imshow(
            psd_db,
            aspect="auto",
            origin="lower",
            extent=[freqs[0], freqs[-1], 0.5, n_ch + 0.5],
            cmap=CMAP_SEQ_BLUE,
            interpolation="nearest",
        )
        ax.set_yticks(range(1, n_ch + 1))
        ax.set_yticklabels([f"{i+1}" for i in range(n_ch)], fontsize=7)
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_title(label, fontsize=10)
        fig.colorbar(im, ax=ax, label="dB", fraction=0.046, pad=0.04)

    plt.tight_layout()
    _save(fig, out_path)


# ── figure 3: cross-correlation matrix ───────────────────────────────────────

def plot_crosscorr_matrix(emg_l, emg_r, out_path: Path) -> None:
    """Pearson correlation matrix for both bands side by side."""
    n_ch = emg_l.shape[1]
    labels = [f"{i+1}" for i in range(n_ch)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cross-Correlation Matrix (Pearson) Between Electrode Channels", fontsize=13)

    for ax, emg, title in [
        (axes[0], emg_r, "Right hand"),
        (axes[1], emg_l, "Left hand"),
    ]:
        corr = np.corrcoef(emg.T)  # (C, C)

        im = ax.imshow(corr, vmin=0, vmax=1, cmap=CMAP_SEQ, aspect="auto")
        ax.set_xticks(range(n_ch))
        ax.set_xticklabels(labels, rotation=90, fontsize=9)
        ax.set_yticks(range(n_ch))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Channel Number")
        ax.set_ylabel("Channel Number")

        # Annotate cells
        for i in range(n_ch):
            for j in range(n_ch):
                val = corr[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color="black" if abs(val) < 0.7 else "white")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

    plt.tight_layout()
    _save(fig, out_path)


# ── figure 4: cross-correlation lags (selected channel pairs) ────────────────

def plot_crosscorr_lags(emg_l, emg_r, out_path: Path, max_lag_ms: float = 50.0) -> None:
    """
    For each band, compute time-lagged cross-correlation between neighbouring
    electrode channels (ch1↔ch2, ch2↔ch3, …, ch15↔ch16) and display as a
    heatmap of lag × channel-pair.
    """
    max_lag = int(max_lag_ms * FS / 1000)
    lags = np.arange(-max_lag, max_lag + 1) / FS * 1000  # in ms
    n_ch = emg_l.shape[1]
    pairs = [(i, i + 1) for i in range(n_ch - 1)]
    pair_labels = [f"{i+1}↔{j+1}" for i, j in pairs]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(
        f"Time-Lagged Cross-Correlation — Adjacent Channel Pairs (±{max_lag_ms:.0f} ms)",
        fontsize=13,
    )

    for ax, emg, title in [
        (axes[0], emg_r, "Right hand"),
        (axes[1], emg_l, "Left hand"),
    ]:
        cc_mat = []
        for i, j in pairs:
            x = emg[:, i].astype(np.float64)
            y = emg[:, j].astype(np.float64)
            x -= x.mean(); y -= y.mean()
            full_cc = np.correlate(x, y, mode="full")
            # Normalise to [-1, 1]
            norm = np.sqrt((x ** 2).sum() * (y ** 2).sum()) or 1.0
            full_cc /= norm
            mid = len(full_cc) // 2
            cc_mat.append(full_cc[mid - max_lag : mid + max_lag + 1])

        cc_mat = np.array(cc_mat)  # (pairs, lags)

        im = ax.imshow(
            cc_mat,
            aspect="auto",
            origin="lower",
            extent=[lags[0], lags[-1], -0.5, len(pairs) - 0.5],
            cmap=CMAP_SEQ,
            vmin=0, vmax=1,
            interpolation="nearest",
        )
        ax.axvline(0, color="black", lw=0.8, linestyle="--", alpha=0.6)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pair_labels, fontsize=7)
        ax.set_xlabel("Lag (ms)", fontsize=9)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalised CC")

    plt.tight_layout()
    _save(fig, out_path)


# ── figure 5: per-keystroke-triggered average ────────────────────────────────

def plot_keystroke_average(
    emg_l, emg_r, t_win, ks_win, w0: float,
    out_path: Path,
    pre_ms: float = 200.0,
    post_ms: float = 400.0,
) -> None:
    """
    Epoch the signal around each keystroke and plot the trial-averaged waveform
    ± std for every channel.  Good for understanding the muscle activation pattern.
    """
    pre  = int(pre_ms  * FS / 1000)
    post = int(post_ms * FS / 1000)
    n_ch = emg_l.shape[1]
    epoch_t = np.linspace(-pre_ms, post_ms, pre + post) / 1000  # in seconds

    fig, axes = plt.subplots(
        2, n_ch, figsize=(22, 5), sharex=True, sharey="row",
        gridspec_kw={"hspace": 0.35, "wspace": 0.08},
    )
    fig.suptitle(
        f"Keystroke-Triggered Average  (pre={pre_ms:.0f} ms, post={post_ms:.0f} ms)",
        fontsize=13,
    )

    for row, (emg, label) in enumerate([(emg_r, "Right"), (emg_l, "Left")]):
        epochs_per_ch = [[] for _ in range(n_ch)]

        for ks in ks_win:
            tr = ks["start"] - w0
            center = np.searchsorted(t_win, tr)
            s, e = center - pre, center + post
            if s < 0 or e > len(emg):
                continue
            epoch = emg[s:e, :]  # (pre+post, C)
            for ch in range(n_ch):
                epochs_per_ch[ch].append(epoch[:, ch])

        for ch in range(n_ch):
            ax = axes[row, ch]
            if not epochs_per_ch[ch]:
                ax.set_visible(False)
                continue
            stack = np.stack(epochs_per_ch[ch])  # (trials, time)
            mu  = stack.mean(axis=0)
            std = stack.std(axis=0)
            ax.plot(epoch_t, mu, color=CMAP_16(ch), lw=1.2)
            ax.fill_between(epoch_t, mu - std, mu + std,
                            color=CMAP_16(ch), alpha=0.2)
            ax.axvline(0, color="black", lw=0.7, linestyle="--", alpha=0.5)
            ax.set_title(f"{ch+1}", fontsize=7, pad=2)
            ax.set_xticks([-0.1, 0, 0.2])
            ax.tick_params(labelsize=6)
            if ch == 0:
                ax.set_ylabel(f"{label} hand\nAmplitude", fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("t (s)", fontsize=7)

    plt.tight_layout()
    _save(fig, out_path)


# ── figure 7: cross-correlation matrix between hands ─────────────────────────

def plot_crosscorr_between_hands(emg_l, emg_r, out_path: Path) -> None:
    """
    16×16 Pearson correlation matrix with left-hand channels on one axis and
    right-hand channels on the other.  Reveals which electrode pairs across
    wrists share the most signal (e.g. due to synchronous bilateral typing).
    """
    n_ch = emg_l.shape[1]
    labels = [f"{i+1}" for i in range(n_ch)]

    # Stack into (T, 32), transpose to (32, T), compute full (32, 32) corr
    combined = np.concatenate([emg_r, emg_l], axis=1).T  # (32, T)
    corr_full = np.corrcoef(combined)                     # (32, 32)
    corr_rl = corr_full[:n_ch, n_ch:]                     # (16_right, 16_left)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Cross-Correlation Between Hands (Right ch × Left ch)", fontsize=13)

    im = ax.imshow(corr_rl, vmin=-1, vmax=1, cmap=CMAP_DIVERGE, aspect="auto")
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(labels, rotation=90, fontsize=9)
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Left Hand — Channel Number")
    ax.set_ylabel("Right Hand — Channel Number")

    for i in range(n_ch):
        for j in range(n_ch):
            val = corr_rl[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=4.5, color="black" if abs(val) < 0.6 else "white")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    plt.tight_layout()
    _save(fig, out_path)


# ── figure 8: time-lagged cross-correlation between hands ─────────────────────

def plot_crosscorr_lags_between_hands(
    emg_l, emg_r, out_path: Path, max_lag_ms: float = 100.0
) -> None:
    """
    For every right-hand channel, find the left-hand channel it correlates with
    most strongly at zero lag, then plot the full lag profile for that pair.
    Also shows a heatmap of peak-lag (ms) for all 16×16 pairs so you can see
    whether one hand systematically leads the other.
    """
    n_ch = emg_l.shape[1]
    max_lag = int(max_lag_ms * FS / 1000)
    lags = np.arange(-max_lag, max_lag + 1) / FS * 1000  # ms

    # --- compute all 16×16 lag profiles ---
    # peak_lag[i, j] = lag in ms at which R-ch(i) and L-ch(j) correlate most
    peak_lag   = np.zeros((n_ch, n_ch))
    peak_val   = np.zeros((n_ch, n_ch))

    for i in range(n_ch):
        x = emg_r[:, i].astype(np.float64)
        x -= x.mean()
        for j in range(n_ch):
            y = emg_l[:, j].astype(np.float64)
            y -= y.mean()
            full_cc = np.correlate(x, y, mode="full")
            norm = np.sqrt((x**2).sum() * (y**2).sum()) or 1.0
            full_cc /= norm
            mid = len(full_cc) // 2
            window = full_cc[mid - max_lag : mid + max_lag + 1]
            best_idx = np.argmax(np.abs(window))
            peak_lag[i, j] = lags[best_idx]
            peak_val[i, j] = window[best_idx]

    labels = [f"{i+1}" for i in range(n_ch)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"Between-Hand Cross-Correlation (±{max_lag_ms:.0f} ms)",
        fontsize=13,
    )

    # Left panel: peak correlation value heatmap
    ax = axes[0]
    im0 = ax.imshow(peak_val, vmin=-1, vmax=1, cmap=CMAP_DIVERGE, aspect="auto")
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(labels, rotation=90, fontsize=9)
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Peak Correlation Value")
    ax.set_xlabel("Left Hand — Channel Number")
    ax.set_ylabel("Right Hand — Channel Number")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04, label="Peak |r|")

    # Right panel: lag at peak (ms) — positive = right leads left
    ax = axes[1]
    abs_max = np.percentile(np.abs(peak_lag), 95)  # robust colour scale
    im1 = ax.imshow(peak_lag, vmin=-abs_max, vmax=abs_max, cmap=CMAP_DIVERGE, aspect="auto")
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(labels, rotation=90, fontsize=9)
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Lag at Peak (ms)\n[positive = Right leads Left]")
    ax.set_xlabel("Left Hand — Channel Number")
    ax.set_ylabel("Right Hand — Channel Number")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label="Lag (ms)")

    plt.tight_layout()
    _save(fig, out_path)


# ── utility ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--start", type=float, default=None,
                        help="Start offset (seconds from session start). "
                             "Default: auto-selects densest typing window.")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--out_dir", type=Path, default=Path("plots/emg_analysis"))
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        t0_abs = float(f["emg2qwerty"]["timeseries"]["time"][0])

    if args.start is None:
        w0_abs = find_dense_window(args.hdf5, args.duration)
        print(f"Auto-selected window at t0 + {w0_abs - t0_abs:.2f}s")
    else:
        w0_abs = t0_abs + args.start

    emg_l, emg_r, t_win, ks_win, w0 = load_window(args.hdf5, w0_abs, args.duration)
    print(f"Loaded {len(emg_l)} samples, {len(ks_win)} keystrokes")

    out = args.out_dir

    print("\n[1/5] Signal visualization ...")
    plot_signal_v2(emg_l, emg_r, t_win, ks_win, w0, out / "1_signal.png")

    print("[2/5] FFT overlay ...")
    plot_fft(emg_l, emg_r, out / "2_fft_overlay.png")

    print("[3/5] FFT heatmap ...")
    plot_fft_heatmap(emg_l, emg_r, out / "3_fft_heatmap.png")

    print("[4/5] Cross-correlation matrix ...")
    plot_crosscorr_matrix(emg_l, emg_r, out / "4_crosscorr_matrix.png")

    print("[5/5] Cross-correlation lags ...")
    plot_crosscorr_lags(emg_l, emg_r, out / "5_crosscorr_lags.png")

    print("[6/8] Keystroke-triggered average ...")
    plot_keystroke_average(emg_l, emg_r, t_win, ks_win, w0,
                           out / "6_keystroke_average.png")

    print("[7/8] Cross-correlation matrix between hands ...")
    plot_crosscorr_between_hands(emg_l, emg_r, out / "7_crosscorr_between_hands.png")

    print("[8/8] Time-lagged cross-correlation between hands ...")
    plot_crosscorr_lags_between_hands(emg_l, emg_r, out / "8_crosscorr_lags_between_hands.png")

    print(f"\nAll plots saved to {out}/")


if __name__ == "__main__":
    main()
