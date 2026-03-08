#!/usr/bin/env bash
# run_channel_ablation_2000hz.sh
#
# Trains 8, 4, 2 ch/hand at 2000 Hz using the original pipeline:
#   ToTensor → BandRotation → TemporalJitter → LogSpectrogram → ChannelSelect → SpecAugment
#
# No bandpass filter, no temporal downsampling.
# The 16 ch/hand baseline is the standard log_spectrogram run (already exists).
#
# Run from the repo root with the venv active:
#
#   source /home/benforbes/emg2qwerty/venv/bin/activate
#   pip install -e .   # only needed once
#   bash Playground_Ben/scripts/run_channel_ablation_2000hz.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
PLAYGROUND="${REPO_ROOT}/Playground_Ben"
USER_CFG="${USER_CFG:-single_user}"
FREQ_BINS=33  # n_fft//2+1 for n_fft=64

# ── 1. Patch main emg2qwerty module ──────────────────────────────────────────
echo "==> Patching emg2qwerty module with Playground_Ben versions..."
cp "${PLAYGROUND}/emg2qwerty/transforms.py" "${REPO_ROOT}/emg2qwerty/transforms.py"
cp "${PLAYGROUND}/emg2qwerty/lightning.py"  "${REPO_ROOT}/emg2qwerty/lightning.py"

# ── 2. Copy 2000 Hz channel configs into main config tree ────────────────────
echo "==> Copying 2000 Hz channel configs..."
cp "${PLAYGROUND}/config/transforms"/channel_stride*_2000hz.yaml \
   "${REPO_ROOT}/config/transforms/"

cd "${REPO_ROOT}"

# ── 3. Train 8, 4, 2 ch/hand at 2000 Hz ─────────────────────────────────────
declare -A NUM_CHANNELS=(
    [channel_stride2_2000hz]=8
    [channel_stride4_2000hz]=4
    [channel_stride8_2000hz]=2
)

for TRANSFORM in channel_stride2_2000hz channel_stride4_2000hz channel_stride8_2000hz; do
    NCH="${NUM_CHANNELS[$TRANSFORM]}"
    IN_FEATURES=$(( NCH * FREQ_BINS ))

    echo ""
    echo "================================================================"
    echo "  Training: ${TRANSFORM}  (user=${USER_CFG})"
    echo "  num_channels=${NCH}  in_features=${IN_FEATURES}  @ 2000 Hz"
    echo "================================================================"

    python -m emg2qwerty.train \
        transforms="${TRANSFORM}" \
        user="${USER_CFG}" \
        ++module.in_features="${IN_FEATURES}"
done

# ── 4. Log results to team CSVs ───────────────────────────────────────────────
echo ""
echo "==> Logging channel results..."
python "${PLAYGROUND}/scripts/log_channel_results.py"

# ── 5. Generate plots ─────────────────────────────────────────────────────────
echo ""
echo "==> Generating channel ablation plots..."
python "${PLAYGROUND}/scripts/plot_channel_ablation.py"

echo ""
echo "Done. Plots saved to ${PLAYGROUND}/plots/"
