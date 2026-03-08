#!/usr/bin/env bash
# run_channel_ablation.sh
#
# Trains 8, 4, 2 ch/hand models using the best temporal config found from
# the temporal ablation study (bandpass + best downsample factor).
# The 16 ch/hand baseline is the best temporal run itself (reused, not retrained).
#
# Run from the repo root with the venv active:
#
#   source /home/benforbes/emg2qwerty/venv/bin/activate
#   pip install -e .   # only needed once
#   bash Playground_Ben/scripts/run_channel_ablation.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
PLAYGROUND="${REPO_ROOT}/Playground_Ben"
USER_CFG="${USER_CFG:-single_user}"

# ── 1. Patch main emg2qwerty module ──────────────────────────────────────────
echo "==> Patching emg2qwerty module with Playground_Ben versions..."
cp "${PLAYGROUND}/emg2qwerty/transforms.py" "${REPO_ROOT}/emg2qwerty/transforms.py"
cp "${PLAYGROUND}/emg2qwerty/lightning.py"  "${REPO_ROOT}/emg2qwerty/lightning.py"

# ── 2. Find best temporal factor ─────────────────────────────────────────────
echo "==> Finding best temporal factor from completed runs..."
BEST_FACTOR=$(python "${PLAYGROUND}/scripts/find_best_temporal_factor.py" 2>/dev/tty)

if [[ -z "${BEST_FACTOR}" ]]; then
    echo "ERROR: could not determine best temporal factor. Run temporal study first."
    exit 1
fi

echo "==> Best temporal factor: ${BEST_FACTOR}x ($(( 2000 / BEST_FACTOR )) Hz)"

# window_length and padding matching the best temporal factor
declare -A WINDOW_LENGTHS=([1]=8000 [2]=8000  [4]=16000  [8]=32000  [16]=64000)
declare -A PAD_LEFT=(      [1]=1800 [2]=1800  [4]=3600   [8]=7200   [16]=14400)
declare -A PAD_RIGHT=(     [1]=200  [2]=200   [4]=400    [8]=800    [16]=1600)

WIN="${WINDOW_LENGTHS[$BEST_FACTOR]}"
PAD_L="${PAD_LEFT[$BEST_FACTOR]}"
PAD_R="${PAD_RIGHT[$BEST_FACTOR]}"

# ── 3. Generate combined channel configs with the best temporal factor ────────
# Pipeline: ToTensor → BandRotation → TemporalJitter → Bandpass
#           → TemporalDownsample(BEST_FACTOR) → LogSpec → ChannelSelect → SpecAugment

echo "==> Writing combined channel configs (factor=${BEST_FACTOR})..."

# Config map: transform_name → channel_indices, num_channels
declare -A CHANNEL_CONFIGS
CHANNEL_CONFIGS["channel_stride2"]="[0, 2, 4, 6, 8, 10, 12, 14]"
CHANNEL_CONFIGS["channel_stride4"]="[0, 4, 8, 12]"
CHANNEL_CONFIGS["channel_stride8"]="[0, 8]"

declare -A NUM_CHANNELS
NUM_CHANNELS["channel_stride2"]=8
NUM_CHANNELS["channel_stride4"]=4
NUM_CHANNELS["channel_stride8"]=2

FREQ_BINS=33  # n_fft//2+1 for n_fft=64

for TRANSFORM in channel_stride2 channel_stride4 channel_stride8; do
    INDICES="${CHANNEL_CONFIGS[$TRANSFORM]}"
    OUT="${PLAYGROUND}/config/transforms/${TRANSFORM}.yaml"

    cat > "${OUT}" <<YAML
# @package _global_
# Channel ablation with best temporal config (factor=${BEST_FACTOR}, $(( 2000 / BEST_FACTOR )) Hz).
# Pipeline: bandpass (20-460 Hz) → downsample → logspec → channel_select
# window_length=${WIN}  padding=[${PAD_L},${PAD_R}]  (must be CLI overrides)

to_tensor:
  _target_: emg2qwerty.transforms.ToTensor
  fields: [emg_left, emg_right]

band_rotation:
  _target_: emg2qwerty.transforms.ForEach
  transform:
    _target_: emg2qwerty.transforms.RandomBandRotation
    offsets: [-1, 0, 1]

temporal_jitter:
  _target_: emg2qwerty.transforms.TemporalAlignmentJitter
  max_offset: 120

bandpass:
  _target_: emg2qwerty.transforms.BandpassFilter
  low_hz: 20.0
  high_hz: 460.0
  sample_rate: 2000
  num_taps: 101

temporal_downsample:
  _target_: emg2qwerty.transforms.TemporalDownsample
  factor: ${BEST_FACTOR}

logspec:
  _target_: emg2qwerty.transforms.LogSpectrogram
  n_fft: 64
  hop_length: 16

channel_select:
  _target_: emg2qwerty.transforms.ChannelSelect
  indices: ${INDICES}

specaug:
  _target_: emg2qwerty.transforms.SpecAugment
  n_time_masks: 3
  time_mask_param: 25
  n_freq_masks: 2
  freq_mask_param: 4

transforms:
  train:
    - \${to_tensor}
    - \${band_rotation}
    - \${temporal_jitter}
    - \${bandpass}
    - \${temporal_downsample}
    - \${logspec}
    - \${channel_select}
    - \${specaug}
  val:
    - \${to_tensor}
    - \${bandpass}
    - \${temporal_downsample}
    - \${logspec}
    - \${channel_select}
  test: \${transforms.val}
YAML

    echo "  Wrote: ${OUT}"
done

# ── 4. Copy updated configs into main config tree ────────────────────────────
cp "${PLAYGROUND}/config/transforms"/channel_stride*.yaml \
   "${REPO_ROOT}/config/transforms/"

# ── 5. Train 8, 4, 2 ch/hand ─────────────────────────────────────────────────
cd "${REPO_ROOT}"

for TRANSFORM in channel_stride2 channel_stride4 channel_stride8; do
    NCH="${NUM_CHANNELS[$TRANSFORM]}"
    IN_FEATURES=$(( NCH * FREQ_BINS ))

    echo ""
    echo "================================================================"
    echo "  Training: ${TRANSFORM}  factor=${BEST_FACTOR}x  (user=${USER_CFG})"
    echo "  window_length=${WIN}  padding=[${PAD_L},${PAD_R}]"
    echo "  num_channels=${NCH}  in_features=${IN_FEATURES}"
    echo "================================================================"

    python -m emg2qwerty.train \
        transforms="${TRANSFORM}" \
        user="${USER_CFG}" \
        datamodule.window_length="${WIN}" \
        "datamodule.padding=[${PAD_L},${PAD_R}]" \
        ++module.in_features="${IN_FEATURES}"
done

# ── 6. Log results to team CSVs ───────────────────────────────────────────────
echo ""
echo "==> Logging channel results..."
python "${PLAYGROUND}/scripts/log_channel_results.py"

# ── 7. Generate plots ─────────────────────────────────────────────────────────
echo ""
echo "==> Generating channel ablation plots..."
python "${PLAYGROUND}/scripts/plot_channel_ablation.py"

echo ""
echo "Done. Plots saved to ${PLAYGROUND}/plots/"
