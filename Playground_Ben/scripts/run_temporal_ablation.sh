#!/usr/bin/env bash
# run_temporal_ablation.sh
#
# Trains one model per temporal downsampling factor (2, 4, 8, 16) and then
# generates a summary ablation plot.
#
# Run from the repo root with the venv active:
#
#   source /home/benforbes/emg2qwerty/venv/bin/activate
#   pip install -e .   # only needed once to point the package at this repo
#   bash Playground_Ben/scripts/run_temporal_ablation.sh
#
# Optional: override the user config (default: single_user)
#   USER_CFG=multi_user bash Playground_Ben/scripts/run_temporal_ablation.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
PLAYGROUND="${REPO_ROOT}/Playground_Ben"

# ── 1. Apply playground patches to the main emg2qwerty module ─────────────────
echo "==> Patching emg2qwerty module with Playground_Ben versions..."
cp "${PLAYGROUND}/emg2qwerty/transforms.py" "${REPO_ROOT}/emg2qwerty/transforms.py"
cp "${PLAYGROUND}/emg2qwerty/lightning.py"  "${REPO_ROOT}/emg2qwerty/lightning.py"

# ── 2. Copy new transform configs into the main config tree ───────────────────
echo "==> Copying temporal downsample configs..."
cp "${PLAYGROUND}/config/transforms"/temporal_downsample_*.yaml \
   "${REPO_ROOT}/config/transforms/"

# ── 3. Train — one job per factor ─────────────────────────────────────────────
# window_length and padding scale by factor so the TDS encoder always receives
# ~250 spectrogram frames (baseline 8000 / hop 16 = 500; half that after ds).
# These MUST be CLI overrides — Hydra's _self_ (base.yaml) wins over config-group files.
#
#   factor | window_length | padding       | spec frames after ds
#   -------+---------------+---------------+---------------------
#     2    |    8000       | [1800, 200]   | 8000/2/16 = 250
#     4    |   16000       | [3600, 400]   | 16000/4/16 = 250
#     8    |   32000       | [7200, 800]   | 32000/8/16 = 250
#    16    |   64000       | [14400, 1600] | 64000/16/16 = 250

USER_CFG="${USER_CFG:-single_user}"

declare -A WINDOW_LENGTHS=([2]=8000   [4]=16000  [8]=32000  [16]=64000)
declare -A PAD_LEFT=(      [2]=1800   [4]=3600   [8]=7200   [16]=14400)
declare -A PAD_RIGHT=(     [2]=200    [4]=400    [8]=800    [16]=1600)

FACTORS=(2 4 8 16)
LOG_DIRS=()

cd "${REPO_ROOT}"

for FACTOR in "${FACTORS[@]}"; do
    echo ""
    echo "================================================================"
    echo "  Training: temporal_downsample_${FACTOR}  (user=${USER_CFG})"
    echo "  window_length=${WINDOW_LENGTHS[$FACTOR]}"
    echo "  padding=[${PAD_LEFT[$FACTOR]},${PAD_RIGHT[$FACTOR]}]"
    echo "================================================================"

    python -m emg2qwerty.train \
        transforms="temporal_downsample_${FACTOR}" \
        user="${USER_CFG}" \
        datamodule.window_length="${WINDOW_LENGTHS[$FACTOR]}" \
        "datamodule.padding=[${PAD_LEFT[$FACTOR]},${PAD_RIGHT[$FACTOR]}]"

    # Capture the log dir written by Hydra (most recent timestamp dir)
    LATEST_LOG=$(ls -td "${REPO_ROOT}/logs"/*/* 2>/dev/null | head -1)
    LOG_DIRS+=("${LATEST_LOG}")
    echo "  --> Saved to: ${LATEST_LOG}"
done

# ── 4. Plot the ablation summary ──────────────────────────────────────────────
echo ""
echo "==> Generating temporal ablation plot..."

LABELS=()
for FACTOR in "${FACTORS[@]}"; do
    LABELS+=("${FACTOR}x ($(( 2000 / FACTOR )) Hz)")
done

python "${PLAYGROUND}/scripts/plot_results.py" channel_ablation \
    --channels "${FACTORS[@]}" \
    --log_dirs  "${LOG_DIRS[@]}" \
    --labels    "${LABELS[@]}" \
    --out       "${PLAYGROUND}/plots/temporal_ablation.png"

echo ""
echo "Done. Summary plot: ${PLAYGROUND}/plots/temporal_ablation.png"
echo "Individual training logs:"
for i in "${!FACTORS[@]}"; do
    echo "  factor=${FACTORS[$i]}: ${LOG_DIRS[$i]}"
done
