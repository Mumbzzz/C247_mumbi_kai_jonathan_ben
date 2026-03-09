#!/usr/bin/env bash
# run_channel_ablation.sh
#
# Runs channel ablation for the CNN+LSTM hybrid model at 2000 Hz, 100% data.
# Trains four configurations: 16, 8, 4, 2 electrodes per hand.
#
# Channel selection mirrors Ben's 2000 Hz ablation (same stride indices):
#   16 ch/hand — all 16 electrodes (no ChannelSelect)
#    8 ch/hand — every other:  [0, 2, 4, 6, 8, 10, 12, 14]
#    4 ch/hand — every 4th:    [0, 4, 8, 12]
#    2 ch/hand — every 8th:    [0, 8]
#
# Run from the repo root with your venv active:
#   bash Playground_Mumbi/scripts/run_channel_ablation.sh
#
# Optional: pass --from-hyperparams to load tuned CNN+LSTM hyperparams, e.g.:
#   HYPERPARAMS=Playground_Mumbi/checkpoints/best_hyperparams_cnn_lstm.yaml \
#   bash Playground_Mumbi/scripts/run_channel_ablation.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
PLAYGROUND="${REPO_ROOT}/Playground_Mumbi"

EPOCHS="${EPOCHS:-80}"
HYPERPARAMS="${HYPERPARAMS:-}"
HP_FLAG=""
if [[ -n "${HYPERPARAMS}" ]]; then
    HP_FLAG="--from-hyperparams ${HYPERPARAMS}"
    echo "==> Loading hyperparams from: ${HYPERPARAMS}"
fi
echo "==> Epochs per run: ${EPOCHS}"

cd "${REPO_ROOT}"

# ── 1. 16 ch/hand — all electrodes (baseline) ────────────────────────────────
echo ""
echo "================================================================"
echo "  Training: 16 ch/hand (all electrodes) — 2000 Hz, 100% data"
echo "================================================================"
python -m Playground_Mumbi.train \
    --model cnn_lstm \
    --num-channels 16 \
    --train-fraction 1.0 \
    --notes ablation_channels \
    ${HP_FLAG}

# ── 2. 8 ch/hand — every other electrode (stride 2, auto-computed) ───────────
echo ""
echo "================================================================"
echo "  Training: 8 ch/hand (stride 2) — 2000 Hz, 100% data"
echo "================================================================"
python -m Playground_Mumbi.train \
    --model cnn_lstm \
    --num-channels 8 \
    --train-fraction 1.0 \
    --notes ablation_channels \
    ${HP_FLAG}

# ── 3. 4 ch/hand — every 4th electrode (stride 4, auto-computed) ─────────────
echo ""
echo "================================================================"
echo "  Training: 4 ch/hand (stride 4) — 2000 Hz, 100% data"
echo "================================================================"
python -m Playground_Mumbi.train \
    --model cnn_lstm \
    --num-channels 4 \
    --train-fraction 1.0 \
    --notes ablation_channels \
    ${HP_FLAG}

# ── 4. 2 ch/hand — every 8th electrode (stride 8, auto-computed) ─────────────
echo ""
echo "================================================================"
echo "  Training: 2 ch/hand (stride 8) — 2000 Hz, 100% data"
echo "================================================================"
python -m Playground_Mumbi.train \
    --model cnn_lstm \
    --num-channels 2 \
    --train-fraction 1.0 \
    --notes ablation_channels \
    ${HP_FLAG}

# ── 5. Generate plots ─────────────────────────────────────────────────────────
echo ""
echo "==> Generating channel ablation plots..."
python "${PLAYGROUND}/scripts/plot_channel_ablation.py"

echo ""
echo "Done. Plots saved to ${PLAYGROUND}/plots/"
