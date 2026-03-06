"""RNN-based EMG-to-keystroke model for Playground_Kai.

Architecture:
    SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten → BiLSTM → Linear → LogSoftmax

Input  shape: (T, N, 2, 16, freq)   — time-first batch of log-spectrogram EMG windows
Output shape: (T, N, num_classes)   — log-softmax activations for use with nn.CTCLoss

The preprocessing blocks (SpectrogramNorm + MultiBandRotationInvariantMLP) are
reused directly from emg2qwerty.modules; only the temporal encoder is replaced
with a Bidirectional LSTM, making the model better at capturing long-range
sequential dependencies than the original TDS-conv design.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Ensure workspace root is on sys.path so `emg2qwerty` is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from emg2qwerty.charset import charset
from emg2qwerty.modules import MultiBandRotationInvariantMLP, SpectrogramNorm


class RNNEncoder(nn.Module):
    """Bidirectional LSTM encoder for EMG-to-keystroke CTC decoding.

    This module accepts the same input format produced by
    ``emg2qwerty.data.WindowedEMGDataset`` after the ``LogSpectrogram``
    transform — i.e. a time-first (TNC) batch of shape
    ``(T, N, num_bands, electrode_channels, freq_bins)``.

    The forward pass is:
    1. ``SpectrogramNorm``: per-channel 2-D batch normalisation over
       ``(N, freq, T)`` slices, one normaliser per electrode channel.
    2. ``MultiBandRotationInvariantMLP``: rotation-invariant MLP applied
       independently per band, mapping each time-step feature vector from
       ``(electrode_channels * freq_bins,)`` → ``mlp_features[-1]``.
    3. Flatten bands: ``(T, N, 2, F)`` → ``(T, N, 2F)``.
    4. BiLSTM stack: ``(T, N, 2F)`` → ``(T, N, 2 * hidden_size)``.
    5. Linear projection + LogSoftmax: ``(T, N, num_classes)``.

    No temporal downsampling occurs, so ``emission_lengths == input_lengths``
    when computing CTC loss.

    Args:
        in_features: Flattened feature count per band per time step fed to the
            MLP.  Equals ``(n_fft // 2 + 1) * electrode_channels``.
            With the default ``LogSpectrogram(n_fft=64)`` this is
            ``33 * 16 = 528``.
        mlp_features: Hidden layer widths for the rotation-invariant MLP
            applied per band.  The last entry sets the per-band feature
            dimension going into the RNN.
        hidden_size: Hidden size of each LSTM direction.  The concatenated
            bidirectional output has dimension ``2 * hidden_size``.
        num_layers: Number of stacked BiLSTM layers.
        dropout: Dropout probability applied *between* LSTM layers.
            Automatically disabled when ``num_layers == 1``.
    """

    NUM_BANDS: int = 2
    ELECTRODE_CHANNELS: int = 16

    def __init__(
        self,
        in_features: int = 528,
        mlp_features: Sequence[int] = (384,),
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # --- Preprocessing (reused from emg2qwerty) ---

        # Per-channel batch normalisation over (T, N, 2, 16, freq)
        self.spec_norm = SpectrogramNorm(
            channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS  # 32
        )

        # Per-band, rotation-invariant MLP:
        # (T, N, 2, 16, freq) → (T, N, 2, mlp_features[-1])
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=list(mlp_features),
            num_bands=self.NUM_BANDS,
        )

        # --- Temporal encoder ---

        rnn_in = self.NUM_BANDS * list(mlp_features)[-1]  # e.g. 2 * 384 = 768

        # Bidirectional LSTM preserves sequence length T
        # (T, N, rnn_in) → (T, N, 2 * hidden_size)
        self.rnn = nn.LSTM(
            input_size=rnn_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,  # TNC convention throughout
        )

        # --- Classification head ---

        # (T, N, 2 * hidden_size) → (T, N, num_classes)
        self.head = nn.Linear(2 * hidden_size, charset().num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the encoder.

        Args:
            inputs: Float tensor of shape ``(T, N, 2, 16, freq)`` — a
                time-first padded batch of log-spectrogram EMG windows.

        Returns:
            Log-softmax activations of shape ``(T, N, num_classes)``.
            Suitable to pass directly to ``torch.nn.CTCLoss``.
        """
        x = self.spec_norm(inputs)   # (T, N, 2, 16, freq)
        x = self.mlp(x)              # (T, N, 2, mlp_features[-1])
        x = x.flatten(start_dim=2)   # (T, N, 2 * mlp_features[-1])
        x = x.contiguous()           # cuDNN LSTM requires contiguous memory
        x, _ = self.rnn(x)           # (T, N, 2 * hidden_size)
        x = self.head(x)             # (T, N, num_classes)
        return F.log_softmax(x, dim=-1)
