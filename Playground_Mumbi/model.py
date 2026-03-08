"""CNN+LSTM hybrid model for EMG keystroke prediction.

Architecture:
    SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten
    → 1D CNN blocks (local temporal feature extraction)
    → BiLSTM (long-range sequential modelling)
    → Linear → LogSoftmax

Input  shape: (T, N, 2, 16, freq)  — time-first batch of log-spectrogram EMG windows
Output shape: (T, N, num_classes)  — log-softmax activations for use with nn.CTCLoss

The CNN front-end learns local temporal patterns in the EMG spectrogram.
The BiLSTM then models long-range sequential dependencies — the "state"
s_t = f(s_{t-1}, x_t), with LSTM cell state acting as the gradient highway
that prevents vanishing gradients over the sequence (analogous to ResNet skip
connections).  BiLSTM is justified because we process fixed-length windows,
so future context within the window is available.

Hyperparameters exposed for tuner:
    cnn_channels: width of CNN feature maps
    cnn_kernel:   kernel size for 1D convs
    cnn_layers:   number of CNN blocks (1, 2, or 3)
    lstm_hidden:  BiLSTM hidden size (per direction)
    lstm_layers:  number of stacked BiLSTM layers
    dropout:      applied after CNN blocks and between LSTM layers
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from emg2qwerty.modules import MultiBandRotationInvariantMLP, SpectrogramNorm


class CNNBlock1D(nn.Module):
    """Single 1D convolutional block: Conv1d → BatchNorm1d → GELU → Dropout.

    Operates on ``(N, channels, T)`` tensors.  Same-padding preserves the
    time dimension so the BiLSTM downstream sees T unchanged.

    GELU is chosen over ReLU because it is smoother and empirically works well
    with sequence models.

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count.
        kernel_size:  Conv1d kernel width.
        dropout:      Dropout probability applied after activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2  # same-padding to preserve time dimension
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNLSTMModel(nn.Module):
    """CNN + BiLSTM hybrid model for CTC-based EMG keystroke decoding.

    Drop-in replacement for ``RNNEncoder``: identical preprocessing front-end
    (SpectrogramNorm → MultiBandRotationInvariantMLP), but the temporal encoder
    is a stack of 1D CNN blocks followed by a BiLSTM rather than a bare BiLSTM.
    The CNN blocks extract local temporal features before the BiLSTM models
    long-range sequential dependencies.

    Input  shape: ``(T, N, 2, 16, freq)``  — time-first padded batch of
                  log-spectrogram EMG windows.
    Output shape: ``(T, N, num_classes)``  — log-softmax activations,
                  ready for ``nn.CTCLoss``.

    No temporal downsampling occurs, so ``emission_lengths == input_lengths``
    when computing CTC loss.

    Args:
        in_features:  Flattened feature count per band per time step fed to
            the MLP.  With ``LogSpectrogram(n_fft=64)`` this is
            ``33 * 16 = 528``.
        mlp_features: Hidden layer widths for the rotation-invariant MLP
            applied per band.  The last entry sets the per-band feature
            dimension going into the CNN.
        num_bands:    Number of frequency bands (one per hand, default 2).
        cnn_channels: Number of channels in each 1D CNN block.
        cnn_kernel:   Kernel width for 1D convolutions.
        cnn_layers:   Number of stacked CNN blocks (1–3).
        lstm_hidden:  Hidden units per direction in the BiLSTM.
        lstm_layers:  Number of stacked BiLSTM layers.
        dropout:      Dropout probability applied after CNN blocks, between
            LSTM layers, and on the final LSTM output.
        num_classes:  Output vocabulary size (28: a–z + blank + 1 extra).
    """

    NUM_BANDS: int = 2
    ELECTRODE_CHANNELS: int = 16

    def __init__(
        self,
        in_features: int = 528,
        mlp_features: list[int] = [384],
        num_bands: int = 2,
        cnn_channels: int = 256,
        cnn_kernel: int = 3,
        cnn_layers: int = 2,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 28,
    ) -> None:
        super().__init__()

        # --- Preprocessing (reused from emg2qwerty, identical to RNNEncoder) ---

        # Per-channel batch normalisation over (T, N, 2, 16, freq)
        self.spec_norm = SpectrogramNorm(
            channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS  # 32
        )

        # Per-band rotation-invariant MLP:
        # (T, N, 2, 16, freq) → (T, N, 2, mlp_features[-1])
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=num_bands,
        )

        flat_features = num_bands * mlp_features[-1]  # e.g. 2 * 384 = 768

        # --- 1D CNN feature extractor ---
        # Stacks cnn_layers blocks; same-padding preserves time dimension T.
        # (T, N, flat_features) → permute → (N, flat_features, T)
        #                       → CNN     → (N, cnn_channels, T)
        cnn_blocks = []
        in_ch = flat_features
        for _ in range(cnn_layers):
            cnn_blocks.append(CNNBlock1D(in_ch, cnn_channels, cnn_kernel, dropout))
            in_ch = cnn_channels
        self.cnn = nn.Sequential(*cnn_blocks)

        # --- BiLSTM temporal encoder ---
        # bidirectional=True: justified because windowed inference means future
        # context within the window is available — matching RNNEncoder's choice.
        # lstm_layers > 1 stacks BiLSTMs; dropout applied *between* layers.
        # (T, N, cnn_channels) → (T, N, 2 * lstm_hidden)
        self.rnn = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=False,   # TNC convention throughout
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.rnn_dropout = nn.Dropout(dropout)

        # --- Classification head ---
        # (T, N, 2 * lstm_hidden) → (T, N, num_classes)
        self.head = nn.Linear(lstm_hidden * 2, num_classes)

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

        # CNN expects (N, channels, T)
        x = x.permute(1, 2, 0)      # (N, flat_features, T)
        x = self.cnn(x)              # (N, cnn_channels, T)

        # BiLSTM expects (T, N, features)
        x = x.permute(2, 0, 1)      # (T, N, cnn_channels)
        x = x.contiguous()           # cuDNN LSTM requires contiguous memory
        x, _ = self.rnn(x)           # (T, N, 2 * lstm_hidden)
        x = self.rnn_dropout(x)

        x = self.head(x)             # (T, N, num_classes)
        return F.log_softmax(x, dim=-1)
