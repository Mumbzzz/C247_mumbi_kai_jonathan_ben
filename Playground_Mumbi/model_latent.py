"""Latent-space CNN+LSTM model for EMG keystroke prediction.

Operates on pre-computed AE latent vectors (256-dim @ 32 ms/frame).
The SpectrogramNorm + MultiBandRotationInvariantMLP front-end from the raw-EMG
pipeline is replaced by a single nn.Linear(latent_dim, proj_features) projection,
followed by LayerNorm and Dropout.  The rest of the architecture (1D CNN blocks
→ BiLSTM → head) is identical to CNNLSTMModel.

Architecture:
    Linear(latent_dim → proj_features) → LayerNorm → Dropout
    → 1D CNN blocks (local temporal feature extraction)
    → BiLSTM (long-range sequential modelling)
    → Linear → LogSoftmax

Input  shape: (T, N, latent_dim)  — time-first batch of latent EMG vectors
Output shape: (T, N, num_classes) — log-softmax activations for nn.CTCLoss
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

from Playground_Mumbi.model import CNNBlock1D


class LatentCNNLSTMModel(nn.Module):
    """CNN + BiLSTM hybrid model operating on pre-computed latent EMG vectors.

    Drop-in replacement for ``LatentRNNEncoder`` when you want the CNN+LSTM
    hybrid architecture on latent inputs.  The spectrogram preprocessing
    front-end (SpectrogramNorm → MultiBandRotationInvariantMLP) is replaced by
    a single linear projection layer so the model operates directly on the
    256-dim AE latent vectors.

    Input  shape: ``(T, N, latent_dim)``  — time-first batch of latent vectors.
    Output shape: ``(T, N, num_classes)`` — log-softmax activations,
                  ready for ``nn.CTCLoss``.

    No temporal downsampling occurs (same-padding in CNN, LSTM preserves T),
    so ``emission_lengths == input_lengths`` when computing CTC loss.

    Args:
        latent_dim:    Dimensionality of the input latent vectors (default 256).
        proj_features: Output size of the linear projection layer fed into CNN.
        cnn_channels:  Number of channels in each 1D CNN block.
        cnn_kernel:    Kernel width for 1D convolutions.
        cnn_layers:    Number of stacked CNN blocks (1–3).
        lstm_hidden:   Hidden units per direction in the BiLSTM.
        lstm_layers:   Number of stacked BiLSTM layers.
        dropout:       Dropout probability applied after the projection, after CNN
                       blocks, between LSTM layers, and on the final LSTM output.
        num_classes:   Output vocabulary size (default 28: a–z + blank + space).
    """

    def __init__(
        self,
        latent_dim: int = 256,
        proj_features: int = 384,
        cnn_channels: int = 256,
        cnn_kernel: int = 3,
        cnn_layers: int = 2,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 28,
    ) -> None:
        super().__init__()

        # --- Linear projection (replaces SpectrogramNorm + MLP) ---
        # (T, N, latent_dim) → (T, N, proj_features)
        self.proj = nn.Linear(latent_dim, proj_features)
        self.proj_norm = nn.LayerNorm(proj_features)
        self.proj_dropout = nn.Dropout(dropout)

        # --- 1D CNN feature extractor ---
        # Same-padding preserves the time dimension T throughout.
        cnn_blocks = []
        in_ch = proj_features
        for _ in range(cnn_layers):
            cnn_blocks.append(CNNBlock1D(in_ch, cnn_channels, cnn_kernel, dropout))
            in_ch = cnn_channels
        self.cnn = nn.Sequential(*cnn_blocks)

        # --- BiLSTM temporal encoder ---
        # bidirectional=True: justified for windowed inference (future context available).
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
        """Run a forward pass through the latent CNN+LSTM encoder.

        Args:
            inputs: Float tensor of shape ``(T, N, latent_dim)`` — a
                time-first batch of pre-computed latent EMG vectors.

        Returns:
            Log-softmax activations of shape ``(T, N, num_classes)``.
            Suitable to pass directly to ``torch.nn.CTCLoss``.
        """
        # Linear projection + normalisation
        x = self.proj(inputs)        # (T, N, proj_features)
        x = self.proj_norm(x)
        x = self.proj_dropout(x)

        # CNN expects (N, channels, T)
        x = x.permute(1, 2, 0)      # (N, proj_features, T)
        x = self.cnn(x)              # (N, cnn_channels, T)

        # BiLSTM expects (T, N, features)
        x = x.permute(2, 0, 1)      # (T, N, cnn_channels)
        x = x.contiguous()           # cuDNN LSTM requires contiguous memory
        x, _ = self.rnn(x)           # (T, N, 2 * lstm_hidden)
        x = self.rnn_dropout(x)

        x = self.head(x)             # (T, N, num_classes)
        return F.log_softmax(x, dim=-1)
