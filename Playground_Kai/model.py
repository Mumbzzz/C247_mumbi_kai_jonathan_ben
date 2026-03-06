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

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchaudio.models import Conformer as TorchAudioConformer

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


# ---------------------------------------------------------------------------
# Conformer encoder
# ---------------------------------------------------------------------------

class ConformerEncoder(nn.Module):
    """Conformer-based encoder for EMG-to-keystroke CTC decoding.

    Drop-in replacement for ``RNNEncoder``: identical preprocessing front-end
    (SpectrogramNorm → MultiBandRotationInvariantMLP) and classification head,
    but the temporal encoder is ``torchaudio.models.Conformer`` instead of a
    BiLSTM.  Each block combines local depthwise convolution with global
    multi-head self-attention, outperforming both TDS-Conv and BiLSTM on
    tasks with both local and long-range temporal structure.

    Architecture::

        SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten
        → Linear (input_proj) → Sinusoidal PE
        → torchaudio.models.Conformer (N blocks)
        → Linear (head) → LogSoftmax

    ``torchaudio.models.Conformer`` is batch-first (NTC); inputs are transposed
    around the call and transposed back, preserving TNC convention throughout
    the rest of the module.

    .. note:: Self-attention is O(T²).  Whole-session test sequences (~40k
        frames) will OOM.  At test time use a finite ``window_length`` or
        chunk the session before passing it to the model.

    Args:
        in_features: Flattened per-band feature dim fed to the MLP.
            Default 528 = 33 freq bins × 16 electrode channels.
        mlp_features: Hidden widths for the rotation-invariant MLP per band.
        d_model: Uniform feature dimension throughout all Conformer blocks.
        num_heads: Number of attention heads.  Must evenly divide ``d_model``.
        num_layers: Number of stacked Conformer blocks.
        conv_kernel_size: Depthwise conv kernel size (must be odd).
        dropout: Dropout probability used across all sub-modules.
    """

    NUM_BANDS: int = 2
    ELECTRODE_CHANNELS: int = 16

    def __init__(
        self,
        in_features: int = 528,
        mlp_features: Sequence[int] = (384,),
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        # --- Preprocessing (identical to RNNEncoder) ---
        self.spec_norm = SpectrogramNorm(
            channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS  # 32
        )
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=list(mlp_features),
            num_bands=self.NUM_BANDS,
        )

        # --- Input projection: 2 * mlp_features[-1] → d_model ---
        rnn_in = self.NUM_BANDS * list(mlp_features)[-1]  # e.g. 768
        self.input_proj = nn.Linear(rnn_in, d_model)

        # --- Conformer blocks (via torchaudio) ---
        self.conformer = TorchAudioConformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=4 * d_model,   # standard 4× expansion ratio
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        # --- Classification head ---
        self.head = nn.Linear(d_model, charset().num_classes)

    @staticmethod
    def _sinusoidal_pe(T: int, d_model: int, device: torch.device) -> torch.Tensor:
        """Return a ``(T, 1, d_model)`` sinusoidal positional encoding."""
        pos = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(T, 1, d_model, device=device)
        pe[:, 0, 0::2] = torch.sin(pos * div)
        cos_len = pe[:, 0, 1::2].shape[-1]
        pe[:, 0, 1::2] = torch.cos(pos * div[:cos_len])
        return pe

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            inputs: Float tensor of shape ``(T, N, 2, 16, freq)``.

        Returns:
            Log-softmax activations of shape ``(T, N, num_classes)``.
        """
        x = self.spec_norm(inputs)    # (T, N, 2, 16, freq)
        x = self.mlp(x)               # (T, N, 2, mlp_features[-1])
        x = x.flatten(start_dim=2)    # (T, N, 2 * mlp_features[-1])
        x = self.input_proj(x)        # (T, N, d_model)
        T, N = x.shape[0], x.shape[1]
        x = x + self._sinusoidal_pe(T, x.shape[2], x.device)
        # torchaudio.Conformer is batch-first (N, T, D); transpose around the call
        x = x.transpose(0, 1)         # (N, T, d_model)
        lengths = torch.full((N,), T, dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths)  # (N, T, d_model)
        x = x.transpose(0, 1)         # (T, N, d_model)
        x = self.head(x)              # (T, N, num_classes)
        return F.log_softmax(x, dim=-1)
