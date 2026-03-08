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
# from torchaudio.models import Conformer as TorchAudioConformer  # replaced by hand-built RPE Conformer

# # To try out mamba-ssm's efficient causal Conv1D for sequence modeling, 
# # which could be a drop-in replacement for the BiLSTM in this module.
# from mamba_ssm import Mamba

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

    def __init__(
        self,
        in_features: int = 528,
        mlp_features: Sequence[int] = (384,),
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        electrode_channels: int = 16,
    ) -> None:
        super().__init__()

        # --- Preprocessing (reused from emg2qwerty) ---

        # Per-channel batch normalisation over (T, N, 2, electrode_channels, freq)
        self.spec_norm = SpectrogramNorm(
            channels=self.NUM_BANDS * electrode_channels
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
# Relative positional encoding helpers
# ---------------------------------------------------------------------------

def _rel_sinusoidal_pe(T: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Return a ``(2T-1, d_model)`` sinusoidal relative positional encoding.

    Index ``k`` corresponds to signed relative distance ``k - (T-1)``, so the
    table covers distances ``-(T-1)`` (far past) through ``+(T-1)`` (far
    future).  Called once per forward pass so it adapts automatically to any
    sequence length without caching.
    """
    distances = torch.arange(-(T - 1), T, dtype=torch.float32, device=device)  # (2T-1,)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(2 * T - 1, d_model, device=device)
    pe[:, 0::2] = torch.sin(distances.unsqueeze(1) * div.unsqueeze(0))
    cos_len = pe[:, 1::2].shape[-1]
    pe[:, 1::2] = torch.cos(distances.unsqueeze(1) * div[:cos_len].unsqueeze(0))
    return pe


class RPEMultiHeadAttention(nn.Module):
    """Multi-head self-attention with Transformer-XL style relative positional
    encoding (RPE).

    Pre-softmax scores are the sum of two terms:

    - **Content-to-content**: ``(Q + u) @ K^T``  — standard attention biased
      by a learnable global query vector ``u``.
    - **Content-to-position**: ``_rel_shift((Q + v) @ R^T)``  — each query
      attends to relative position embeddings ``R`` biased by a learnable
      global vector ``v``.  The ``_rel_shift`` gather trick rearranges the
      ``(N, H, T, 2T-1)`` raw logits into the ``(N, H, T, T)`` score matrix
      without any extra memory allocation.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.  Must evenly divide ``d_model``.
        dropout: Dropout applied to attention weights and output projection.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        # Projects RPE table into positional-key space; no bias (position-only signal)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable global content/position biases (Transformer-XL, eq. 4)
        self.u = nn.Parameter(torch.zeros(num_heads, self.d_head))  # content bias
        self.v = nn.Parameter(torch.zeros(num_heads, self.d_head))  # position bias

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    @staticmethod
    def _rel_shift(BD_raw: torch.Tensor) -> torch.Tensor:
        """Skew ``(N, H, T, 2T-1)`` → ``(N, H, T, T)`` via gather.

        For query at row ``i`` and key at column ``j`` we want
        ``BD_raw[n, h, i, i - j + (T-1)]``.  Building the gather index once
        and collecting all entries in a single pass achieves this in O(T²)
        time and memory.
        """
        N, H, T, _ = BD_raw.shape
        # j_idx[j] = T-1-j; i_idx[i] = i  →  src[i,j] = i - j + (T-1)
        j_idx = torch.arange(T - 1, -1, -1, device=BD_raw.device)   # (T,)
        i_idx = torch.arange(T, device=BD_raw.device).unsqueeze(1)   # (T, 1)
        src = (j_idx + i_idx).expand(N, H, T, T)                     # (N, H, T, T)
        return BD_raw.gather(-1, src)

    def forward(self, x: torch.Tensor, rel_pe: torch.Tensor) -> torch.Tensor:
        """Compute RPE multi-head self-attention.

        Args:
            x: ``(N, T, d_model)`` input sequence (batch-first NTC).
            rel_pe: ``(2T-1, d_model)`` relative sinusoidal embeddings.

        Returns:
            ``(N, T, d_model)`` output.
        """
        N, T, _ = x.shape
        H, d = self.num_heads, self.d_head

        def _split(mat: torch.Tensor) -> torch.Tensor:
            """(N, T, d_model) → (N, H, T, d_head)."""
            return mat.view(N, T, H, d).transpose(1, 2)

        Q = _split(self.W_Q(x))   # (N, H, T, d_head)
        K = _split(self.W_K(x))   # (N, H, T, d_head)
        V = _split(self.W_V(x))   # (N, H, T, d_head)

        # --- Content-to-content ---
        # u: (H, d_head) → (1, H, 1, d_head) for broadcast over N and T
        AC = (Q + self.u.unsqueeze(0).unsqueeze(2)) @ K.transpose(-1, -2)  # (N, H, T, T)

        # --- Content-to-position ---
        # Project RPE table: (2T-1, d_model) → (H, d_head, 2T-1)
        R = self.pos_proj(rel_pe)                              # (2T-1, d_model)
        R = R.view(2 * T - 1, H, d).permute(1, 2, 0)          # (H, d_head, 2T-1)
        # v: (H, d_head) → (1, H, 1, d_head) for broadcast
        BD_raw = (Q + self.v.unsqueeze(0).unsqueeze(2)) @ R.unsqueeze(0)   # (N, H, T, 2T-1)
        BD = self._rel_shift(BD_raw)                           # (N, H, T, T)

        # --- Scaled dot-product attention ---
        scores = (AC + BD) * self.scale                        # (N, H, T, T)
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = attn @ V                                         # (N, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(N, T, H * d)   # (N, T, d_model)
        return self.out_dropout(self.W_O(out))


class _FeedForward(nn.Module):
    """Conformer macaron feed-forward sub-layer (pre-LN).

    Applies ``LayerNorm → Linear → SiLU → Dropout → Linear → Dropout`` and
    returns ``0.5 × output`` so that the caller's residual connection
    ``x = x + ff(x)`` naturally implements the half-step scaling from the
    Conformer paper.

    Args:
        d_model: Input/output feature dimension.
        ffn_dim: Hidden width.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.net(self.norm(x))


class _ConvModule(nn.Module):
    """Conformer convolution sub-layer (pre-LN).

    Applies::

        LayerNorm → Pointwise(×2) → GLU → Depthwise Conv1d
        → BatchNorm1d → SiLU → Pointwise → Dropout

    The depthwise convolution captures local temporal structure; padding is
    ``kernel_size // 2`` so the output length equals the input length.

    Args:
        d_model: Feature dimension.
        kernel_size: Depthwise conv kernel size (must be odd).
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "conv_kernel_size must be odd"
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_expand = nn.Linear(d_model, 2 * d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_contract = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``(N, T, d_model)``."""
        out = self.norm(x)
        out = self.pointwise_expand(out)           # (N, T, 2*d_model)
        out = F.glu(out, dim=-1)                   # (N, T, d_model)
        out = out.transpose(1, 2)                  # (N, d_model, T) — Conv1d format
        out = self.depthwise(out)                  # (N, d_model, T)
        out = self.batch_norm(out)
        out = F.silu(out)
        out = out.transpose(1, 2)                  # (N, T, d_model)
        out = self.pointwise_contract(out)
        return self.dropout(out)


class RPEConformerBlock(nn.Module):
    """Single Conformer block with Transformer-XL relative positional encoding.

    Macaron structure::

        x = x + FF1(x)                      # half-step FF (pre-LN, 0.5× scale)
        x = x + Attn(LN(x), rel_pe)         # RPE multi-head self-attention
        x = x + Conv(x)                     # depthwise conv module
        x = x + FF2(x)                      # half-step FF (pre-LN, 0.5× scale)
        x = FinalLN(x)

    The relative positional embedding table ``rel_pe`` is computed once per
    forward pass by ``ConformerEncoder`` and threaded into every block.

    Args:
        d_model: Feature dimension.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward hidden dimension.
        conv_kernel_size: Depthwise conv kernel size (must be odd).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff1      = _FeedForward(d_model, ffn_dim, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn     = RPEMultiHeadAttention(d_model, num_heads, dropout)
        self.conv     = _ConvModule(d_model, conv_kernel_size, dropout)
        self.ff2      = _FeedForward(d_model, ffn_dim, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, rel_pe: torch.Tensor) -> torch.Tensor:
        """x: ``(N, T, d_model)``; rel_pe: ``(2T-1, d_model)``."""
        x = x + self.ff1(x)
        x = x + self.attn(self.attn_norm(x), rel_pe)
        x = x + self.conv(x)
        x = x + self.ff2(x)
        return self.final_norm(x)


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

    def __init__(
        self,
        in_features: int = 528,
        mlp_features: Sequence[int] = (384,),
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        electrode_channels: int = 16,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        # --- Preprocessing (identical to RNNEncoder) ---
        self.spec_norm = SpectrogramNorm(
            channels=self.NUM_BANDS * electrode_channels
        )
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=list(mlp_features),
            num_bands=self.NUM_BANDS,
        )

        # --- Input projection: 2 * mlp_features[-1] → d_model ---
        rnn_in = self.NUM_BANDS * list(mlp_features)[-1]  # e.g. 768
        self.input_proj = nn.Linear(rnn_in, d_model)

        # --- Conformer blocks (hand-built RPE version) ---
        # [torchaudio path kept for reference — commented out]
        # self.conformer = TorchAudioConformer(
        #     input_dim=d_model,
        #     num_heads=num_heads,
        #     ffn_dim=4 * d_model,   # standard 4× expansion ratio
        #     num_layers=num_layers,
        #     depthwise_conv_kernel_size=conv_kernel_size,
        #     dropout=dropout,
        # )
        self.layers = nn.ModuleList([
            RPEConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=4 * d_model,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # --- Classification head ---
        self.head = nn.Linear(d_model, charset().num_classes)

    # [Absolute sinusoidal PE — kept for reference; replaced by _rel_sinusoidal_pe]
    # @staticmethod
    # def _sinusoidal_pe(T: int, d_model: int, device: torch.device) -> torch.Tensor:
    #     """Return a ``(T, 1, d_model)`` sinusoidal positional encoding."""
    #     pos = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
    #     div = torch.exp(
    #         torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
    #         * (-math.log(10000.0) / d_model)
    #     )
    #     pe = torch.zeros(T, 1, d_model, device=device)
    #     pe[:, 0, 0::2] = torch.sin(pos * div)
    #     cos_len = pe[:, 0, 1::2].shape[-1]
    #     pe[:, 0, 1::2] = torch.cos(pos * div[:cos_len])
    #     return pe

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
        # [Absolute PE line kept for reference — relative PE is now threaded into each block]
        # x = x + self._sinusoidal_pe(T, x.shape[2], x.device)

        # Compute relative positional embedding table once for this sequence length
        rel_pe = _rel_sinusoidal_pe(T, x.shape[2], x.device)   # (2T-1, d_model)

        # Conformer blocks expect batch-first (N, T, d_model)
        x = x.transpose(0, 1)         # (T, N, d_model) → (N, T, d_model)
        for layer in self.layers:
            x = layer(x, rel_pe)       # (N, T, d_model)
        x = x.transpose(0, 1)         # (N, T, d_model) → (T, N, d_model)

        # [torchaudio path kept for reference — commented out]
        # lengths = torch.full((N,), T, dtype=torch.long, device=x.device)
        # x, _ = self.conformer(x, lengths)  # (N, T, d_model)
        # x = x.transpose(0, 1)         # (T, N, d_model)

        x = self.head(x)              # (T, N, num_classes)
        return F.log_softmax(x, dim=-1)
