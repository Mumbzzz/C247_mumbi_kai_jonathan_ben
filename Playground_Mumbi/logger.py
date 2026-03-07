"""Shared experiment logging utility for the C247 team.

Writes experiment results to two CSV files per model architecture:
    results/results_summary_{MODEL}.csv  — one row per run, scalar metrics
    results/results_curves_{MODEL}.csv   — one row per epoch, training curves

The results/ directory is created at the workspace root automatically.

Typical usage
-------------
    from Playground_Mumbi.logger import log_epoch, log_summary, make_run_id

    run_id = make_run_id(
        model="CNN",
        num_channels=8,
        sampling_rate_hz=2000,
        train_fraction=0.5,
    )

    # Call once per epoch during training:
    log_epoch(run_id, model="CNN", epoch=1,
              train_loss=0.42, val_loss=0.50,
              train_cer=12.3, val_cer=15.7)

    # Call once after training completes:
    log_summary(run_id, model="CNN", epochs=50,
                num_channels=8, sampling_rate_hz=2000,
                train_fraction=0.5, input_type="spectrogram",
                final_train_loss=0.12, final_val_loss=0.18,
                final_train_cer=3.2, final_val_cer=5.8,
                test_cer=6.1, training_time_sec=1823.4,
                notes="baseline run")
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Workspace root is two levels above this file (…/C247_mumbikaijonathanben/)
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _WORKSPACE_ROOT / "results"

# Valid model names accepted by this logger
_VALID_MODELS = {"CNN", "RNN", "CNN_LSTM"}

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

_SUMMARY_COLUMNS = [
    "run_id",
    "timestamp",
    "model",
    "epochs",
    "num_channels",
    "sampling_rate_hz",
    "train_fraction",
    "input_type",
    "final_train_loss",
    "final_val_loss",
    "final_train_cer",
    "final_val_cer",
    "test_cer",
    "training_time_sec",
    "notes",
]

_CURVES_COLUMNS = [
    "run_id",
    "epoch",
    "train_loss",
    "val_loss",
    "train_cer",
    "val_cer",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _results_dir() -> Path:
    """Return the results directory, creating it if necessary."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _RESULTS_DIR


def _csv_path(model: str, kind: str) -> Path:
    """Return the CSV path for *model* and *kind* ('summary' or 'curves').

    Args:
        model: One of 'CNN', 'RNN', 'CNN_LSTM'.
        kind:  Either 'summary' or 'curves'.

    Returns:
        Absolute Path to the CSV file.

    Raises:
        ValueError: If *model* is not a recognised model name.
    """
    if model not in _VALID_MODELS:
        raise ValueError(
            f"Unknown model '{model}'. Expected one of: {sorted(_VALID_MODELS)}"
        )
    return _results_dir() / f"results_{kind}_{model}.csv"


def _append_row(csv_path: Path, columns: list[str], row: dict) -> None:
    """Append *row* to *csv_path*, writing the header if the file is new.

    Args:
        csv_path: Destination CSV file.
        columns:  Ordered list of column names (also used as header).
        row:      Mapping of column name to value.
    """
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in columns})

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_run_id(
    model: str,
    num_channels: int,
    sampling_rate_hz: int,
    train_fraction: float,
    timestamp: str | None = None,
) -> str:
    """Build a canonical run identifier.

    Format: ``{MODEL}_{num_channels}ch_{sampling_rate}hz_{pct}pct_{timestamp}``
    Example: ``CNN_8ch_2000hz_50pct_20260306_225638``

    Args:
        model:            Model name, e.g. 'CNN'.
        num_channels:     Number of EMG channels.
        sampling_rate_hz: Sampling rate in Hz.
        train_fraction:   Fraction of data used for training (0.0 – 1.0).
        timestamp:        Optional pre-formatted timestamp string
                          (``YYYYMMDD_HHMMSS``). Generated from the current
                          time when omitted.

    Returns:
        Run ID string.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pct = int(round(train_fraction * 100))
    return f"{model}_{num_channels}ch_{sampling_rate_hz}hz_{pct}pct_{timestamp}"


def log_epoch(
    run_id: str,
    model: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_cer: float,
    val_cer: float,
) -> None:
    """Append one epoch's training curve values to the curves CSV.

    Creates ``results/results_curves_{model}.csv`` with a header row on the
    first call; subsequent calls append without re-writing the header.

    Args:
        run_id:     Unique run identifier (see :func:`make_run_id`).
        model:      Model name — 'CNN', 'RNN', or 'CNN_LSTM'.
        epoch:      Epoch number (1-indexed recommended).
        train_loss: Mean training CTC loss for this epoch.
        val_loss:   Mean validation CTC loss for this epoch.
        train_cer:  Training character error rate (%) for this epoch.
        val_cer:    Validation character error rate (%) for this epoch.
    """
    row = {
        "run_id":     run_id,
        "epoch":      epoch,
        "train_loss": train_loss,
        "val_loss":   val_loss,
        "train_cer":  train_cer,
        "val_cer":    val_cer,
    }
    _append_row(_csv_path(model, "curves"), _CURVES_COLUMNS, row)


def log_summary(
    run_id: str,
    model: str,
    epochs: int,
    num_channels: int,
    sampling_rate_hz: int,
    train_fraction: float,
    input_type: str,
    final_train_loss: float,
    final_val_loss: float,
    final_train_cer: float,
    final_val_cer: float,
    test_cer: float,
    training_time_sec: float,
    notes: str = "",
) -> None:
    """Append a run summary row to the summary CSV.

    Creates ``results/results_summary_{model}.csv`` with a header row on the
    first call; subsequent calls append without re-writing the header.

    Args:
        run_id:             Unique run identifier (see :func:`make_run_id`).
        model:              Model name — 'CNN', 'RNN', or 'CNN_LSTM'.
        epochs:             Total number of training epochs completed.
        num_channels:       Number of EMG electrode channels.
        sampling_rate_hz:   EMG sampling rate in Hz.
        train_fraction:     Fraction of data used for training (0.0 – 1.0).
        input_type:         Description of the input representation,
                            e.g. 'spectrogram' or 'raw'.
        final_train_loss:   Training CTC loss at the last epoch.
        final_val_loss:     Validation CTC loss at the last epoch.
        final_train_cer:    Training CER (%) at the last epoch.
        final_val_cer:      Validation CER (%) at the last epoch.
        test_cer:           CER (%) on the held-out test set.
        training_time_sec:  Wall-clock training duration in seconds.
        notes:              Optional free-text annotation for this run.
    """
    row = {
        "run_id":            run_id,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":             model,
        "epochs":            epochs,
        "num_channels":      num_channels,
        "sampling_rate_hz":  sampling_rate_hz,
        "train_fraction":    train_fraction,
        "input_type":        input_type,
        "final_train_loss":  final_train_loss,
        "final_val_loss":    final_val_loss,
        "final_train_cer":   final_train_cer,
        "final_val_cer":     final_val_cer,
        "test_cer":          test_cer,
        "training_time_sec": training_time_sec,
        "notes":             notes,
    }
    _append_row(_csv_path(model, "summary"), _SUMMARY_COLUMNS, row)
