"""
Reads TensorBoard logs from the temporal downsampling ablation runs and
writes results into the shared team logger CSVs (results/results_summary_CNN.csv
and results/results_curves_CNN.csv).

Run from the repo root:
    python Playground_Ben/scripts/log_temporal_results.py

TDSConvCTC is logged under model="CNN" (nearest valid model name).
"""

import sys
from pathlib import Path
import glob

# ── imports ───────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.logger import log_epoch, log_summary, make_run_id
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator, SCALARS,
)

# ── run registry ──────────────────────────────────────────────────────────────
# Auto-detects the most recent completed run for each factor by scanning logs/.
# A run is considered valid if it has the correct window_length CLI override
# and a best checkpoint (not just last.ckpt).

FACTOR_TO_HZ = {2: 1000, 4: 500, 8: 250, 16: 125}

def _find_best_run(factor: int) -> Path | None:
    """Return the most recent log dir for this factor that has a best checkpoint
    and the correct window_length override."""
    pattern = str(REPO_ROOT / "logs" / "*" / "*")
    candidates = []
    for d in sorted(Path(REPO_ROOT / "logs").glob("*/*"), reverse=True):
        hydra = d / "hydra_configs" / "hydra.yaml"
        if not hydra.exists():
            continue
        text = hydra.read_text()
        if f"temporal_downsample_{factor}" not in text:
            continue
        # Require the window_length override for factors > 2
        if factor > 2 and "window_length" not in text:
            continue
        # Require at least one non-last checkpoint
        ckpts = [c for c in (d / "checkpoints").glob("*.ckpt")
                 if c.name != "last.ckpt"]
        if not ckpts:
            continue
        candidates.append(d)
    return candidates[0] if candidates else None

RUNS = {
    factor: (_find_best_run(factor), hz)
    for factor, hz in FACTOR_TO_HZ.items()
}

MODEL       = "CNN"   # TDSConvCTC → closest valid name
NUM_CHANNELS = 16
TRAIN_FRACTION = 1.0
INPUT_TYPE  = "spectrogram"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_scalars(log_dir: Path) -> dict:
    """Read all scalar series from all event files under log_dir."""
    tb_dir = log_dir / "lightning_logs" / "version_0"
    event_files = sorted(glob.glob(str(tb_dir / "events.out.tfevents.*")))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {tb_dir}")

    merged: dict = {}
    wall_times: list[float] = []

    for ef in event_files:
        ea = EventAccumulator(ef, size_guidance={SCALARS: 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            wall_times.extend(e.wall_time for e in events)
            if tag not in merged:
                merged[tag] = events
            else:
                merged[tag] = sorted(merged[tag] + events, key=lambda e: e.step)

    return merged, (min(wall_times), max(wall_times)) if wall_times else (0, 0)


def per_epoch_series(scalars: dict, tag: str) -> dict[int, float]:
    """Return {epoch: value} from a scalar tag, deduplicating by taking last."""
    result = {}
    epoch_events = scalars.get("epoch", [])
    # Build step→epoch mapping
    step_to_epoch = {e.step: int(e.value) for e in epoch_events}

    for event in scalars.get(tag, []):
        epoch = step_to_epoch.get(event.step, None)
        if epoch is not None:
            result[epoch] = event.value
    return result


def last_value(scalars: dict, tag: str, default: float = float("nan")) -> float:
    events = scalars.get(tag, [])
    return events[-1].value if events else default


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    for factor, (rel_log_dir, sample_rate) in RUNS.items():
        log_dir = REPO_ROOT / rel_log_dir
        print(f"\n--- factor={factor}x  ({sample_rate} Hz)  [{rel_log_dir}] ---")

        try:
            scalars, (t_start, t_end) = load_scalars(log_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        training_time_sec = t_end - t_start

        # Build per-epoch series
        train_loss_by_epoch = per_epoch_series(scalars, "train/loss")
        val_loss_by_epoch   = per_epoch_series(scalars, "val/loss")
        val_cer_by_epoch    = per_epoch_series(scalars, "val/CER")

        all_epochs = sorted(
            set(train_loss_by_epoch) | set(val_loss_by_epoch) | set(val_cer_by_epoch)
        )
        total_epochs = max(all_epochs) if all_epochs else 0

        # Final / summary scalars
        final_train_loss = last_value(scalars, "train/loss")
        final_val_loss   = last_value(scalars, "val/loss")
        final_val_cer    = last_value(scalars, "val/CER")
        test_cer         = last_value(scalars, "test/CER")

        print(f"  epochs={total_epochs}  train_loss={final_train_loss:.4f}"
              f"  val_loss={final_val_loss:.4f}  val_CER={final_val_cer:.2f}%"
              f"  test_CER={test_cer:.2f}%  time={training_time_sec/3600:.2f}h")

        # Build run ID (timestamp from the log dir name)
        date_str  = log_dir.parent.name.replace("-", "")   # 20260306
        time_str  = log_dir.name.replace("-", "")           # 111009
        ts        = f"{date_str}_{time_str}"
        run_id    = make_run_id(MODEL, NUM_CHANNELS, sample_rate,
                                TRAIN_FRACTION, timestamp=ts)

        # Log per-epoch curves
        for epoch in all_epochs:
            log_epoch(
                run_id=run_id,
                model=MODEL,
                epoch=epoch,
                train_loss=train_loss_by_epoch.get(epoch, float("nan")),
                val_loss=val_loss_by_epoch.get(epoch, float("nan")),
                val_cer=val_cer_by_epoch.get(epoch, float("nan")),
            )

        # Log summary row
        log_summary(
            run_id=run_id,
            model=MODEL,
            epochs=total_epochs,
            num_channels=NUM_CHANNELS,
            sampling_rate_hz=sample_rate,
            train_fraction=TRAIN_FRACTION,
            input_type=INPUT_TYPE,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            final_val_cer=final_val_cer,
            test_cer=test_cer,
            training_time_sec=training_time_sec,
            notes="ablation_sampling_rate",
        )
        print(f"  Logged run_id: {run_id}")

    print(f"\nDone. Results written to {REPO_ROOT / 'results'}/")


if __name__ == "__main__":
    main()
