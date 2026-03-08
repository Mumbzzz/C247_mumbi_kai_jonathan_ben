"""
Reads TensorBoard logs from the channel ablation runs and writes results
into the shared team logger CSVs (results/results_summary_CNN.csv and
results/results_curves_CNN.csv).

Channel ablation configs:
    log_spectrogram  → 16 ch/hand (baseline)
    channel_stride2  →  8 ch/hand (every other electrode)
    channel_stride4  →  4 ch/hand (every 4th electrode)
    channel_stride8  →  2 ch/hand (every 8th electrode)

Run from the repo root:
    python Playground_Ben/scripts/log_channel_results.py
"""

import sys
import glob
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
EMG_LOGS    = Path("/home/benforbes/emg2qwerty/logs")
sys.path.insert(0, str(REPO_ROOT))

from scripts.logger import log_epoch, log_summary, make_run_id
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator, SCALARS,
)

# ── run registry ──────────────────────────────────────────────────────────────
# Maps num_channels → transform tag used in hydra config
CHANNEL_CONFIGS = {
    16: "log_spectrogram",
    8:  "channel_stride2",
    4:  "channel_stride4",
    2:  "channel_stride8",
}

def _find_best_run(transform_tag: str) -> Path | None:
    """Return the most recent log dir matching transform_tag with a best checkpoint."""
    search_roots = [EMG_LOGS, REPO_ROOT / "logs"]
    candidates = []
    for root in search_roots:
        if not root.exists():
            continue
        for d in sorted(root.glob("*/*"), reverse=True):
            hydra = d / "hydra_configs" / "hydra.yaml"
            if not hydra.exists():
                continue
            text = hydra.read_text()
            if transform_tag not in text:
                continue
            # Exclude 1-epoch test runs
            ckpts = [c for c in (d / "checkpoints").glob("*.ckpt")
                     if c.name != "last.ckpt"]
            if not ckpts:
                continue
            best_epoch = max(
                int(c.stem.split("epoch=")[1].split("-")[0]) for c in ckpts
            )
            if best_epoch < 10:
                continue
            candidates.append((best_epoch, d))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


# ── TensorBoard helpers (same as log_temporal_results.py) ────────────────────

def load_scalars(log_dir: Path) -> tuple[dict, tuple[float, float]]:
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
    step_to_epoch = {e.step: int(e.value) for e in scalars.get("epoch", [])}
    result = {}
    for event in scalars.get(tag, []):
        epoch = step_to_epoch.get(event.step)
        if epoch is not None:
            result[epoch] = event.value
    return result


def last_value(scalars: dict, tag: str) -> float:
    events = scalars.get(tag, [])
    return events[-1].value if events else float("nan")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    for num_channels, transform_tag in CHANNEL_CONFIGS.items():
        log_dir = _find_best_run(transform_tag)
        print(f"\n--- {num_channels} ch/hand  [{transform_tag}] ---")

        if log_dir is None:
            print(f"  SKIP: no completed run found for '{transform_tag}'")
            continue
        print(f"  Using: {log_dir}")

        try:
            scalars, (t_start, t_end) = load_scalars(log_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        training_time_sec = t_end - t_start

        train_loss_by_epoch = per_epoch_series(scalars, "train/loss")
        val_loss_by_epoch   = per_epoch_series(scalars, "val/loss")
        val_cer_by_epoch    = per_epoch_series(scalars, "val/CER")

        all_epochs  = sorted(
            set(train_loss_by_epoch) | set(val_loss_by_epoch) | set(val_cer_by_epoch)
        )
        total_epochs = max(all_epochs) if all_epochs else 0

        final_train_loss = last_value(scalars, "train/loss")
        final_val_loss   = last_value(scalars, "val/loss")
        final_val_cer    = last_value(scalars, "val/CER")
        test_cer         = last_value(scalars, "test/CER")

        print(f"  epochs={total_epochs}  train_loss={final_train_loss:.4f}"
              f"  val_loss={final_val_loss:.4f}  val_CER={final_val_cer:.2f}%"
              f"  test_CER={test_cer:.2f}%  time={training_time_sec/3600:.2f}h")

        date_str = log_dir.parent.name.replace("-", "")
        time_str = log_dir.name.replace("-", "")
        ts       = f"{date_str}_{time_str}"
        run_id   = make_run_id("CNN", num_channels, 2000, 1.0, timestamp=ts)

        for epoch in all_epochs:
            log_epoch(
                run_id=run_id,
                model="CNN",
                epoch=epoch,
                train_loss=train_loss_by_epoch.get(epoch, float("nan")),
                val_loss=val_loss_by_epoch.get(epoch, float("nan")),
                val_cer=val_cer_by_epoch.get(epoch, float("nan")),
            )

        log_summary(
            run_id=run_id,
            model="CNN",
            epochs=total_epochs,
            num_channels=num_channels,
            sampling_rate_hz=2000,
            train_fraction=1.0,
            input_type="spectrogram",
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            final_val_cer=final_val_cer,
            test_cer=test_cer,
            training_time_sec=training_time_sec,
            notes="ablation_channels",
        )
        print(f"  Logged run_id: {run_id}")

    print(f"\nDone. Results written to {REPO_ROOT / 'results'}/")


if __name__ == "__main__":
    main()
