"""
Scans logs/ for completed temporal downsampling runs and prints the factor
with the lowest final val/CER to stdout. Used by run_channel_ablation.sh.

Usage:
    python Playground_Ben/scripts/find_best_temporal_factor.py
"""

import sys
import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator, SCALARS,
)

FACTORS = [2, 4, 8, 16]

def last_val_cer(log_dir: Path) -> float:
    tb_dir = log_dir / "lightning_logs" / "version_0"
    event_files = sorted(glob.glob(str(tb_dir / "events.out.tfevents.*")))
    if not event_files:
        return float("nan")
    best = float("nan")
    for ef in event_files:
        ea = EventAccumulator(ef, size_guidance={SCALARS: 0})
        ea.Reload()
        events = ea.Scalars("val/CER") if "val/CER" in ea.Tags().get("scalars", []) else []
        if events:
            best = events[-1].value
    return best


def find_best_run(factor: int) -> Path | None:
    candidates = []
    for d in sorted((REPO_ROOT / "logs").glob("*/*"), reverse=True):
        hydra = d / "hydra_configs" / "hydra.yaml"
        if not hydra.exists():
            continue
        text = hydra.read_text()
        if f"temporal_downsample_{factor}" not in text:
            continue
        if factor > 2 and "window_length" not in text:
            continue
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
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


results = {}
for factor in FACTORS:
    log_dir = find_best_run(factor)
    if log_dir is None:
        continue
    cer = last_val_cer(log_dir)
    if not (cer != cer):  # skip NaN
        results[factor] = (cer, log_dir)

if not results:
    print("ERROR: no completed temporal runs found", file=sys.stderr)
    sys.exit(1)

best_factor = min(results, key=lambda f: results[f][0])
best_cer, best_dir = results[best_factor]

# Print factor to stdout (captured by shell script)
print(best_factor)

# Print human-readable summary to stderr (not captured)
print(f"\nTemporal ablation results:", file=sys.stderr)
for f in sorted(results):
    cer, d = results[f]
    marker = " <-- BEST" if f == best_factor else ""
    print(f"  factor={f}x ({2000//f} Hz): val/CER={cer:.2f}%  [{d.name}]{marker}",
          file=sys.stderr)
