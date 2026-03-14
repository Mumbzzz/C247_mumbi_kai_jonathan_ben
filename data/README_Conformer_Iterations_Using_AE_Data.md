# Conformer_Iterations_Using_AE_Data

Conformer CTC model (v6 architecture) trained on autoencoder-reconstructed EMG data.

## File Placement

Run from the `data/` directory. The notebook expects:

```
data/
├── Conformer_Iterations_Using_AE_Data.ipynb  ← this notebook
└── 89335547/
    └── <session_id>_recons_v3.hdf5           ← one file per session (18 total)
```

## Dependencies

- `emg2qwerty/charset.py` — loaded via `sys.path.insert(0, '..')`, so the repo root must be the parent of `data/`
- Reconstructed HDF5 files are produced by `Autoencoder_for_Recons_v3.ipynb`

## Sessions

16 train / 1 val / 1 test sessions, all from subject `0efbe614-9ae6-4131-9192-4398359b4f5f`.
