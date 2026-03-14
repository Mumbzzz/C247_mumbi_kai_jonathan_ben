# Autoencoder_for_Recons_v3

Trains a vanilla autoencoder per session on raw EMG, saves latents and reconstructions.

## File Placement

Run from the `data/` directory. The notebook reads raw HDF5 files and writes outputs alongside them:

```
data/
├── Autoencoder_for_Recons_v3.ipynb           ← this notebook
└── 89335547/
    ├── <session_id>.hdf5                      ← raw input (one per session)
    ├── <session_id>_latent_v3.hdf5            ← written by training cell
    ├── <session_id>_ae_v3.pt                  ← model weights, written by training cell
    └── <session_id>_recons_v3.hdf5            ← written by batch reconstruct cell
```

## Dependencies

- `emg2qwerty/charset.py` — loaded via `sys.path.insert(0, '..')`, repo root must be parent of `data/`
- The reconstruction cells reference a hardcoded example session (`2021-07-22-1627004019-...`) — update `WEIGHTS_PATH`, `LATENT_FILE`, and `STEM` for a different session

## Output

`_recons_v3.hdf5` files are the input expected by `Conformer_Iterations_Using_AE_Data.ipynb`.
