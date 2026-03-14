# emg_autoencoder_playground

Sandbox notebook for experimenting with the vanilla EMG autoencoder on a single session.

## File Placement

Run from the `data/` directory:

```
data/
├── emg_autoencoder_playground.ipynb          ← this notebook
├── emg_latent_vanilla_v5.hdf5                ← written by the notebook on first run
└── 89335547/
    └── 2021-07-22-1627004019-...-0efbe614-...9b4f5f.hdf5  ← raw input session
```

## Dependencies

- `emg2qwerty/charset.py` — loaded via `sys.path.insert(0, '..')`, repo root must be parent of `data/`
- `SESSION_HDF5` points to a single hardcoded session — update this path to use a different session
- Latent output is written to `emg_latent_vanilla_v5.hdf5` in `data/` (not inside `89335547/`)
