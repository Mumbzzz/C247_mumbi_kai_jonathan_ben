# Final Models

Notebooks for all final experiments.

| Notebook | Experiment | Best Val CER | Best Test CER |
|---|---|---|---|
| [baseline_cnn.ipynb](baseline_cnn.ipynb) | CNN — 16 ch/hand, 2000 Hz, log-spectrogram | 18.52% | 22.28% |
| [channel_ablation.ipynb](channel_ablation.ipynb) | CNN channel ablation (16 / 8 / 4 / 2 ch) | 18.52% (16 ch) | 22.28% |
| [temporal_downsampling_ablation.ipynb](temporal_downsampling_ablation.ipynb) | CNN temporal ablation (2000–125 Hz) | 18.52% (2000 Hz) | 22.28% |
| [latent_ae_cnn.ipynb](latent_ae_cnn.ipynb) | CNN on AE latent vectors (1024-dim @ 62.5 Hz) | — | — |
| [reconstructed_emg_cnn.ipynb](reconstructed_emg_cnn.ipynb) | CNN on AE-reconstructed EMG (62.5 Hz) | 83.33% | 81.96% |
| [biophysics_pipeline_cnn.ipynb](biophysics_pipeline_cnn.ipynb) | CNN with biophysics preprocessing (Mel, 1000 Hz) | — | — |

---

## Mumbi Whidby — `Mumbi_C247A.ipynb`

All training was done in **Google Colab** (GPU: NVIDIA L4) using `Mumbi_C247A.ipynb`.

### Running the Notebook

The notebook clones the repo fresh to `/content/C247_mumbikaijonathanben` and works from there — its location on your Drive doesn't matter. It symlinks `results/` and `Playground_Mumbi/checkpoints/` to persistent Google Drive folders so outputs survive session resets.

### Data Folder Structure

```
data/
├── 89335547/            # Raw EMG sessions (18 HDF5 files, ~4.4 GB)
└── 89335547_recons_v3/  # AE-reconstructed EMG (18 HDF5 files, ~125 MB)
```

Both datasets are downloaded inside the notebook via UCLA Box links.

### Results

Full data in `results_summary_CNN_LSTM.csv` and `results_summary_CNN_training_fraction_ablation.csv` (this folder).

| Model | Input | Val CER | Test CER |
|-------|-------|---------|----------|
| CNN+LSTM | spectrogram | 15.8% | 19.0% |
| CNN+LSTM (biophys) | Mel spectrogram (8ch, 1kHz) | 17.9% | 21.4% |
| CNN+LSTM (recons v3) | AE-reconstructed EMG | 62.2% | 69.2% |

#### TDS CNN Training Fraction Ablation

| Train Fraction | Val CER | Test CER |
|----------------|---------|----------|
| 25% | 28.2% | 30.1% |
| 50% | 23.0% | 24.9% |
| 75% | 21.1% | 22.2% |
| 100% | 18.9% | 21.2% |
