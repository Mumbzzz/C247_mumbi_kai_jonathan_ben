# Final Models

Notebooks for all final experiments.

## CNN & Ablation Studies (Ben)

| Notebook | Experiment | Best Val CER | Best Test CER |
|---|---|---|---|
| [baseline_cnn.ipynb](baseline_cnn.ipynb) | CNN — 16 ch/hand, 2000 Hz, log-spectrogram | 18.52% | 22.28% |
| [channel_ablation.ipynb](channel_ablation.ipynb) | CNN channel ablation (16 / 8 / 4 / 2 ch) | 18.52% (16 ch) | 22.28% |
| [temporal_downsampling_ablation.ipynb](temporal_downsampling_ablation.ipynb) | CNN temporal ablation (2000–125 Hz) | 18.52% (2000 Hz) | 22.28% |
| [training_fraction_ablation.ipynb](training_fraction_ablation.ipynb) | CNN training fraction ablation (25–100%) | 18.9% (100%) | 21.2% |
| [latent_ae_cnn.ipynb](latent_ae_cnn.ipynb) | CNN on AE latent vectors (1024-dim @ 62.5 Hz) | — | — |
| [reconstructed_emg_cnn.ipynb](reconstructed_emg_cnn.ipynb) | CNN on AE-reconstructed EMG (62.5 Hz) | 83.33% | 81.96% |
| [biophysics_pipeline_cnn.ipynb](biophysics_pipeline_cnn.ipynb) | CNN with biophysics preprocessing (Mel, 1000 Hz) | — | — |

---

## CNN w/BiLSTM (Mumbi)

Runs from [`cnn_lstm.ipynb`](cnn_lstm.ipynb) (GPU: NVIDIA L4). Works in Google Colab and locally.

> Requires `data/89335547/` (raw EMG, ~4.4 GB) and `data/89335547_recons_v3/` (AE-reconstructed EMG, ~125 MB), downloaded automatically in Colab or manually for local runs.

Results CSVs are in `results/`.

| Data Condition | Val CER | Test CER |
|---|---|---|
| Spectrogram | 15.8% | 19.0% |
| Mel spectrogram (biophys, 8ch, 1kHz) | 17.9% | 21.4% |
| AE-reconstructed EMG (recons v3) | 62.2% | 69.2% |

---

## RNN w/BiLSTM (Kai + Jonathan)

The original model files are in a somewhat convoluted pipeline from `Playground_Kai/train.py` with a series of custom arguments and flags.

These are self-contained `.ipynb` files that can be run standalone. They come packaged with any and all required functions from the `emg2qwerty` team.

Just ensure that the raw data is stored in `data`, and the Latent AE data is stored under `data_recons` at root.

| Data Condition | Val CER | Test CER | Model ipynb Link | Results Summary Location | Results CSV Location |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Full Raw Data | 18.8% | 18.2% | [RNN_Raw.ipynb](RNN_Raw.ipynb) | `results/kai_results_summary_RNN_fullraw.csv` | `results/kai_results_curves_RNN_fullraw.csv` |
| Biophys Aug | 22.9% | 22.8% | [RNN_Biophys.ipynb](RNN_Biophys.ipynb) | `results/kai_results_summary_RNN_preprocess.csv` | `results/kai_results_curves_RNN_preprocess.csv` |
| Latent AE (unoptimized) | 78.5% | 104.0% | [RNN_Recons_Unoptimized.ipynb](RNN_Recons_Unoptimized.ipynb) | `results/kai_results_summary_RNN_preprocess.csv` | `results/kai_results_curves_RNN_preprocess.csv` |
| Latent AE (improved) | 73.2% | 99.9% | [RNN_Recons.ipynb](RNN_Recons.ipynb) | `results/results_summary_RNN.csv` | `results/kai_results_curves_RNN.csv` |

---

## Conformer (Kai + Jonathan)

The original model files are in a somewhat convoluted pipeline from `Playground_Kai/train.py` with a series of custom arguments and flags.

These are self-contained `.ipynb` files that can be run standalone. They come packaged with any and all required functions from the `emg2qwerty` team.

Just ensure that the raw data is stored in `data`, and the Latent AE data is stored under `data_recons` at root.

| Data Condition | Val CER | Test CER | Model ipynb Link | Results Summary Location | Results CSV Location |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Full Raw Data | 12.5% | 14.1% | [Conformer_Raw.ipynb](Conformer_Raw.ipynb) | `results/kai_results_summary_CONFORMER_fullraw.csv` | `results/kai_results_curves_CONFORMER_fullraw.csv` |
| Biophys Aug | 14.29% | 15.9% | [Conformer_Biophys.ipynb](Conformer_Biophys.ipynb) | `results/kai_results_summary_CONFORMER_preprocess.csv` | `results/kai_results_curves_CONFORMER_preprocess.csv` |
| Latent AE (unoptimized) | 80.8% | 85.0% | [Conformer_Recons_Unoptimized.ipynb](Conformer_Recons_Unoptimized.ipynb) | `results/kai_results_summary_CONFORMER_recons.csv` | `results/kai_results_curves_CONFORMER_recons.csv` |
| Latent AE (improved) | 48.13% | 51.3% | [../data/Conformer_Iterations_Using_AE_Data.ipynb](../data/Conformer_Iterations_Using_AE_Data.ipynb) | `results/kai_results_summary_CONFORMER.csv` | `results/kai_results_curves_CONFORMER.csv` |

## Autoencoder Notebooks

| Notebook | Scope | Val Recon Loss | MSE (mean / rest / keystroke) | Notebook Link | Notes |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Autoencoder for Recons v3 | Per-session AE (18 sessions) | ~0.10 (after 80 epochs) | — | [Autoencoder_for_Recons_v3.ipynb](../data/Autoencoder_for_Recons_v3.ipynb) | Trains one AE per session; outputs `_recons_v3.hdf5` used by Conformer AE notebooks |
| Autoencoder Playground | Single-session AE (2021-07-22) | 2.07 (epoch 20 val recon) | 0.214 / 0.202 / 0.241 | [emg_autoencoder_playground.ipynb](../data/emg_autoencoder_playground.ipynb) | Spectral loss (λ=10), latent dim=256, 1.15M params; keystroke windows have higher MSE than rest |

