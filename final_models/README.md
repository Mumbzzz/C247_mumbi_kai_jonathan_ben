# Final Models

Notebooks for all final experiments.

## CNN & Ablation Studies (Ben)

| Notebook | Experiment | Best Val CER | Best Test CER |
|---|---|---|---|
| [baseline_cnn.ipynb](baseline_cnn.ipynb) | CNN — 16 ch/hand, 2000 Hz, log-spectrogram | 18.52% | 22.28% |
| [channel_ablation.ipynb](channel_ablation.ipynb) | CNN channel ablation (16 / 8 / 4 / 2 ch) | 18.52% (16 ch) | 22.28% |
| [temporal_downsampling_ablation.ipynb](temporal_downsampling_ablation.ipynb) | CNN temporal ablation (2000–125 Hz) | 18.52% (2000 Hz) | 22.28% |
| [training_ablation_and_cnnlstm.ipynb](training_ablation_and_cnnlstm.ipynb) | CNN — 16 ch/hand, 2000 Hz, log-spectrogram, training fraction ablation (25–100%) | 18.9% (100%) | 21.2% |
| [latent_ae_cnn.ipynb](latent_ae_cnn.ipynb) | CNN on AE latent vectors (1024-dim @ 62.5 Hz) | — | — |
| [reconstructed_emg_cnn.ipynb](reconstructed_emg_cnn.ipynb) | CNN on AE-reconstructed EMG (62.5 Hz) | 83.33% | 81.96% |
| [biophysics_pipeline_cnn.ipynb](biophysics_pipeline_cnn.ipynb) | CNN with biophysics preprocessing (Mel, 1000 Hz) | — | — |

---

## CNN w/BiLSTM (Mumbi)

Colab-based CNN+LSTM runs from `training_ablation_and_cnnlstm.ipynb` (GPU: NVIDIA L4).

> Requires `data/89335547/` (raw EMG, ~4.4 GB) and `data/89335547_recons_v3/` (AE-reconstructed EMG, ~125 MB), both downloaded inside the notebook via UCLA Box links.

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
| Latent AE (improved) | XX% | 99.9% | [RNN_recons.ipynb](RNN_recons.ipynb) | `archive/` | `archive/` |

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
| Latent AE (improved) | XXX% | 51.3% | [../data/Conformer_Iterations_Using_AE_Data.ipynb](../data/Conformer_Iterations_Using_AE_Data.ipynb) | `archive/` | `archive/` |
