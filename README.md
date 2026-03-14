# C147/247 Final Project
### Winter 2026 

## Project Group Members

| Name | Department | Email |
|------|-----------|-------|
| Mumbi Whidby | UCLA MAE | mwhidby@ucla.edu |
| He Kai Lim | UCLA MAE | limhekai@ucla.edu |
| Jonathan Gray | UCLA MAE | jonvgray@ucla.edu |
| Benjamin Forbes | UCLA MAE | benforbes@ucla.edu |

## Final Models & Instructions

All notebooks live in [`final_models/`](final_models/). The submission report is in [`final_report/`](final_report/).

---

### Baseline CNN — 16 ch/hand · 2000 Hz · Log-Spectrogram
- Val CER **18.52%** · Test CER **22.28%** · ~3 h 51 m training
- Model: TDSConvCTC (5.3 M params), standard `log_spectrogram` pipeline
- [baseline_cnn.ipynb](final_models/baseline_cnn.ipynb)

---

### Channel Ablation (CNN)
Sweeps 16 → 8 → 4 → 2 channels/hand using every-Nth-electrode selection (`ChannelSelect` transform):

| Channels/hand | Val CER | Test CER |
|---|---|---|
| 16 (baseline) | 18.52% | 22.28% |
| 8 | 18.65% | 23.30% |
| 4 | 24.88% | 27.12% |
| 2 | 40.85% | 45.00% |

- [channel_ablation.ipynb](final_models/channel_ablation.ipynb)

---

### Temporal Downsampling Ablation (CNN)
Sweeps 2000 → 1000 → 500 → 250 → 125 Hz via anti-aliased `TemporalDownsample` transform:

| Sample rate | Val CER | Test CER |
|---|---|---|
| 2000 Hz (baseline) | 18.52% | 22.28% |
| 1000 Hz | 52.53% | 38.38% |
| 500 Hz | 58.42% | 46.57% |
| 250 Hz | 79.15% | 76.12% |
| 125 Hz | 99.41% | 99.98% |

- [temporal_downsampling_ablation.ipynb](final_models/temporal_downsampling_ablation.ipynb)

---

### Training Fraction Ablation (CNN)
Sweeps training data from 25% → 100% using the same TDSConvCTC baseline (16 ch/hand, 2000 Hz, log-spectrogram):

| Train Fraction | Val CER | Test CER |
|---|---|---|
| 25% | 28.2% | 30.1% |
| 50% | 23.0% | 24.9% |
| 75% | 21.1% | 22.2% |
| 100% | 18.9% | 21.2% |

- [training_ablation_and_cnnlstm.ipynb](final_models/training_ablation_and_cnnlstm.ipynb)

---

### Latent AE CNN
Trains TDSConvCTC on pre-computed autoencoder latent vectors (1024-dim @ 62.5 Hz) from `emg_latent_ae_v2.hdf5`. Replaces the spectrogram front-end with a single linear projection.
- [latent_ae_cnn.ipynb](final_models/latent_ae_cnn.ipynb)

---

### Reconstructed EMG CNN
Trains TDSConvCTC on AE-decoded EMG signals (`*_recons_v3.hdf5`) at 62.5 Hz.
- Val CER **83.33%** · Test CER **81.96%** (150 epochs)
- [reconstructed_emg_cnn.ipynb](final_models/reconstructed_emg_cnn.ipynb)

---

### Biophysics Pipeline CNN
Trains TDSConvCTC with the full biophysics preprocessing pipeline (bandpass filter → 2× decimation → 32-bin Mel spectrogram) on 8 channels/wrist at 1000 Hz.
- [biophysics_pipeline_cnn.ipynb](final_models/biophysics_pipeline_cnn.ipynb)

---

### RNN w/BiLSTM
Self-contained notebooks for Kai + Jonathan's BiLSTM pipeline variants.

| Data condition | Val CER | Test CER | Notebook |
|---|---|---|---|
| Full raw data | 18.8% | 18.2% | [RNN_Raw.ipynb](final_models/RNN_Raw.ipynb) |
| Biophysics aug | 22.9% | 22.8% | [RNN_Biophys.ipynb](final_models/RNN_Biophys.ipynb) |
| Reconstructed EMG (unoptimized) | 78.5% | 104.0% | [RNN_Recons_Unoptimized.ipynb](final_models/RNN_Recons_Unoptimized.ipynb) |

- Results logs are written under `results/` during training.

---

### CNN-LSTM
Colab-based CNN+LSTM runs from Mumbi's notebook.

| Variant | Input | Val CER | Test CER |
|---|---|---|---|
| CNN+LSTM | spectrogram | 15.8% | 19.0% |
| CNN+LSTM (biophys) | Mel spectrogram (8ch, 1000 Hz) | 17.9% | 21.4% |
| CNN+LSTM (recons v3) | AE-reconstructed EMG | 62.2% | 69.2% |

- Notebook: [training_ablation_and_cnnlstm.ipynb](final_models/training_ablation_and_cnnlstm.ipynb)
- Results logs are written under `results/` during training.

---

### Conformer
Self-contained notebooks for Kai + Jonathan's Conformer pipeline variants.

| Data condition | Val CER | Test CER | Notebook |
|---|---|---|---|
| Full raw data | 12.5% | 14.1% | [Conformer_Raw.ipynb](final_models/Conformer_Raw.ipynb) |
| Biophysics aug | 14.29% | 15.9% | [Conformer_Biophys.ipynb](final_models/Conformer_Biophys.ipynb) |
| Reconstructed EMG (improved) | 48.13% | 51.3%  | [../data/Conformer_Iterations_Using_AE_Data.ipynb](../data/Conformer_Iterations_Using_AE_Data.ipynb) |

- Results logs are written under `results/` during training.

---

## Repo Notes:
- No models/ folder from emg2qwerty, because those are large files.
- `Playground_` folders are individual workspaces for each team member's experimentation and development.
- The dataset is quite a large folder (data/). I've added `*.hdf5` to `.gitignore` for now. I think its best to manually add them into your local workspace.
  - https://ucla.box.com/s/3xc4nwpfjfpo6ydjs94t0v2kuq37d5eg
