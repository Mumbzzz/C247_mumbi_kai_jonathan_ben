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

