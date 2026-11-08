# large/ — million-row and streaming experiments

`NDI_large_scale_colab.ipynb` runs everything on Google Colab Pro with the data and the results on Google Drive
(resumable). Without Colab:

```bash
DATA_DIR=/data/ndi_large OUT_DIR=results_large python large/get_large_data.py          # 4 GB of downloads once, 3 GB of .npy
EXP=ACC,ONLINE DATA_DIR=/data/ndi_large OUT_DIR=results_large python large/ndi_large.py
EXP=DEEP DEEP_N=50000 MIWAE_EPOCHS=200 GAIN_ITERS=50000 SINKHORN_ITERS=2000 REMASKER_EPOCHS=100 ... python large/ndi_large.py   # GPU
python large/make_large_tables.py results_large
QUICK=1 python large/ndi_large.py                                                        # smoke test on synthetic data
```

| File | Content |
|---|---|
| `ndi_large.py` | ACC (error and time against n: 10k, 50k, 100k, 1M, full rows; MCAR 30 %, MAR/MNAR 30 % up to 1M), DEEP (GPU imputers on 50k-row samples, same masks), ONLINE (streaming: frozen, updated and refitted statistics; `OnlineNDIF` accumulates the four correlation sufficient-statistic matrices), DOWN (HistGradientBoosting on 100k-row samples). |
| `large_data.py`, `get_large_data.py` | Loaders and downloads: MiniBooNE, YearPredictionMSD, covertype (quantitative columns), KDD Cup 99 (numeric columns), SUSY, HEPMASS, HIGGS, cached as float32 `.npy`. |
| `make_large_tables.py` | `summary_*.csv`, `large_tables.json`, figures L1–L3 (run by the notebook). |
| `make_large_paper.py` | `large_paper.json` and `figure10_large.png`: every number and the figure of Section 5.11 of the paper. |

Size limits per method (rows; `MAX_N_<METHOD>` overrides, with `-` written as `_`, e.g. `MAX_N_EM_FA`, `MAX_N_GaussCM`): EM-FA and EM-PPCA to convergence 2,000,000; GaussCM 1,000,000
(one solve per distinct missingness pattern); SoftImpute 2.5·10⁷ cells; MICE 20,000; MissForest and KNN 5,000. NDI-S, NDI-F,
NDI-F-V and EM-FA stopped after 1 or 3 iterations run on the full tables; their imputation is done in row blocks
(`ROW_BLOCK`, default 1,000,000) so that the full imputed matrix is never held in memory.

DEEP details: every GPU imputer runs in a fresh process (`DEEP_SUBPROCESS=1`), so no GPU memory is carried over between
methods; the budgets `GAIN_ITERS`, `MIWAE_EPOCHS`, `SINKHORN_ITERS`, `REMASKER_EPOCHS` label the result rows (`GAIN@50000`,
`MIWAE@500`, `Sinkhorn@2000`, `ReMasker@100`), rows with errors are moved to `deep_long_errors.csv` and recomputed, and rows
written by the first version without labels are relabelled with `LEGACY_BUDGETS` (default `GAIN:2000,MIWAE:200,Sinkhorn:200,ReMasker:100`).
