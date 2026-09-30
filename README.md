# NDI-F: closed-form factor-model imputation for numerical tabular data

Code, data and complete results for

> Kırışoğlu, S., & Kotan, K. *Normal Deviate Imputation (NDI-F): A One-Pass Alternative to EM for Factor-Model
> Imputation of Numerical Tabular Data.* Manuscript under review.

## The method

NDI-F standardizes each column, factors the pairwise-complete correlation matrix once
(R ≈ ΛΛᵀ + Ψ; principal factors, k chosen by the Kaiser rule, at most 10) and imputes the missing
standardized values of each row by the conditional mean of this factor model given the row's observed values:

    f̂ = (I_k + Λ_Jᵀ Ψ_J⁻¹ Λ_J)⁻¹ Λ_Jᵀ Ψ_J⁻¹ z_J,     ẑ_K = Λ_K f̂,     x̂ = clip(μ + ẑ·σ, observed min, observed max)

This is the conditional mean of the classical factor-analysis model; NDI-F estimates the model's parameters in a
single pass instead of by EM. There is no training loop, no iteration, no randomness and no tuned hyperparameter.
The cost is O(n p k²) after one O(n p²) correlation pass, and new rows can be imputed with the stored μ, σ, Λ and Ψ.
`NDI("mean")`, `NDI("signed")` and `NDI("shrink")` give the simpler estimators NDI-0 (the original
normal-deviate rule), NDI-C and NDI-S, and `EMFA("fa")` fits the same factor model by maximum likelihood with EM,
starting from NDI-F (more accurate, about a hundred times slower).

## Quick start

```python
from ndi import NDIF

imp = NDIF()                               # Kaiser k, clipping to the observed range
X_train_imp = imp.fit_transform(X_train)   # numpy array, NaN = missing, numerical columns only
X_test_imp  = imp.transform(X_test)        # reuses the training statistics (μ, σ, Λ, Ψ)
```

`ndi.py` needs only numpy and pandas (scipy for `CopulaWrap`).

## Main results

All numbers below are read from `results/tables.json`.

- 15 complete OpenML datasets under MCAR (10, 30, 50 %), MAR (30 %) and MNAR (30 %) missingness with three seeds,
  plus three naturally incomplete UCI datasets.
- Among 16 methods over 75 dataset × setting blocks, NDI-F has average rank 6.21
  (5th). It beats mean substitution in 73 of 75 blocks, is not significantly different from MICE,
  SoftImpute and the Gaussian conditional mean (Nemenyi test), and only MissForest (rank 2.61) is
  significantly more accurate in that test.
- The same factor model fitted by maximum likelihood with EM (EM-FA, rank 3.59) is more accurate in 64 of
  75 blocks (pooled z-RMSE 0.794 against 0.830) at 72–87 times the cost (0.84 s against
  10 ms per dataset); NDI-F obtains 86 % of its gain over mean substitution in one pass.
- With MIWAE, ReMasker, Sinkhorn optimal-transport imputation and the AutoML imputer HyperImpute added
  (20 methods, 30 % settings, seed 0), NDI-F has average rank 9.06 (8th) and does not differ
  significantly from MIWAE, ReMasker or Sinkhorn imputation.
- Mean imputation time per dataset (n ≤ 2,000): NDI-F 10 ms, EM-FA 0.84 s, MissForest 4.4 s,
  MIWAE 11.5 s, HyperImpute 34 s, ReMasker 192 s, Sinkhorn imputation 361 s. NDI-F imputes a
  100,000 × 50 matrix in 1.8 s.

## Repository contents

| Path | Content |
|---|---|
| `ndi.py` | Standalone implementation: `NDIF` (NDI-F), `NDI` (NDI-0, NDI-C, NDI-S), `EMFA` (EM-FA and EM-PPCA), `GaussCM` (Gaussian conditional mean) and `CopulaWrap`. Same code as in the benchmark. |
| `ndi_benchmark.py` | The complete pipeline: data loading, missingness generation, all 20 imputers, downstream classification, runtime scaling, the warm-start experiment and the statistical tests. One run produces every number in the paper. |
| `NDI_benchmark_colab.ipynb` | Runs the pipeline on Google Colab (T4 GPU). |
| `make_tables.py`, `make_figures.py` | Build `tables.json` and the figures of the paper from a results folder. |
| `results/` | Raw and summary results of the run reported in the paper. |
| `figures/` | Figures 1–6 of the paper at 300 dpi. |
| `kidney_disease.csv`, `heart.csv`, `Data_Cortex_Nuclear.csv` | UCI Chronic Kidney Disease, Heart Disease and Mice Protein Expression files, read by the pipeline. |
| `requirements.txt` | Python packages. |

Files in `results/`:

| File | Content |
|---|---|
| `rmse_long.csv` | One row per dataset × mechanism × rate × seed × method: z-RMSE, z-MAE, share of imputed values outside the observed range, time, error message. |
| `downstream_long.csv` | Accuracy and macro-F1 of LR, KNN, RF and SVC (5-fold stratified CV) on the imputed data. |
| `scaling_long.csv`, `warmstart_long.csv` | Runtime scaling experiment; MissForest started from the column mean and from NDI-F. |
| `summary_*.csv`, `per_dataset_rmse_*.csv`, `natural_*.csv` | Summary tables. |
| `friedman_nemenyi_wilcoxon.txt`, `SUMMARY.md` | Statistical tests and a readable summary. |
| `tables.json` | All numbers used in the paper (built by `make_tables.py`). |
| `datasets.csv`, `env.json`, `run_log.txt` | Datasets used, package versions and hardware, log of the main run. |
| `method_info_long.csv` | EM iterations, number of factors and convergence of every EM-FA and EM-PPCA run. |
| `run_log_em.txt`, `env_resumed_*.json`, `timing_check.txt` | Log and environment of the second run that added EM-FA and EM-PPCA, and the speed of its machine relative to the main run (NDI-F on the 100,000 × 50 scaling matrix). |

## Reproducing the paper

Linux or Google Colab:

```bash
git clone https://github.com/kurbankotan/ndi-imputation.git
cd ndi-imputation
pip install -r requirements.txt

# optional: the official ReMasker code with two compatibility fixes (ReMasker is skipped without it)
git clone https://github.com/tydusky/remasker.git
sed -i 's/^from tkinter import E$/# from tkinter import E/' remasker/model_mae.py
sed -i 's/np\.float\b/float/g; s/np\.int\b/int/g; s/np\.bool\b/bool/g' remasker/remasker_impute.py remasker/model_mae.py remasker/utils.py remasker/plugin_mae.py

QUICK=1 python ndi_benchmark.py                             # smoke test, about 10 min -> results_quick/
PYTHONHASHSEED=3 DEEP_SEEDS=0 OUT_DIR=results_rerun python ndi_benchmark.py  # full run as in the paper, about 12.5 h -> results_rerun/
cd results_rerun && python ../make_tables.py && python ../make_figures.py
```

Notes:

- EM-FA and EM-PPCA were added to the shipped results by a second run of the same, resumable script
  (`PYTHONHASHSEED=3 DEEP_SEEDS=0 OUT_DIR=results python ndi_benchmark.py`, CPU only, about 20 minutes), which computes only
  the (dataset, setting, seed, method) combinations missing from `results/`. Its machine was
  {T['second_run_machine_ratio']:.1f} times slower than that of the main run on the reference computation in `timing_check.txt`.
- Write a new run to its own folder (`OUT_DIR=results_rerun`). The pipeline is resumable and skips every combination
  already present in the output folder, so a run into `results/` would only rebuild the summaries of the shipped results.
- The paper's run used 2 CPU cores and a Tesla T4 GPU, Python 3.13.15, numpy 2.1.3, pandas 2.2.3,
  scikit-learn 1.6.1, SciPy 1.16.3 and PyTorch 2.11.0.
- The 15 OpenML datasets are downloaded by their ids (`results/datasets.csv`); the three UCI files are read from this repository.
- The four learning-based baselines run only in the three 30 % settings and on the naturally incomplete datasets;
  `DEEP_SEEDS=0` uses the first seed, as in the paper. Methods whose packages are missing are skipped.
- `pandas<3` is required because the hyperimpute package silently returns the column mean under pandas 3 (the pipeline
  checks this and disables HyperImpute if the check fails); scikit-learn must be older than 1.8 for hyperimpute 0.1.17.
- With the same package versions, the masks and the results of the seeded CPU methods reproduce exactly; the
  GPU-trained baselines (GAIN, MIWAE, Sinkhorn imputation, ReMasker) can differ slightly, and timings depend on the hardware.
- HyperImpute needs two more settings. For columns with fewer than 500 observed values, hyperimpute 0.1.17 takes its
  candidate learners from a Python set, so their order depends on the interpreter's hash seed; on ecoli (MCAR 30 %)
  this changes its z-RMSE from 2.267 (logistic regression first, as in the paper's run) to 0.882 (random forest first).
  `PYTHONHASHSEED=3` gives the order of the paper's run. Its results also depend on the xgboost version: the reported
  yeast values are reproduced with xgboost 3.4.1 (Python >= 3.12, pinned in `requirements.txt`) but not with 3.2.0.
  With both settings and Python 3.13, our re-runs reproduced the HyperImpute values of wine, ecoli, yeast, CKD, Heart
  and Mice exactly and those of ionosphere to within 0.005.

## License

MIT, see `LICENSE`.
