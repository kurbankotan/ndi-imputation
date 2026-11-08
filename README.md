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
normal-deviate rule), NDI-C and NDI-S; `NDIFCV()` is NDI-F-V, in which k is chosen by internal validation instead of the
Kaiser rule (version 4.1; about ten times the cost of NDI-F, still one pass); and `EMFA("fa")` fits the same factor model by
maximum likelihood with EM, starting from NDI-F (more accurate, about a hundred times slower; `EMFA("fa", max_iter=3, tol=0.0)`
stops after three iterations, which recover about 80 % of EM's gain at four times the cost of NDI-F).

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
| `figures/` | Figures 1–10 of the paper (600 dpi). |
| `extra/`, `results_extra/` | Version 4.1: the additional experiments (i)–(vi) of Section 4.7 of the paper (scripts, data loaders, raw results, `extra_tables.json`); see below. |
| `large/`, `results_large/` | Version 4.1: the million-row and streaming experiments (vii)–(viii) (scripts, Colab notebook, results, `large_paper.json`); see below. |
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
  1.2 times slower than that of the main run on the reference computation in `timing_check.txt`.
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

## Version 4.1: additional experiments (`extra/`, `results_extra/`)

Six further experiments, reported in Section 4.7 and Sections 5.2, 5.3, 5.7, 5.8 and 5.10 of the paper, and one new
variant, NDI-F-V (`NDIFCV` in `ndi.py`: the number of factors chosen by hiding 10 % of the observed cells of at most
5,000 rows and testing 12 candidate values on them; deterministic, no iteration). All numbers below are read from
`results_extra/extra_tables.json`.

- **EM stopped early.** EM-FA started from NDI-F and stopped after m iterations (8 benchmark datasets,
  40 dataset × setting blocks): one iteration gives 56 % of the converged gain
  (median block) at 2.5 times the time of NDI-F, three give 80 % at
  4.2 times, ten give 95 % at 10 times; convergence
  (median 110 iterations, 132 times) adds nothing further. Three starts converge to the same fit.
- **Number of factors.** No fixed k is best for every dataset; the validated k of NDI-F-V (pooled z-RMSE 0.887 against
  0.901 for the Kaiser rule) is better than NDI-F in 27 of 40 blocks (Wilcoxon p = 0.002) at
  9.6 times its cost. The eigenvalue floor matters little.
- **70 and 80 % MCAR.** The iterative methods (KNN, MICE, MissForest, EM) fall back to the mean level; the closed-form
  estimators NDI-S, NDI-F-V and GaussCM remain the most accurate.
- **Small samples** (seven datasets with p ≥ 33 sub-sampled to 50–500 rows): SoftImpute is the most accurate, then
  MissForest and NDI-F-V; NDI-F beats GaussCM at n ≤ 200; EM-FA is unstable at n = 50 (worse than the mean on
  3 of seven datasets).
- **Scaling with p** (synthetic three-factor data, 30 % MCAR): NDI-F imputes 100,000 × 500 in 6.5 s,
  NDI-F-V in 13.9 s, EM-FA in 15.7 s and GaussCM in 642 s
  (its per-pattern solves; 1.7 s with 20 patterns at p = 200), at equal accuracy.
- **Nine larger PMLB datasets** (2,000–70,000 rows, 10–719 columns; MCAR, MAR, MNAR 30 %, seed 0): under MCAR GaussCM is more
  accurate than NDI-F with k ≤ 10 on every dataset, because the Kaiser cap limits NDI-F on wide data; NDI-F-V closes the
  gap (14 wins, 11 losses against GaussCM over 25 blocks) at a fraction of its cost
  (mnist, 70,000 × 719: NDI-F 8 s, NDI-F-V 36 s, EM-FA 43 s, GaussCM 996 s).
  Under MAR and MNAR on strongly correlated data, MICE and EM, which re-estimate the column statistics, are far more
  accurate than every one-pass estimator.

Implementation change in 4.1: the pairwise-complete correlation matrix is computed with four matrix products
(`pairwise_corr`) instead of `pandas.DataFrame.corr()`; the result is identical up to rounding, and NDI-F at
100,000 × 500 drops from 61 s to 6.5 s. The shipped `results/` (main design) were produced with the pandas call;
the chunk size of the batched solves is now bounded so that wide matrices (mnist) fit in memory.

Running the additional experiments:

```bash
python extra/get_data.py                 # PMLB copies and the banknote file -> extra/data/ (about 20 MB)
python extra/verify_datasets.py          # which local copies reproduce the paper's tables exactly
EXP=EM,K,RATE,SMALLN,SCALEP,LARGE OUT_DIR=results_extra_rerun python extra/ndi_extra.py
python extra/make_extra_tables.py results_extra_rerun && python extra/make_extra_figures.py results_extra_rerun figures_rerun
```

The experiments that use the benchmark datasets (EM, K, RATE) take the datasets from OpenML as the main benchmark does;
without network access, or to run exactly what the shipped results used, export the OpenML tables to
`extra/openml_csv/<name>.csv` (instructions in `extra/data_local.py`) or rely on the PMLB copies, which reproduce the
paper's tables exactly for wine, ecoli, ionosphere, yeast, wine-quality-red, spambase and MagicTelescope
(`results_extra/paper_datasets_used.csv` lists the datasets of the shipped run). `extra/main_design_ndifv.py` adds NDI-F-V
to the main design (same masks and seeds as `results/rmse_long.csv`). Everything is resumable.

Files in `results_extra/`: `em_iterations_long.csv`, `k_sensitivity_long.csv`, `high_rate_long.csv`, `small_n_long.csv`,
`scaling_p_long.csv`, `large_long.csv` (one row per run), `large_datasets.csv`, `paper_datasets_used.csv`,
`extra_tables.json` (all numbers used in Section 4.7 and the related results sections), `env_extra.json`,
`run_log_extra.txt`, `timing_check_extra.txt` and `timing_paths.json` (speed of the machine of this run relative to the
main run, with both correlation routines).

## Million-row experiments (`large/`, `results_large/`)

Experiments (vii)–(viii) of Section 4.7 of the paper (Section 5.11, Tables 18–19, A6–A7, Fig. 10): seven public tables with
129,593 to 11,000,000 rows (MiniBooNE, YearPredictionMSD, Covertype, KDD Cup 1999, SUSY, HEPMASS, HIGGS), run on Google Colab
(8 vCPUs, Tesla T4) with `large/NDI_large_scale_colab.ipynb`. All numbers below are read from `results_large/large_paper.json`.

- **Cost.** On the full HIGGS table (11 M × 28, 30 % MCAR) NDI-F took 93 s, NDI-F-V 97 s,
  EM-FA after three iterations 626 s and the Gaussian conditional mean 719 s; EM-FA to convergence needed
  18–50 minutes already at 10⁶ rows.
- **Accuracy.** With millions of rows the full correlation matrix is estimated precisely, and the Gaussian conditional mean
  (MCAR, MAR) and EM-FA (MNAR) were more accurate than NDI-F (average ranks over 84 blocks: EM-FA 2.33, GaussCM
  2.80, EM-FA@3 3.02, NDI-F-V 4.14, NDI-F 5.75); NDI-F-V beat NDI-F in 65 of 84 blocks.
- **Learning-based imputers (50,000-row samples).** ReMasker was the most accurate method (rank 2.14 of 12) at
  1,566 s per block on the GPU; GAIN and MIWAE were unstable at this size; Sinkhorn imputation stayed behind NDI-F.
- **Streaming.** The statistics of NDI-F are sums over rows and can be updated exactly block by block (identical to a refit,
  0.08 s per 100,000-row update). On a non-stationary stream (KDD Cup 99) frozen statistics failed
  (1.618, worse than frozen mean substitution, 0.994); updated ones reached 0.485.

Files: `results_large/acc_long.csv`, `deep_long.csv` (+ `deep_long_errors.csv`: the first-pass MIWAE runs that ran out of GPU
memory), `online_long.csv`, `downstream_long.csv` (one row per run), the summaries and logs of the Colab runs, and
`large_paper.json` (all numbers of the paper's Section 5.11, written by `python large/make_large_paper.py results_large`).
See `large/README.md` to re-run.

## License

MIT, see `LICENSE`.
