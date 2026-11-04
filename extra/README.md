# extra/ — the additional experiments of version 4.1

See the section "Version 4.1: additional experiments" of the main README. In short:

```bash
python extra/get_data.py                 # downloads the PMLB copies and the banknote file into extra/data/
python extra/verify_datasets.py          # checks which local copies reproduce the paper's tables exactly
EXP=EM,K,RATE,SMALLN,SCALEP,LARGE OUT_DIR=results_extra_rerun python extra/ndi_extra.py
python extra/make_extra_tables.py results_extra_rerun
python extra/make_extra_figures.py results_extra_rerun figures_rerun
python extra/main_design_ndifv.py        # NDI-F-V on the masks and seeds of the main design -> results_main_ndifv/
```

| File | Content |
|---|---|
| `ndi_extra.py` | The six experiments (EM, K, RATE, SMALLN, SCALEP, LARGE), resumable; imports the methods, masks and scoring of `ndi_benchmark.py`. |
| `data_local.py` | Loaders: the benchmark datasets (OpenML export, OpenML, or verified PMLB copies) and the nine larger PMLB datasets. |
| `get_data.py` | Downloads the local data files. |
| `verify_datasets.py` | Reproduces the paper's NDI-F values (MCAR 30 %) on the local copies. |
| `make_extra_tables.py`, `make_extra_figures.py` | Build `extra_tables.json` and Figures 7–9 from a results folder. |
| `main_design_ndifv.py` | NDI-F-V as an additional method of the main design (same datasets, masks and seeds). |
| `openml_csv/` | Optional: exact OpenML exports `<name>.csv` with a `__target__` column (see `data_local.py`). |
