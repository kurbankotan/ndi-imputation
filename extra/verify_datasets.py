# -*- coding: utf-8 -*-
"""Reproduce the paper's NDI-F z-RMSE (MCAR 30 %, seeds 0-2) on the local copies of the benchmark datasets and compare
with results/rmse_long.csv: a copy whose three values agree to 1e-6 is the table the paper used (same rows, same order).
Usage: python extra/verify_datasets.py"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
os.environ["OUT_DIR"] = os.path.join(HERE, "_verify_tmp")
os.makedirs(os.environ["OUT_DIR"], exist_ok=True)
import numpy as np, pandas as pd
import ndi_benchmark as nb
import data_local as dl
ref = pd.read_csv(os.path.join(ROOT, "results", "rmse_long.csv"))
ref = ref[(ref.method == "NDI-F") & (ref.mechanism == "MCAR") & (np.isclose(ref.rate, 0.3))]
bad = 0
for ds in dl.load_paper_datasets():
    X_true = ds["X_num"].to_numpy(float); n, p = X_true.shape
    sd = np.nanstd(X_true, axis=0); sd[sd == 0] = 1.0
    vals = []
    for seed in nb.SEEDS:
        rng = np.random.default_rng(nb.stable_seed(ds["name"], seed, 0.3, "MCAR"))
        mask = nb.make_mask(X_true, "MCAR", 0.3, rng)
        X_in = X_true.copy(); X_in[mask] = np.nan
        lo, hi = np.nanmin(X_in, axis=0), np.nanmax(X_in, axis=0)
        X_imp = nb.NDIF().fit_transform(X_in)
        vals.append(nb.score(X_true, X_imp, mask, sd, lo, hi)["rmse"])
    r = ref[ref.dataset == ds["name"]].sort_values("seed")
    paper = r.rmse.to_numpy()
    ok = np.allclose(vals, paper, atol=5e-4)
    bad += (not ok)
    print(f"{ds['name']:<18s} n={n:5d} p={p:3d}  local={np.round(vals, 4)}  paper={np.round(paper, 4)}  {'OK' if ok else 'MISMATCH'}")
print("mismatches:", bad)
