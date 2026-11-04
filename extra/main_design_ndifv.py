#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NDI-F-V (k chosen by internal validation) in the main design of the paper: the same datasets, masks and seeds as
results/rmse_long.csv, plus the natural datasets, the downstream classification and the scaling experiment.
Writes results_main_ndifv/{rmse_long,downstream_long,scaling_long}_ndifv.csv (resumable), to be merged into the
shipped results as the 17th method."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results_main_ndifv"); os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("OUT_DIR", OUT)
import numpy as np, pandas as pd
import ndi_benchmark as nb
from ndi_benchmark import make_mask, score, stable_seed, append_row, done_keys, rkey, log, downstream
import data_local as dl
from ndi_benchmark import NDIFCV

METHOD = "NDI-F-V"
RMSE_CSV = os.path.join(OUT, "rmse_long_ndifv.csv"); DOWN_CSV = os.path.join(OUT, "downstream_long_ndifv.csv"); SCALE_CSV = os.path.join(OUT, "scaling_long_ndifv.csv")
fn = lambda X, s: NDIFCV(seed=s).fit_transform(X)


def run_main(datasets):
    key_cols = ["dataset", "mechanism", "rate", "seed", "method"]
    done_r = done_keys(RMSE_CSV, key_cols); done_d = done_keys(DOWN_CSV, key_cols)
    for ds in datasets:
        X_true = ds["X_num"].to_numpy(float); n, p = X_true.shape
        settings = [("MCAR", nb.NATURAL_EXTRA_MASK)] if ds["natural"] else nb.SETTINGS
        for mech, rate in settings:
            for seed in nb.SEEDS:
                rng = np.random.default_rng(stable_seed(ds["name"], seed, rate, mech))
                observed = ~np.isnan(X_true)
                mask = (observed & (rng.random((n, p)) < rate)) if ds["natural"] else make_mask(X_true, mech, rate, rng)
                X_in = X_true.copy(); X_in[mask] = np.nan
                sd = np.nanstd(X_true, axis=0); sd[sd == 0] = 1.0
                lo, hi = np.nanmin(X_in, axis=0), np.nanmax(X_in, axis=0)
                do_down = (mech, rate) in nb.DOWNSTREAM_SETTINGS and not ds["natural"]
                key = (ds["name"], mech, rkey(rate), str(seed), METHOD)
                if key in done_r and (not do_down or key in done_d): continue
                X_imp, dt, err = nb.run_method(fn, X_in, seed)
                sc = score(X_true, X_imp, mask, sd, lo, hi)
                if key not in done_r:
                    append_row(RMSE_CSV, dict(dataset=ds["name"], natural=ds["natural"], n=n, p=p, mechanism=mech, rate=rate, seed=seed, method=METHOD,
                                              rmse=sc["rmse"], mae=sc["mae"], oob=sc["oob"], nan_left=sc["nan_left"], time_s=dt, error=err))
                if do_down and key not in done_d and X_imp is not None:
                    for clf, (acc, f1) in downstream(X_imp, ds["X_cat"], ds["y"], seed).items():
                        append_row(DOWN_CSV, dict(dataset=ds["name"], natural=False, mechanism=mech, rate=rate, seed=seed, method=METHOD, classifier=clf, accuracy=acc, f1_macro=f1))
                log(f"{ds['name']:<18s} {mech:4s} {rate:.2f} seed={seed} rmse={sc['rmse']:.4f} {dt*1000:.0f} ms")
        if ds["natural"]:                                                     # downstream with the dataset's own missing values
            for seed in nb.SEEDS:
                X_nat = ds["X_num"].to_numpy(float); rate = round(float(np.isnan(X_nat).mean()), 4)
                key = (ds["name"], "natural", rkey(rate), str(seed), METHOD)
                if key in done_d: continue
                X_imp, dt, err = nb.run_method(fn, X_nat, seed)
                if X_imp is None or np.isnan(X_imp).any(): continue
                for clf, (acc, f1) in downstream(X_imp, ds["X_cat"], ds["y"], seed).items():
                    append_row(DOWN_CSV, dict(dataset=ds["name"], natural=True, mechanism="natural", rate=rate, seed=seed, method=METHOD, classifier=clf, accuracy=acc, f1_macro=f1))
            log(f"natural downstream {ds['name']} done")


def run_scaling():
    done = done_keys(SCALE_CSV, ["n", "p", "method"])
    for n in nb.SCALING_N:
        for p in nb.SCALING_P:
            if (str(n), str(p), METHOD) in done: continue
            rng = np.random.default_rng(1)
            F = rng.standard_normal((n, 3)); L = rng.standard_normal((3, p))
            X_true = F @ L + 0.5 * rng.standard_normal((n, p))
            mask = rng.random((n, p)) < 0.3
            X_in = X_true.copy(); X_in[mask] = np.nan; sd = X_true.std(0)
            X_imp, dt, err = nb.run_method(fn, X_in, 0)
            rmse = float(np.sqrt(np.mean(((X_imp[mask] - X_true[mask]) / sd[np.where(mask)[1]]) ** 2)))
            append_row(SCALE_CSV, dict(n=n, p=p, method=METHOD, time_s=dt, rmse=rmse, status=err or "ok"))
            log(f"scaling n={n:>7d} p={p:3d} {METHOD} {dt:8.2f} s rmse={rmse:.3f}")


if __name__ == "__main__":
    names = os.environ.get("PAPER_NAMES"); names = names.split(",") if names else dl.PAPER
    datasets = dl.load_paper_datasets(names)
    for loader in (nb.load_ckd, nb.load_heart, nb.load_mice):
        datasets.append(loader())
    run_main(datasets)
    run_scaling()
    log("=== NDI-F-V main-design rows finished ===")
