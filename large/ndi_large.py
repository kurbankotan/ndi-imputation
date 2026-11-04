#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ndi_large.py — million-row experiments for NDI-F (resumable; results in OUT_DIR, default results_large/).

Datasets (cached as float32 .npy in DATA_DIR by get_large_data.py):
    MiniBooNE 130,064 x 50 | YearPredictionMSD 515,345 x 90 | covertype 581,012 x 10 (the quantitative columns) |
    KDDCup99 4,898,431 x ~34 (numeric, non-constant) | SUSY 5,000,000 x 18 | HEPMASS 10,500,000 x 28 | HIGGS 11,000,000 x 28

Experiments (EXP=ACC,DEEP,ONLINE,DOWN; default ACC,ONLINE):
    ACC     accuracy and wall-clock time against the number of rows: random row samples of n = 10k, 50k, 100k, 1M and the
            full table, 30 % MCAR (MAR and MNAR 30 % up to 1M rows), three seeds up to 100k rows and one seed above.
            Methods: Mean, NDI-S, NDI-F, NDI-F-V, EM-FA after 1 and 3 iterations, EM-FA and EM-PPCA to convergence,
            GaussCM, SoftImpute, MICE, MissForest, KNN — each up to the size given in MAX_N below.
    DEEP    the GPU methods (GAIN, MIWAE, Sinkhorn, ReMasker) and HyperImpute on the n = DEEP_N (50,000) samples with
            the seed-0 masks of ACC, so that they are compared with the cheap methods on identical cells.
    ONLINE  streaming: the statistics are estimated on the first block of rows and the following blocks are imputed
            (a) with frozen statistics, (b) with statistics updated after every block (the four sufficient-statistic
            matrices of the pairwise correlation are additive), (c) by refitting on all rows seen so far, and (d) with
            a frozen EM-FA model; error and time per block.
    DOWN    downstream classification on the 100k samples (HistGradientBoosting, 5-fold CV) for the cheap methods.

Usage (Google Colab, see NDI_large_scale_colab.ipynb):
    DATA_DIR=/content/drive/MyDrive/ndi_large/data OUT_DIR=/content/drive/MyDrive/ndi_large/results python large/ndi_large.py
    DATASETS=MiniBooNE,SUSY EXP=ACC python large/ndi_large.py          (subsets; everything is resumable)
    QUICK=1 python large/ndi_large.py                                   (synthetic data, small sizes: smoke test)
"""
import os, sys, time, json, platform
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
QUICK = os.environ.get("QUICK", "0") == "1"
OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "results_large_quick" if QUICK else "results_large")); os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("OUT_DIR", OUT)
import numpy as np, pandas as pd
import ndi_benchmark as nb
from ndi_benchmark import NDIF, NDIFCV, NDI, GaussCM, EMFA, pairwise_corr, make_mask, stable_seed, append_row, done_keys, rkey, log
import large_data as ld

EXP = os.environ.get("EXP", "ACC,ONLINE").split(",")
DATASETS = os.environ.get("DATASETS", "synthetic" if QUICK else ",".join(ld.ORDER)).split(",")
N_GRID = [10_000, 50_000, 100_000, 1_000_000, "full"]
MECHS_ALL = [("MCAR", .30)]; MECHS_SMALL = [("MCAR", .30), ("MAR", .30), ("MNAR", .30)]
MECH_MAX_N = 1_000_000                     # MAR / MNAR masks up to this many rows
SEEDS_SMALL, SEEDS_LARGE, SEED_SPLIT = [0, 1, 2], [0], 100_000
DEEP_N = int(os.environ.get("DEEP_N", "50000"))
BLOCK = int(os.environ.get("ROW_BLOCK", "1000000"))        # rows per block for the row-separable imputations
ONLINE_BLOCK = int(os.environ.get("ONLINE_BLOCK", "100000"))
if QUICK:
    N_GRID = [2_000, 5_000, "full"]; MECH_MAX_N = 5_000; SEED_SPLIT = 2_000; DEEP_N = 2_000; ONLINE_BLOCK = 2_000; BLOCK = 3_000

# the largest number of rows each method is run on (env MAX_N_<METHOD> overrides, e.g. MAX_N_EM-FA=11000000)
MAX_N = {"Mean": None, "NDI-S": None, "NDI-F": None, "NDI-F-V": None, "EM-FA@1": None, "EM-FA@3": None,
         "EM-FA": 2_000_000, "EM-PPCA": 2_000_000, "GaussCM": 1_000_000, "SoftImpute": 1_000_000, "MICE": 20_000,
         "MissForest": 5_000, "KNN": 5_000}
for m in list(MAX_N):
    v = os.environ.get("MAX_N_" + m.replace("@", "_"))
    if v: MAX_N[m] = int(float(v))
MAX_CELLS = {"SoftImpute": 2.5e7}
ROWSEP = {"NDI-F": lambda s: NDIF(), "NDI-F-V": lambda s: NDIFCV(seed=s), "NDI-S": lambda s: NDI("shrink"),
          "EM-FA@1": lambda s: EMFA("fa", max_iter=1, tol=0.0), "EM-FA@3": lambda s: EMFA("fa", max_iter=3, tol=0.0),
          "EM-FA": lambda s: EMFA("fa"), "EM-PPCA": lambda s: EMFA("ppca")}        # fitted once, imputed block by block
FULL = {"Mean": nb.m_mean, "GaussCM": lambda X, s: GaussCM().fit_transform(X), "SoftImpute": nb.m_softimpute,
        "MICE": nb.m_mice, "MissForest": nb.m_missforest, "KNN": nb.m_knn}
ACC_METHODS = ["Mean", "NDI-S", "NDI-F", "NDI-F-V", "EM-FA@1", "EM-FA@3", "EM-FA", "EM-PPCA", "GaussCM", "SoftImpute", "MICE", "MissForest", "KNN"]
DEEP_METHODS = [m for m in os.environ.get("DEEP_METHODS", "GAIN,MIWAE,Sinkhorn,ReMasker,HyperImpute").split(",") if m in nb.METHODS]
CSV = {k: os.path.join(OUT, f) for k, f in dict(ACC="acc_long.csv", DEEP="deep_long.csv", ONLINE="online_long.csv", DOWN="downstream_long.csv").items()}


def timed(fn, *a):
    t0 = time.perf_counter(); out = fn(*a); return out, time.perf_counter() - t0


def masked(X_true, name, mech, rate, seed):
    """the mask of (dataset, n, mechanism, rate, seed) is reproduced from stable_seed(), as in the paper"""
    rng = np.random.default_rng(stable_seed(f"{name}|{X_true.shape[0]}", seed, rate, mech))
    mask = make_mask(X_true, mech, rate, rng)
    X_in = X_true.copy(); X_in[mask] = np.nan
    return X_in, mask


class Scorer:
    """z-RMSE, z-MAE and out-of-range share accumulated block by block (no full imputed matrix needed)"""
    def __init__(self, X_true, mask, sd, lo, hi):
        self.X_true, self.mask, self.sd, self.lo, self.hi = X_true, mask, sd, lo, hi
        self.se = self.ae = self.oob = 0.0; self.n = 0; self.nan_left = 0

    def add(self, X_imp, s=0):
        m = self.mask[s:s + len(X_imp)]
        if np.isnan(X_imp[m]).any():
            self.nan_left += int(np.isnan(X_imp[m]).sum()); return
        cols = np.where(m)[1]
        d = (X_imp[m] - self.X_true[s:s + len(X_imp)][m]) / self.sd[cols]
        self.se += float((d ** 2).sum()); self.ae += float(np.abs(d).sum()); self.n += len(d)
        self.oob += float(((X_imp[m] < self.lo[cols]) | (X_imp[m] > self.hi[cols])).sum())

    def result(self):
        if self.nan_left or self.n == 0: return dict(rmse=np.nan, mae=np.nan, oob=np.nan, nan_left=self.nan_left)
        return dict(rmse=float(np.sqrt(self.se / self.n)), mae=self.ae / self.n, oob=self.oob / self.n, nan_left=0)


def run_rowsep(factory, X_in, seed, scorer):
    """fit on all rows, impute in row blocks (NDI-F, NDI-F-V, NDI-S, EM-*) -> (scores, time, error, info)"""
    t0 = time.perf_counter(); info = {}
    try:
        model = factory(seed).fit(X_in)
        for s in range(0, len(X_in), BLOCK):
            scorer.add(model.transform(X_in[s:s + BLOCK]), s)
        for a in ("k_", "k_cv_", "n_iter_", "converged_"):
            if hasattr(model, a): info[a.rstrip("_")] = getattr(model, a)
        err = ""
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:80]}"
    return scorer.result(), time.perf_counter() - t0, err, info


def run_full(fn, X_in, seed, scorer):
    t0 = time.perf_counter()
    try:
        scorer.add(np.asarray(fn(X_in, seed), float)); err = ""
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:80]}"
    return scorer.result(), time.perf_counter() - t0, err, {}


def sample_rows(X, n, seed):
    if n == "full" or n >= len(X): return X, len(X)
    rng = np.random.default_rng(seed)
    return X[np.sort(rng.choice(len(X), n, replace=False))], n


def col_stats(X):
    sd = X.std(0); sd[sd == 0] = 1.0
    return sd, X.min(0), X.max(0)


def run_acc(name, X_full):
    done = done_keys(CSV["ACC"], ["dataset", "n", "mechanism", "rate", "seed", "method"])
    for n in N_GRID:
        if n != "full" and n >= len(X_full): continue
        mechs = MECHS_SMALL if (n != "full" and n <= MECH_MAX_N) or len(X_full) <= MECH_MAX_N else MECHS_ALL
        seeds = SEEDS_SMALL if (n != "full" and n <= SEED_SPLIT) else SEEDS_LARGE
        for seed in seeds:
            X_true, n_rows = sample_rows(X_full, n, seed); X_true = np.asarray(X_true, float)
            sd, lo, hi = col_stats(X_true)
            for mech, rate in mechs:
                todo = [m for m in ACC_METHODS if (name, str(n_rows), mech, rkey(rate), str(seed), m) not in done
                        and (MAX_N[m] is None or n_rows <= MAX_N[m]) and n_rows * X_true.shape[1] <= MAX_CELLS.get(m, np.inf)]
                if not todo: continue
                X_in, mask = masked(X_true, name, mech, rate, seed)
                lo_in, hi_in = np.nanmin(X_in, 0), np.nanmax(X_in, 0)
                for m in todo:
                    sc = Scorer(X_true, mask, sd, lo_in, hi_in)
                    res, dt, err, info = run_rowsep(ROWSEP[m], X_in, seed, sc) if m in ROWSEP else run_full(FULL[m], X_in, seed, sc)
                    append_row(CSV["ACC"], dict(dataset=name, n=n_rows, p=X_true.shape[1], mechanism=mech, rate=rate, seed=seed, method=m,
                                                rmse=res["rmse"], mae=res["mae"], oob=res["oob"], nan_left=res["nan_left"], time_s=dt, error=err,
                                                k=info.get("k_cv", info.get("k")), em_iterations=info.get("n_iter")))
                    log(f"ACC {name:<18s} n={n_rows:>9,d} {mech:4s} seed={seed} {m:<10s} rmse={res['rmse']:.4f} {dt:9.2f} s {err}")
                del X_in, mask


def run_deep(name, X_full):
    if not DEEP_METHODS: log("DEEP: no GPU method available (torch / hyperimpute / remasker missing)"); return
    done = done_keys(CSV["DEEP"], ["dataset", "n", "mechanism", "rate", "seed", "method"])
    n = DEEP_N if DEEP_N < len(X_full) else "full"
    X_true, n_rows = sample_rows(X_full, n, 0); X_true = np.asarray(X_true, float)
    sd, lo, hi = col_stats(X_true)
    for mech, rate in MECHS_SMALL:
        todo = [m for m in DEEP_METHODS if (name, str(n_rows), mech, rkey(rate), "0", m) not in done]
        if not todo: continue
        X_in, mask = masked(X_true, name, mech, rate, 0)
        lo_in, hi_in = np.nanmin(X_in, 0), np.nanmax(X_in, 0)
        for m in todo:
            sc = Scorer(X_true, mask, sd, lo_in, hi_in)
            res, dt, err, _ = run_full(nb.METHODS[m], X_in, 0, sc)
            append_row(CSV["DEEP"], dict(dataset=name, n=n_rows, p=X_true.shape[1], mechanism=mech, rate=rate, seed=0, method=m,
                                         rmse=res["rmse"], mae=res["mae"], oob=res["oob"], nan_left=res["nan_left"], time_s=dt, error=err))
            log(f"DEEP {name:<18s} n={n_rows:>9,d} {mech:4s} {m:<12s} rmse={res['rmse']:.4f} {dt:9.1f} s {err}")


class OnlineNDIF(NDIF):
    """NDI-F whose statistics are updated with every new block: the column sums, sums of squares and the four
    pairwise-correlation matrices (counts, sums, sums of squares, cross-products) are additive, so updating costs one
    pass over the new block and one p x p eigendecomposition.  Imputation of a block uses the statistics of the
    blocks seen before it (prequential evaluation)."""
    def __init__(self, **kw):
        super().__init__(**kw); self.N = None

    def update(self, X):
        X = np.asarray(X, float); O = (~np.isnan(X)).astype(float); X0 = np.where(O > 0, X, 0.0)
        if self.N is None:
            p = X.shape[1]; self.N = np.zeros((p, p)); self.Sx = np.zeros((p, p)); self.Sxx = np.zeros((p, p)); self.Sxy = np.zeros((p, p))
            self.cn = np.zeros(p); self.cs = np.zeros(p); self.css = np.zeros(p); self.lo_ = np.full(p, np.inf); self.hi_ = np.full(p, -np.inf)
        self.N += O.T @ O; self.Sx += X0.T @ O; self.Sxx += (X0 * X0).T @ O; self.Sxy += X0.T @ X0
        self.cn += O.sum(0); self.cs += X0.sum(0); self.css += (X0 * X0).sum(0)
        self.lo_ = np.minimum(self.lo_, np.nanmin(X, 0)); self.hi_ = np.maximum(self.hi_, np.nanmax(X, 0))
        self._refit(); return self

    def _refit(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            self.mu_ = self.cs / self.cn; self.sigma_ = np.sqrt(np.maximum(self.css / self.cn - self.mu_ ** 2, 0))
            self.sigma_[~(self.sigma_ > 0)] = 1.0
            mx = self.Sx / self.N; cov = self.Sxy / self.N - mx * mx.T; vx = self.Sxx / self.N - mx ** 2
            R = cov / np.sqrt(vx * vx.T)
        R[~np.isfinite(R)] = 0.0; np.fill_diagonal(R, 1.0); R = np.clip(R, -1, 1)
        w, V = np.linalg.eigh(R); w = np.clip(w, self.ridge, None); order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
        k = int((w > 1.0).sum()) if self.k == "kaiser" else int(self.k)
        self.k_ = max(1, min(k, self.max_k, R.shape[0] - 1))
        self.L_ = V[:, :self.k_] * np.sqrt(w[:self.k_]); self.psi_ = np.clip(1.0 - (self.L_ ** 2).sum(1), self.ridge, None)


def run_online(name, X_full):
    """prequential: block b is imputed with statistics from blocks < b; the first block only initializes"""
    done = done_keys(CSV["ONLINE"], ["dataset", "block", "variant"])
    X_true = np.asarray(X_full, float); n, p = X_true.shape
    X_in, mask = masked(X_true, name + "|online", "MCAR", .30, 0)
    sd, lo, hi = col_stats(X_true)
    nb_blocks = int(np.ceil(n / ONLINE_BLOCK))
    variants = ["NDI-F frozen", "NDI-F updated", "NDI-F refit", "NDI-F-V frozen", "EM-FA frozen", "Mean frozen"]
    if all((name, str(b), v) in done for b in range(1, nb_blocks) for v in variants): return
    first = X_in[:ONLINE_BLOCK]
    frozen = NDIF().fit(first); frozen_v = NDIFCV(seed=0).fit(first); frozen_em = EMFA("fa").fit(first)
    mu0 = np.nanmean(first, 0)
    online = OnlineNDIF().update(first)
    t_upd = 0.0
    for b in range(1, nb_blocks):
        s, e = b * ONLINE_BLOCK, min(n, (b + 1) * ONLINE_BLOCK); Xb = X_in[s:e]
        def rec(variant, X_imp, dt, extra=None):
            if (name, str(b), variant) in done: return
            sc = Scorer(X_true, mask, sd, np.nanmin(X_in, 0), np.nanmax(X_in, 0)); sc.add(X_imp, s)
            append_row(CSV["ONLINE"], dict(dataset=name, block=b, rows_seen=s, block_rows=e - s, variant=variant, rmse=sc.result()["rmse"], time_s=dt, k=(extra or {}).get("k", np.nan)))
        X_imp, dt = timed(frozen.transform, Xb); rec("NDI-F frozen", X_imp, dt)
        X_imp, dt = timed(frozen_v.transform, Xb); rec("NDI-F-V frozen", X_imp, dt)
        X_imp, dt = timed(frozen_em.transform, Xb); rec("EM-FA frozen", X_imp, dt)
        rec("Mean frozen", np.where(np.isnan(Xb), mu0, Xb), 0.0)
        X_imp, dt = timed(online.transform, Xb); rec("NDI-F updated", X_imp, dt, dict(k=online.k_))
        _, du = timed(online.update, Xb); t_upd += du                                   # update after imputing (prequential)
        if os.environ.get("ONLINE_REFIT", "1") == "1":                                 # O(rows seen) per block; ONLINE_REFIT=0 skips it
            def refit():
                m = NDIF().fit(X_in[:s]); return m.transform(Xb)
            X_imp, dt = timed(refit); rec("NDI-F refit", X_imp, dt)
        log(f"ONLINE {name:<18s} block {b:3d}/{nb_blocks - 1} rows_seen={s:>9,d} update {du:.2f} s")
    log(f"ONLINE {name}: total update time {t_upd:.1f} s")


def run_down(name, X_full, y_full):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    if y_full is None: return
    done = done_keys(CSV["DOWN"], ["dataset", "n", "mechanism", "rate", "seed", "method"])
    n = 100_000 if 100_000 < len(X_full) else "full"
    rng = np.random.default_rng(0); idx = np.sort(rng.choice(len(X_full), n, replace=False)) if n != "full" else np.arange(len(X_full))
    X_true, y = np.asarray(X_full[idx], float), y_full[idx]
    if QUICK: X_true, y = X_true[:3000], y[:3000]
    classes, counts = np.unique(y, return_counts=True)
    keep = np.isin(y, classes[counts >= 10]); X_true, y = X_true[keep], y[keep]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    methods = ["complete", "Mean", "NDI-S", "NDI-F", "NDI-F-V", "EM-FA@3", "EM-FA", "GaussCM", "SoftImpute"]
    for mech, rate in MECHS_SMALL:
        X_in, mask = masked(X_true, name, mech, rate, 0)
        for m in methods:
            if (name, str(len(X_true)), mech, rkey(rate), "0", m) in done: continue
            if m == "complete": X_imp = X_true
            else:
                sc = Scorer(X_true, mask, *col_stats(X_true))
                if m in ROWSEP:
                    model = ROWSEP[m](0).fit(X_in); X_imp = np.vstack([model.transform(X_in[s:s + BLOCK]) for s in range(0, len(X_in), BLOCK)])
                else: X_imp = np.asarray(FULL[m](X_in, 0), float)
            acc = cross_val_score(HistGradientBoostingClassifier(random_state=0), X_imp, y, cv=cv, scoring="accuracy").mean()
            append_row(CSV["DOWN"], dict(dataset=name, n=len(X_true), mechanism=mech, rate=rate, seed=0, method=m, accuracy=float(acc)))
            log(f"DOWN {name:<18s} {mech:4s} {m:<10s} acc={acc:.4f}")


def main():
    json.dump(dict(python=platform.python_version(), platform=platform.platform(), cpu_count=os.cpu_count(), numpy=np.__version__,
                   pandas=pd.__version__, sklearn=__import__("sklearn").__version__, experiments=EXP, datasets=DATASETS, n_grid=[str(x) for x in N_GRID],
                   max_n={k: v for k, v in MAX_N.items()}, deep_n=DEEP_N, gpu=ld.gpu_name(), time=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(os.path.join(OUT, "env_large.json"), "w"), indent=1)
    for name in DATASETS:
        X, y = ld.load(name)
        log(f"=== {name}: {X.shape[0]:,} rows x {X.shape[1]} columns ({X.nbytes / 1e9:.2f} GB as stored)")
        append_row(os.path.join(OUT, "datasets_large.csv"), dict(dataset=name, n=X.shape[0], p=X.shape[1], classes=(len(np.unique(y)) if y is not None else "")))
        if "ACC" in EXP: run_acc(name, X)
        if "DEEP" in EXP: run_deep(name, X)
        if "ONLINE" in EXP: run_online(name, X)
        if "DOWN" in EXP: run_down(name, X, y)
        del X, y
    log("=== large-scale experiments finished ===")


if __name__ == "__main__":
    main()
