#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ndi_extra.py — additional experiments for the NDI-F manuscript (resumable; results in OUT_DIR, default results_extra/)

    EM     early-stopped EM-FA started from NDI-F (1, 2, 3, 5, 10, 20, 50 iterations and until convergence), and the
           iterations, time and error of converged EM-FA from three starts (NDI-F, mean-filled PCA, random)
    K      sensitivity of NDI-F to the number of factors k (1, 2, 3, 5, Kaiser, 10, 20, Kaiser uncapped) and to the
           eigenvalue/uniqueness floor (0.01, 0.05, 0.1, 0.2)
    RATE   the 15 complete datasets under MCAR 70 % and 80 % (the main benchmark stops at 50 %)
    SMALLN the small-sample regime: wide datasets (p >= 30) sub-sampled to n = 50, 100, 200, 500 rows, MCAR 30 %
    SCALEP runtime and error against the number of columns on synthetic factor-model data (n = 10,000 and 100,000;
           p = 50, 100, 200, 500; MCAR 30 %), with the number of distinct missingness patterns, plus one case with
           only 20 patterns
    LARGE  nine larger real datasets from PMLB (n up to 70,000, p up to 719) under MCAR, MAR and MNAR 30 %

Usage:  python extra/get_data.py                     (downloads the PMLB copies and the banknote file once)
        EXP=EM,K python extra/ndi_extra.py          (any subset of EM,K,RATE,SMALLN,SCALEP,LARGE; default: all)
        PAPER_NAMES=wine,ecoli python extra/ndi_extra.py   (restrict the benchmark datasets)
The datasets, missingness generators, methods and scoring are imported from ndi_benchmark.py (same masks as the paper:
the mask of (dataset, mechanism, rate, seed) is reproduced from stable_seed()).
"""
import os, sys, time, json, platform
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)       # extra/ -> repository root
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
os.environ.setdefault("OUT_DIR", os.path.join(ROOT, "results_extra"))
OUT = os.environ["OUT_DIR"]; os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd
import ndi_benchmark as nb
from ndi_benchmark import (NDIF, NDI, GaussCM, EMFA, make_mask, score, stable_seed, append_row, done_keys, rkey, log,
                           m_mean, m_knn, m_mice, m_missforest, m_softimpute)
import data_local as dl
from ndi_benchmark import NDIFCV

EXP = os.environ.get("EXP", "EM,K,RATE,SMALLN,SCALEP,LARGE").split(",")
SEEDS = nb.SEEDS
CSV = {k: os.path.join(OUT, f) for k, f in dict(EM="em_iterations_long.csv", K="k_sensitivity_long.csv",
       RATE="high_rate_long.csv", SMALLN="small_n_long.csv", SCALEP="scaling_p_long.csv", LARGE="large_long.csv").items()}


EXTRA_METHODS = {"NDI-F-k20": lambda X, s: NDIF(max_k=20).fit_transform(X),
                 "NDI-F-kall": lambda X, s: NDIF(max_k=10_000).fit_transform(X),          # Kaiser rule without the cap of 10
                 "EM-FA-k20": lambda X, s: EMFA(max_k=20).fit_transform(X),
                 "NDI-F-V": lambda X, s: NDIFCV(seed=s).fit_transform(X)}                # k chosen by internal validation
def method(name):
    return nb.METHODS.get(name) or EXTRA_METHODS[name]


def timed(fn, *a):
    t0 = time.perf_counter(); out = fn(*a); return out, time.perf_counter() - t0


def z_rmse(X_true, X_imp, mask, sd):
    d = (X_imp[mask] - X_true[mask]) / sd[np.where(mask)[1]]
    return float(np.sqrt(np.mean(d ** 2)))


def masked(ds, mech, rate, seed):
    X_true = ds["X_num"].to_numpy(float)
    rng = np.random.default_rng(stable_seed(ds["name"], seed, rate, mech))
    mask = make_mask(X_true, mech, rate, rng)
    X_in = X_true.copy(); X_in[mask] = np.nan
    sd = np.nanstd(X_true, axis=0); sd[sd == 0] = 1.0
    return X_true, X_in, mask, sd


# ----------------------------------------------------------------------------- EM with other starts
class EMFAStart(EMFA):
    """EM-FA with a choice of starting point: 'ndif' (the paper's EM-FA), 'pca' (principal factors of the
    correlation matrix of the mean-filled standardized data) or 'random' (seeded N(0, 1/p) loadings, psi = 1)."""
    def __init__(self, start="ndif", seed=0, **kw):
        super().__init__(**kw); self.start, self.seed = start, seed

    def fit(self, X, y=None):
        if self.start == "ndif":
            return super().fit(X)
        X = np.asarray(X, float)
        self.mu_ = np.nanmean(X, axis=0); self.sigma_ = np.nanstd(X, axis=0)
        self.active_ = self.sigma_ > 0; self.sigma_[~self.active_] = 1.0
        Xa = X[:, self.active_]
        k_ref = NDIF(k=self.k, max_k=self.max_k, clip=False).fit(Xa)           # same k as NDI-F / EM-FA
        self.k_ = k_ref.k_
        Z = (Xa - self.mu_[self.active_]) / self.sigma_[self.active_]
        O = ~np.isnan(Z); Z0 = np.where(O, Z, 0.0); p = Z.shape[1]
        if self.start == "pca":
            R = np.corrcoef(Z0, rowvar=False); R = np.nan_to_num(R); np.fill_diagonal(R, 1.0)
            w, V = np.linalg.eigh(R); order = np.argsort(w)[::-1][:self.k_]
            self.L_ = V[:, order] * np.sqrt(np.clip(w[order], 0.05, None))
            self.psi_ = np.clip(1.0 - (self.L_ ** 2).sum(1), self.psi_min, None)
        else:
            rng = np.random.default_rng(self.seed)
            self.L_ = rng.standard_normal((p, self.k_)) / np.sqrt(p); self.psi_ = np.ones(p)
        self.m_ = np.zeros(p)
        if self.noise == "ppca": self.psi_ = np.full(p, max(self.psi_min, float(self.psi_.mean())))
        nobs, S = O.sum(0).astype(float), (Z0 ** 2).sum(0)
        cells, prev, self.loglik_path_, self.converged_ = max(1.0, float(O.sum())), -np.inf, [], False
        for it in range(1, self.max_iter + 1):
            _, ll, A, B = self._estep(Z0, O, stats=True)
            self.loglik_path_.append(ll)
            if ll - prev < self.tol * cells:
                self.converged_ = True; break
            prev = ll
            self._mstep(A, B, S, nobs)
        self.n_iter_, self.loglik_ = it, ll
        return self


# ----------------------------------------------------------------------------- EM: early stopping and starts
EM_CAPS = [1, 2, 3, 5, 10, 20, 50]

def run_em(datasets):
    done = done_keys(CSV["EM"], ["dataset", "mechanism", "rate", "seed", "variant"])
    for ds in datasets:
        for mech, rate in nb.SETTINGS:
            for seed in SEEDS:
                X_true, X_in, mask, sd = masked(ds, mech, rate, seed)
                def rec(variant, X_imp, dt, extra):
                    append_row(CSV["EM"], dict(dataset=ds["name"], n=X_true.shape[0], p=X_true.shape[1], mechanism=mech, rate=rate,
                                               seed=seed, variant=variant, rmse=z_rmse(X_true, X_imp, mask, sd), time_s=dt, **extra))
                key = lambda v: (ds["name"], mech, rkey(rate), str(seed), v)
                if key("NDI-F") not in done:
                    X_imp, dt = timed(lambda: NDIF().fit_transform(X_in)); rec("NDI-F", X_imp, dt, dict(n_iter=0, converged=True, loglik=np.nan, k=np.nan))
                if key("NDI-F-noclip") not in done:
                    X_imp, dt = timed(lambda: NDIF(clip=False).fit_transform(X_in)); rec("NDI-F-noclip", X_imp, dt, dict(n_iter=0, converged=True, loglik=np.nan, k=np.nan))
                for cap in EM_CAPS:                                      # exactly `cap` EM iterations from the NDI-F start
                    v = f"EM-FA@{cap}"
                    if key(v) in done: continue
                    def run():
                        m = EMFA(tol=0.0, max_iter=cap).fit(X_in); return m, m.transform(X_in)
                    (m, X_imp), dt = timed(run)
                    rec(v, X_imp, dt, dict(n_iter=m.n_iter_, converged=m.converged_, loglik=m.loglik_, k=m.k_))
                for start in ["ndif", "pca", "random"]:                  # converged EM-FA from three starts
                    v = f"EM-FA@conv:{start}"
                    if key(v) in done: continue
                    def run():
                        m = EMFAStart(start=start, seed=seed).fit(X_in); return m, m.transform(X_in)
                    (m, X_imp), dt = timed(run)
                    rec(v, X_imp, dt, dict(n_iter=m.n_iter_, converged=m.converged_, loglik=m.loglik_, k=m.k_))
                log(f"EM {ds['name']:<18s} {mech:4s} {rate:.2f} seed={seed} done")


# ----------------------------------------------------------------------------- K: sensitivity of NDI-F
K_VARIANTS = [("k=1", dict(k=1)), ("k=2", dict(k=2)), ("k=3", dict(k=3)), ("k=5", dict(k=5)), ("k=Kaiser", dict()),
              ("k=10", dict(k=10)), ("k=20", dict(k=20, max_k=20)), ("k=Kaiser,max_k=p-1", dict(max_k=10_000)),
              ("floor=0.01", dict(ridge=0.01)), ("floor=0.1", dict(ridge=0.1)), ("floor=0.2", dict(ridge=0.2)), ("k=validated", "cv")]

def run_k(datasets):
    done = done_keys(CSV["K"], ["dataset", "mechanism", "rate", "seed", "variant"])
    for ds in datasets:
        for mech, rate in nb.SETTINGS:
            for seed in SEEDS:
                X_true, X_in, mask, sd = masked(ds, mech, rate, seed)
                for v, kw in K_VARIANTS:
                    if (ds["name"], mech, rkey(rate), str(seed), v) in done: continue
                    def run():
                        m = (NDIFCV(seed=seed) if kw == "cv" else NDIF(**kw)).fit(X_in); return m, m.transform(X_in)
                    (m, X_imp), dt = timed(run)
                    append_row(CSV["K"], dict(dataset=ds["name"], n=X_true.shape[0], p=X_true.shape[1], mechanism=mech, rate=rate, seed=seed,
                                              variant=v, k_used=m.k_, rmse=z_rmse(X_true, X_imp, mask, sd), time_s=dt))
        log(f"K {ds['name']} done")


# ----------------------------------------------------------------------------- RATE: 70 % and 80 % MCAR
RATE_METHODS = ["Mean", "NDI-0", "NDI-S", "NDI-F", "GaussCM", "EM-FA", "EM-PPCA", "KNN", "MICE", "MissForest", "SoftImpute", "NDI-F-V"]
RATES = [0.7, 0.8]

def run_rate(datasets):
    done = done_keys(CSV["RATE"], ["dataset", "mechanism", "rate", "seed", "method"])
    for ds in datasets:
        for rate in RATES:
            for seed in SEEDS:
                X_true, X_in, mask, sd = masked(ds, "MCAR", rate, seed)
                lo, hi = np.nanmin(X_in, axis=0), np.nanmax(X_in, axis=0)
                for mname in RATE_METHODS:
                    if (ds["name"], "MCAR", rkey(rate), str(seed), mname) in done: continue
                    X_imp, dt, err = nb.run_method(method(mname), X_in, seed)
                    sc = score(X_true, X_imp, mask, sd, lo, hi)
                    append_row(CSV["RATE"], dict(dataset=ds["name"], n=X_true.shape[0], p=X_true.shape[1], mechanism="MCAR", rate=rate, seed=seed,
                                                 method=mname, rmse=sc["rmse"], mae=sc["mae"], oob=sc["oob"], nan_left=sc["nan_left"], time_s=dt, error=err))
                log(f"RATE {ds['name']:<18s} MCAR {rate:.2f} seed={seed} done")


# ----------------------------------------------------------------------------- SMALLN: the small-sample regime
SMALLN_METHODS = ["Mean", "NDI-S", "NDI-F", "GaussCM", "EM-FA", "EM-PPCA", "KNN", "MICE", "MissForest", "SoftImpute", "NDI-F-V"]
SMALLN_N = [50, 100, 200, 500]

def run_smalln(datasets):
    done = done_keys(CSV["SMALLN"], ["dataset", "n_sub", "seed", "method"])
    for ds in datasets:
        X_all = ds["X_num"].to_numpy(float)
        for n_sub in SMALLN_N:
            if n_sub > X_all.shape[0]: continue
            for seed in SEEDS:
                rng = np.random.default_rng(stable_seed(ds["name"] + f"@{n_sub}", seed, 0.3, "MCAR"))
                rows = rng.choice(X_all.shape[0], n_sub, replace=False)
                X_true = X_all[rows]
                X_true = X_true[:, X_true.std(0) > 0]                       # columns constant in the sub-sample are dropped
                mask = make_mask(X_true, "MCAR", 0.3, rng)
                X_in = X_true.copy(); X_in[mask] = np.nan
                sd = X_true.std(0); lo, hi = np.nanmin(X_in, axis=0), np.nanmax(X_in, axis=0)
                for mname in SMALLN_METHODS:
                    if (ds["name"], str(n_sub), str(seed), mname) in done: continue
                    X_imp, dt, err = nb.run_method(method(mname), X_in, seed)
                    sc = score(X_true, X_imp, mask, sd, lo, hi)
                    append_row(CSV["SMALLN"], dict(dataset=ds["name"], n_sub=n_sub, p=X_true.shape[1], seed=seed, method=mname,
                                                   rmse=sc["rmse"], time_s=dt, error=err))
            log(f"SMALLN {ds['name']:<18s} n={n_sub} done")


# ----------------------------------------------------------------------------- SCALEP: synthetic, growing p
SCALEP_METHODS = ["Mean", "NDI-S", "NDI-F", "GaussCM", "EM-FA", "EM-PPCA", "SoftImpute", "NDI-F-V"]
SCALEP_CASES = [(n, p, "mcar") for n in (10_000, 100_000) for p in (50, 100, 200, 500)] + [(100_000, 200, "20patterns")]
SOFTIMPUTE_MAX_CELLS = 2.5e7

def run_scalep():
    done = done_keys(CSV["SCALEP"], ["n", "p", "patterns", "method"])
    for n, p, pat in SCALEP_CASES:
        rng = np.random.default_rng(1)
        F = rng.standard_normal((n, 3)); L = rng.standard_normal((3, p))
        X_true = F @ L + 0.5 * rng.standard_normal((n, p))                # the generator of the paper's scaling experiment
        if pat == "mcar":
            mask = rng.random((n, p)) < 0.3
        else:                                                             # 20 distinct missingness patterns, 30 % missing each
            pats = rng.random((20, p)) < 0.3
            mask = pats[rng.integers(0, 20, n)]
        X_in = X_true.copy(); X_in[mask] = np.nan
        sd = X_true.std(0)
        n_pat = len({m.tobytes() for m in mask[mask.any(1)]})
        for mname in SCALEP_METHODS:
            if (str(n), str(p), pat, mname) in done: continue
            if mname == "SoftImpute" and n * p > SOFTIMPUTE_MAX_CELLS:
                append_row(CSV["SCALEP"], dict(n=n, p=p, patterns=pat, n_patterns=n_pat, method=mname, time_s=np.nan, rmse=np.nan, status="skipped (too costly)")); continue
            X_imp, dt, err = nb.run_method(method(mname), X_in, 0)
            rmse = np.nan if X_imp is None else z_rmse(X_true, X_imp, mask, sd)
            info = nb.LAST_INFO or {}
            append_row(CSV["SCALEP"], dict(n=n, p=p, patterns=pat, n_patterns=n_pat, method=mname, time_s=dt, rmse=rmse,
                                           status=err or "ok", em_iterations=info.get("iterations", np.nan)))
            log(f"SCALEP n={n:>7d} p={p:3d} {pat:<10s} {mname:<10s} {dt:8.2f} s  rmse={rmse:.4f}")


# ----------------------------------------------------------------------------- LARGE: real datasets from PMLB
LARGE_METHODS = ["Mean", "NDI-0", "NDI-S", "NDI-F", "GaussCM", "EM-FA", "EM-PPCA", "SoftImpute", "KNN", "MICE", "MissForest",
                 "NDI-F-k20", "NDI-F-kall", "EM-FA-k20", "NDI-F-V"]        # larger factor caps and the validated k (second pass)
LARGE_SKIP = dict(nb.SCALING_SKIP)                                        # the paper's size caps for the costly methods
LARGE_SETTINGS = [("MCAR", .30), ("MAR", .30), ("MNAR", .30)]

def run_large():
    done = done_keys(CSV["LARGE"], ["dataset", "mechanism", "rate", "seed", "method"])
    for name, _ in dl.LARGE:
        ds = dl.load_large_dataset(name)
        X_all = ds["X_num"].to_numpy(float); n, p = X_all.shape
        settings = LARGE_SETTINGS if name != "mnist" else [("MCAR", .30)]
        for mech, rate in settings:
            for seed in (SEEDS if n <= 10_000 else [0]):
                X_true, X_in, mask, sd = masked(ds, mech, rate, seed)
                lo, hi = np.nanmin(X_in, axis=0), np.nanmax(X_in, axis=0)
                for mname in LARGE_METHODS:
                    if (name, mech, rkey(rate), str(seed), mname) in done: continue
                    if n > LARGE_SKIP.get(mname, np.inf) or (mname == "SoftImpute" and n * p > SOFTIMPUTE_MAX_CELLS):
                        append_row(CSV["LARGE"], dict(dataset=name, n=n, p=p, mechanism=mech, rate=rate, seed=seed, method=mname,
                                                      rmse=np.nan, time_s=np.nan, error="skipped (too costly)")); continue
                    X_imp, dt, err = nb.run_method(method(mname), X_in, seed)
                    sc = score(X_true, X_imp, mask, sd, lo, hi)
                    info = nb.LAST_INFO or {}
                    append_row(CSV["LARGE"], dict(dataset=name, n=n, p=p, mechanism=mech, rate=rate, seed=seed, method=mname,
                                                  rmse=sc["rmse"], mae=sc["mae"], oob=sc["oob"], time_s=dt, error=err,
                                                  em_iterations=info.get("iterations", np.nan)))
                    log(f"LARGE {name:<14s} n={n:6d} p={p:3d} {mech:4s} seed={seed} {mname:<10s} {dt:8.2f} s  rmse={sc['rmse']:.4f}")
        info_csv = os.path.join(OUT, "large_datasets.csv")
        if not (os.path.exists(info_csv) and name in set(pd.read_csv(info_csv).dataset)):
            append_row(info_csv, dict(dataset=name, n=n, p=p, classes=len(np.unique(ds["y"])), source=ds["source"]))


def timing_check():
    rng = np.random.default_rng(1); n, p = 100_000, 50
    X = rng.standard_normal((n, 3)) @ rng.standard_normal((3, p)) + 0.5 * rng.standard_normal((n, p))
    X[rng.random((n, p)) < 0.3] = np.nan
    NDIF().fit_transform(X)                                                 # warm-up
    t = min(timed(lambda: NDIF().fit_transform(X))[1] for _ in range(3))
    with open(os.path.join(OUT, "timing_check_extra.txt"), "w") as f:
        f.write(f"NDI-F on 100,000 x 50: {t:.3f} s on this machine, 1.833 s in the paper's run (ratio {t/1.833:.2f})\n")
        f.write(open("/proc/cpuinfo").read().split("\n\n")[0] + "\n")
    log(f"timing check: NDI-F 100,000 x 50 = {t:.3f} s (paper's machine 1.833 s)")


def main():
    json.dump(dict(python=platform.python_version(), platform=platform.platform(), cpu_count=os.cpu_count(), numpy=np.__version__,
                   pandas=pd.__version__, sklearn=__import__("sklearn").__version__, experiments=EXP, seeds=SEEDS,
                   time=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(OUT, "env_extra.json"), "w"), indent=1)
    if not os.path.exists(os.path.join(OUT, "timing_check_extra.txt")): timing_check()
    names_env = os.environ.get("PAPER_NAMES")                                  # e.g. only the datasets verified against the paper
    paper = dl.load_paper_datasets(names_env.split(",") if names_env else dl.PAPER)
    names = [d["name"] for d in paper]
    log(f"paper datasets available: {len(paper)}/15 ({', '.join(n for n in dl.PAPER if n not in names) or 'all'} missing)")
    pd.DataFrame([dict(dataset=d["name"], n=len(d["y"]), p=d["X_num"].shape[1], source=d["source"]) for d in paper]).to_csv(os.path.join(OUT, "paper_datasets_used.csv"), index=False)
    for exp in EXP:
        if exp == "SCALEP": run_scalep()
        elif exp == "K": run_k(paper)
        elif exp == "EM": run_em(paper)
        elif exp == "RATE": run_rate(paper)
        elif exp == "SMALLN":
            wide = [d for d in paper if d["X_num"].shape[1] >= 30] + [dl.load_large_dataset(nm) for nm in ("satimage", "optdigits", "texture", "musk2", "mfeat-factors")]
            run_smalln(wide)
        elif exp == "LARGE": run_large()
    log("=== extra experiments finished ===")


if __name__ == "__main__":
    main()
