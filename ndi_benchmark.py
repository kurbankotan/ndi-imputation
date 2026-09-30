#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ndi_benchmark.py (v4) — complete experimental pipeline for the revised NDI manuscript
=================================================================================
One script, one run, every number the paper needs.

What it does
------------
1. Loads 15 complete, all-numeric OpenML classification datasets (n capped at N_CAP by
   stratified sub-sampling) and the three naturally-incomplete datasets of the original
   study (CKD, Heart, Mice; downloaded from the authors' GitHub repository), with the
   leakage problems of the original protocol fixed:
       Heart : duplicated rows removed (1,025 -> 302 unique records)
       Mice  : Genotype/Treatment/Behavior dropped from the features (they define the label)
       CKD   : pcv / wc / rc parsed as numbers (they contain '\\t?' tokens in the raw file)
2. Introduces missingness under MCAR (10/30/50 %), MAR (30 %) and MNAR (30 %) with
   SEEDS repetitions, imputes with every method and records z-RMSE, z-MAE, the share of
   imputed values outside the observed range of the column, and wall-clock time.
3. Downstream 5-fold stratified CV (LR, KNN, RF, SVC) on the imputed data for the 30 %
   settings, for the naturally-incomplete datasets (imputing only their own missing
   values) and on the complete data as a reference.
4. Runtime scaling on synthetic factor-model data, n up to 100,000.
5. Summary tables, Friedman + Nemenyi critical difference and Wilcoxon-Holm tests.

Version 4 adds two baselines that fit the factor model of NDI-F by maximum likelihood with EM
(EM-FA: diagonal uniquenesses; EM-PPCA: isotropic noise).  Their EM iteration counts are written
to method_info_long.csv.

Resumable: every finished (dataset, setting, seed, method) is appended to a CSV at once;
re-running the script skips what is already there.

Usage
-----
    PYTHONHASHSEED=3 DEEP_SEEDS=0 python ndi_benchmark.py   # full run as in the paper (about 12.5 h on 2 CPU cores + T4 GPU)
    python ndi_benchmark.py                 # same, but the four learning-based methods with seeds 0, 1 and 2
    QUICK=1 python ndi_benchmark.py         # smoke test (about 10 min) -> results_quick/
    python ndi_benchmark.py --summarize     # only rebuild the summary tables from results/
    PYTHONHASHSEED=3 DEEP_SEEDS=0 OUT_DIR=results python ndi_benchmark.py   # add the version-4 methods to the
                                            # shipped results: only the missing (dataset, setting, seed, method) runs

PYTHONHASHSEED=3 fixes the order in which hyperimpute tries its candidate learners on small columns, as in
the paper's run; without it the HyperImpute value on ecoli (MCAR 30 %) can change between runs (see README).

Requirements: numpy, pandas < 3, scipy, scikit-learn 1.6.1 (see requirements.txt). Optional, skipped
automatically when missing: torch (GAIN), hyperimpute (MIWAE, Sinkhorn, HyperImpute), timm and the
official ReMasker code in ./remasker (ReMasker).
"""
import os, sys, time, math, json, platform, shutil, warnings, urllib.request, hashlib
from collections import OrderedDict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (must precede IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from scipy import stats

# =============================================================================
#  CONFIGURATION
# =============================================================================
QUICK    = os.environ.get("QUICK", "0") == "1" or "--quick" in sys.argv
OUT_DIR  = os.environ.get("OUT_DIR", "results_quick" if QUICK else "results")
N_CAP    = 2000                       # max rows per dataset (stratified sub-sample)
SEEDS    = [0, 1, 2]                  # repetitions of every (dataset, setting)
SETTINGS = [("MCAR", .10), ("MCAR", .30), ("MCAR", .50), ("MAR", .30), ("MNAR", .30)]
DOWNSTREAM_SETTINGS = [("MCAR", .30), ("MAR", .30), ("MNAR", .30)]
NATURAL_EXTRA_MASK  = .20             # extra MCAR masking of observed cells in CKD/Heart/Mice (for RMSE)
GAIN_ITERS  = 2000
SCALING_N   = [1_000, 10_000, 100_000]
SCALING_P   = [10, 50]
# ---- version 3: deep / AutoML baselines (need `pip install hyperimpute timm` and the ReMasker repository)
DEEP_SETTINGS = [("MCAR", .30), ("MAR", .30), ("MNAR", .30)]   # the costly methods run only here (+ natural datasets)
DEEP_METHODS  = ["MIWAE", "Sinkhorn", "HyperImpute", "ReMasker"]
REMASKER_DIR  = os.environ.get("REMASKER_DIR", "remasker")     # git clone https://github.com/tydusky/remasker.git
REMASKER_EPOCHS = int(os.environ.get("REMASKER_EPOCHS", "600"))
DEEP_EPOCHS   = int(os.environ.get("DEEP_EPOCHS", "500"))      # MIWAE and Sinkhorn training epochs
DEEP_SEEDS    = [int(x) for x in os.environ.get("DEEP_SEEDS", "0,1,2").split(",")]   # e.g. DEEP_SEEDS=0 for a first pass

GITHUB_RAW = "https://raw.githubusercontent.com/kurbankotan/ndi-imputation/main/"
NATURAL_FILES = {"CKD": "kidney_disease.csv", "Heart": "heart.csv", "Mice": "Data_Cortex_Nuclear.csv"}

# (short name, OpenML data_id) — all-numeric classification datasets from several domains
OPENML = [
    ("iris", 61), ("wine", 187), ("glass", 41), ("ecoli", 39), ("ionosphere", 59),
    ("wdbc", 1510), ("diabetes", 37), ("vehicle", 54), ("yeast", 181), ("banknote", 1462),
    ("qsar-biodeg", 1494), ("wine-quality-red", 40691), ("segment", 36), ("spambase", 44),
    ("MagicTelescope", 1120),
]

if QUICK:
    SEEDS = [0]
    SETTINGS = [("MCAR", .30), ("MNAR", .30)]
    DOWNSTREAM_SETTINGS = [("MCAR", .30)]
    DEEP_SETTINGS = [("MCAR", .30)]
    DEEP_SEEDS = [0]
    GAIN_ITERS = 300
    REMASKER_EPOCHS = 20
    DEEP_EPOCHS = 30
    SCALING_N = [1_000, 10_000]
    OPENML = OPENML[:3]

os.makedirs(OUT_DIR, exist_ok=True)
RMSE_CSV  = os.path.join(OUT_DIR, "rmse_long.csv")
DOWN_CSV  = os.path.join(OUT_DIR, "downstream_long.csv")
SCALE_CSV = os.path.join(OUT_DIR, "scaling_long.csv")
DATA_CSV  = os.path.join(OUT_DIR, "datasets.csv")
INFO_CSV  = os.path.join(OUT_DIR, "method_info_long.csv")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# =============================================================================
#  PROPOSED METHOD(S)
# =============================================================================
class NDI:
    """Normal Deviate Imputation (NDI-0) and its two correlation-aware variants (NDI-C, NDI-S).

    All variants use only column means, column standard deviations and (for 'signed' and
    'shrink') the pairwise correlation matrix, each computed once from observed values.
    No model is fitted and there are no iterations.  Given z_ij = (x_ij - mu_j) / sigma_j
    for the observed cells of row i (set J_obs, size q), the missing cell k is imputed as
    mu_k + zbar_ik * sigma_k with

        variant='mean'   (NDI-0)       :  zbar_ik = (1/q) * sum_j z_ij
        variant='signed' (NDI-C)       :  zbar_ik = sum_j r_jk z_ij / sum_j |r_jk|
        variant='shrink' (NDI-S)       :  zbar_ik = sum_j r_jk z_ij / (1 + (q-1) * rho_ik),
                                          rho_ik = (1/q) sum_j |r_jk|

    'shrink' equals the exact Gaussian conditional mean when the columns are
    equicorrelated (up to sign), and reduces to r_jk z_ij when a single column is observed.
    """

    def __init__(self, variant="shrink"):
        assert variant in ("mean", "signed", "shrink")
        self.variant = variant

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.mu_ = np.nanmean(X, axis=0)
        self.sigma_ = np.nanstd(X, axis=0)
        self.sigma_[self.sigma_ == 0] = 1.0
        if self.variant != "mean":
            R = pd.DataFrame(X).corr().to_numpy()          # pairwise-complete Pearson r
            R = np.nan_to_num(R, nan=0.0)
            np.fill_diagonal(R, 0.0)
            self.R_ = R
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        Z = (X - self.mu_) / self.sigma_
        O = ~np.isnan(Z)
        Z0 = np.where(O, Z, 0.0)
        q = O.sum(axis=1, keepdims=True).astype(float)
        if self.variant == "mean":
            zbar = np.divide(Z0.sum(axis=1, keepdims=True), q, out=np.zeros_like(q), where=q > 0)
            zbar = np.repeat(zbar, X.shape[1], axis=1)
        else:
            num = Z0 @ self.R_                              # sum_j r_jk z_ij   (n x p)
            absR = np.abs(self.R_)
            if self.variant == "signed":
                den = O.astype(float) @ absR                # sum_j |r_jk|
            else:
                rho = np.divide(O.astype(float) @ absR, q, out=np.zeros_like(num), where=q > 0)
                den = 1.0 + (q - 1.0) * rho
            zbar = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return np.where(O, X, self.mu_ + zbar * self.sigma_)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class GaussCM:
    """Conditional-mean imputation under a multivariate normal model (Buck, 1960; the
    E-step of EM for the normal model), grouped by missingness pattern.  The pairwise-
    complete correlation matrix is projected onto a well-conditioned PSD matrix by
    clipping its eigenvalues at `ridge`."""

    def __init__(self, ridge=0.05):
        self.ridge = ridge

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.mu_ = np.nanmean(X, axis=0)
        self.sigma_ = np.nanstd(X, axis=0)
        self.sigma_[self.sigma_ == 0] = 1.0
        R = np.nan_to_num(pd.DataFrame(X).corr().to_numpy(), nan=0.0)
        np.fill_diagonal(R, 1.0)
        w, V = np.linalg.eigh(R)
        self.R_ = (V * np.clip(w, self.ridge, None)) @ V.T
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        Z = (X - self.mu_) / self.sigma_
        out = X.copy()
        M = np.isnan(Z)
        patterns = {}
        for i in np.where(M.any(axis=1))[0]:
            patterns.setdefault(M[i].tobytes(), []).append(i)
        for key, rows in patterns.items():
            m = np.frombuffer(key, dtype=bool)
            o = ~m
            rows = np.asarray(rows)
            if o.sum() == 0:
                out[np.ix_(rows, m)] = self.mu_[m]
                continue
            B = np.linalg.solve(self.R_[np.ix_(o, o)], self.R_[np.ix_(o, m)])
            out[np.ix_(rows, m)] = self.mu_[m] + (Z[np.ix_(rows, o)] @ B) * self.sigma_[m]
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class NDIF:
    """NDI-F: k-factor extension of NDI-S (closed form, row-separable, vectorised).

    The correlation matrix is approximated by a k-factor model R ~ L L' + Psi (principal factors of
    the pairwise-complete correlation matrix; Psi = diag(1 - sum_l L_jl^2), floored at `ridge`).
    Under z = L f + e, f ~ N(0, I_k), e ~ N(0, Psi), the conditional mean of the missing deviates of
    a row given its observed set O is

        f_hat = (I_k + L_O' Psi_O^-1 L_O)^-1 L_O' Psi_O^-1 z_O,      z_hat_M = L_M f_hat,

    which equals the exact Gaussian conditional mean whenever R = L L' + Psi.  With k = 1 and equal
    loadings sqrt(rho) this is exactly NDI-S; as k -> p it approaches GaussCM.  Cost O(n p k^2).
    k is chosen by the Kaiser rule (eigenvalues > 1), capped at `max_k`; `k` may also be an integer.
    `clip=True` restricts imputed values to the observed range of their column."""

    def __init__(self, k="kaiser", max_k=10, ridge=0.05, clip=True, chunk=8192):
        self.k, self.max_k, self.ridge, self.clip, self.chunk = k, max_k, ridge, clip, chunk

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        p = X.shape[1]
        self.mu_ = np.nanmean(X, axis=0)
        self.sigma_ = np.nanstd(X, axis=0)
        self.sigma_[self.sigma_ == 0] = 1.0
        self.lo_, self.hi_ = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
        R = np.nan_to_num(pd.DataFrame(X).corr().to_numpy(), nan=0.0)
        np.fill_diagonal(R, 1.0)
        w, V = np.linalg.eigh(R)
        w = np.clip(w, self.ridge, None)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        k = int((w > 1.0).sum()) if self.k == "kaiser" else int(self.k)
        self.k_ = max(1, min(k, self.max_k, p - 1))
        self.L_ = V[:, :self.k_] * np.sqrt(w[:self.k_])                 # p x k loadings
        self.psi_ = np.clip(1.0 - (self.L_ ** 2).sum(axis=1), self.ridge, None)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        Z = (X - self.mu_) / self.sigma_
        O = ~np.isnan(Z)
        Z0 = np.where(O, Z, 0.0)
        L, psi, k = self.L_, self.psi_, self.k_
        Lw = L / np.sqrt(psi)[:, None]                                   # lambda_j / sqrt(psi_j)
        B = (Z0 / psi) @ L                                               # n x k : sum_j O_ij z_ij lambda_j / psi_j
        Zhat = np.empty_like(Z0)
        for s in range(0, X.shape[0], self.chunk):                       # chunked to bound memory
            Oc = O[s:s + self.chunk].astype(float)
            G = np.einsum("ij,jk,jl->ikl", Oc, Lw, Lw, optimize=True) + np.eye(k)   # n_c x k x k
            F = np.linalg.solve(G, B[s:s + self.chunk][..., None])[..., 0]          # posterior factor means
            Zhat[s:s + self.chunk] = F @ L.T
        out = np.where(O, X, self.mu_ + Zhat * self.sigma_)
        if self.clip:
            out = np.where(O, out, np.clip(out, self.lo_, self.hi_))
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class CopulaWrap:
    """Gaussian-copula (normal-score) transform around an imputer that works on standardized data:
    each column is mapped to normal scores through its empirical CDF, the inner imputer fills the
    missing scores, and the imputed scores are mapped back through the empirical quantile function
    (imputed values therefore always lie inside the observed range)."""

    def __init__(self, inner):
        self.inner = inner

    def fit_transform(self, X, y=None):
        from scipy.stats import norm
        X = np.asarray(X, float)
        n, p = X.shape
        Z = np.full_like(X, np.nan)
        qs = []
        for j in range(p):
            o = ~np.isnan(X[:, j])
            v = X[o, j]
            r = pd.Series(v).rank(method="average").to_numpy()
            Z[o, j] = norm.ppf((r - 0.5) / o.sum())
            qs.append(np.sort(v))
        Zi = self.inner.fit_transform(Z)
        out = X.copy()
        for j in range(p):
            m = np.isnan(X[:, j])
            if m.any():
                q = qs[j]
                pos = np.clip(norm.cdf(Zi[m, j]) * len(q) - 0.5, 0, len(q) - 1)
                lo = np.floor(pos).astype(int)
                hi = np.minimum(lo + 1, len(q) - 1)
                fr = pos - lo
                out[m, j] = q[lo] * (1 - fr) + q[hi] * fr
        return out


# =============================================================================
#  BASELINE: THE FACTOR MODEL OF NDI-F FITTED BY MAXIMUM LIKELIHOOD (EM)
# =============================================================================
class EMFA:
    """Factor-model imputation fitted by maximum likelihood with the EM algorithm (baseline for NDI-F).

    Same model, same standardization and same number of factors as NDI-F: z = m + L f + e with
    f ~ N(0, I_k), e ~ N(0, Psi), on columns standardized with their observed mean and SD, and k chosen by
    the Kaiser rule on the pairwise-complete correlation matrix (capped at `max_k`).

        noise='fa'   : Psi = diag(psi_1, ..., psi_p)   maximum-likelihood factor analysis (Rubin & Thayer, 1982)
        noise='ppca' : Psi = sigma^2 I                  probabilistic PCA (Tipping & Bishop, 1999)

    m, L and Psi are estimated from the observed cells only, by EM with the factors as latent variables
    (Ilin & Raiko, 2010): the E-step computes the posterior of f given the observed cells of each row,
    the M-step regresses every column's observed cells on the posterior moments of [f, 1].  EM starts from
    the one-pass estimate of NDI-F and stops when the observed-data log-likelihood increases by less than
    `tol` per observed cell, or after `max_iter` iterations.  Missing cells are imputed with the posterior
    mean m_M + L_M E[f | z_O], as in NDI-F, without clipping.  Uniquenesses are floored at `psi_min`
    (0.005, the lower bound used by R's factanal); columns whose observed values are constant are left out
    of the model and imputed with that value.  Cost O(n p k^2) per iteration."""

    def __init__(self, noise="fa", k="kaiser", max_k=10, tol=1e-6, max_iter=1000, psi_min=0.005, chunk=8192):
        assert noise in ("fa", "ppca")
        self.noise, self.k, self.max_k, self.tol = noise, k, max_k, tol
        self.max_iter, self.psi_min, self.chunk = max_iter, psi_min, chunk

    def _estep(self, Z0, O, stats=False):
        """Posterior factor means for all rows, the observed-data log-likelihood and (optionally) the
        sufficient statistics of the M-step, accumulated over row chunks."""
        n, p = Z0.shape
        k, L, psi, m0 = self.k_, self.L_, self.psi_, self.m_
        LL = (L[:, :, None] * L[:, None, :]).reshape(p, k * k)
        ipsi, lpsi = 1.0 / psi, np.log(psi)
        F = np.empty((n, k))
        ll = 0.0
        A = np.zeros((p, (k + 1) * (k + 1))) if stats else None
        B = np.zeros((p, k + 1)) if stats else None
        for s in range(0, n, self.chunk):
            o = O[s:s + self.chunk]
            of = o.astype(float)
            z = Z0[s:s + self.chunk]
            r = np.where(o, z - m0, 0.0)                                   # centred observed cells
            M = ((of * ipsi) @ LL).reshape(-1, k, k) + np.eye(k)            # I + L_O' Psi_O^-1 L_O
            b = (r * ipsi) @ L                                              # L_O' Psi_O^-1 r_O
            G = np.linalg.inv(M)                                            # posterior covariance of f
            f = np.einsum("nkl,nl->nk", G, b)                               # posterior mean of f
            F[s:s + self.chunk] = f
            ll -= 0.5 * float(np.sum(of.sum(1) * np.log(2 * np.pi) + np.linalg.slogdet(M)[1]
                                     + of @ lpsi + (r * r) @ ipsi - np.einsum("nk,nk->n", b, f)))
            if stats:
                c = len(f)
                E = np.empty((c, k + 1, k + 1))                             # E[[f;1][f;1]'] per row
                E[:, :k, :k] = G + f[:, :, None] * f[:, None, :]
                E[:, :k, k] = f
                E[:, k, :k] = f
                E[:, k, k] = 1.0
                A += of.T @ E.reshape(c, -1)
                B += z.T @ np.hstack([f, np.ones((c, 1))])                  # z is 0 in missing cells
        return F, ll, A, B

    def _mstep(self, A, B, S, nobs):
        k, p = self.k_, len(S)
        W = np.linalg.solve(A.reshape(p, k + 1, k + 1), B[..., None])[..., 0]   # p x (k+1): [L, m]
        resid = S - np.einsum("jk,jk->j", W, B)                                  # residual sum of squares
        psi = resid / np.maximum(nobs, 1.0) if self.noise == "fa" else np.full(p, resid.sum() / nobs.sum())
        self.L_, self.m_ = W[:, :k], W[:, k]
        self.psi_ = np.clip(psi, self.psi_min, None)

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.mu_ = np.nanmean(X, axis=0)
        self.sigma_ = np.nanstd(X, axis=0)
        self.active_ = self.sigma_ > 0                     # constant columns are imputed with their value
        self.sigma_[~self.active_] = 1.0
        Xa = X[:, self.active_]
        start = NDIF(k=self.k, max_k=self.max_k, clip=False).fit(Xa)           # one-pass estimate, same k
        self.k_ = start.k_
        Z = (Xa - self.mu_[self.active_]) / self.sigma_[self.active_]
        O = ~np.isnan(Z)
        Z0 = np.where(O, Z, 0.0)
        nobs, S = O.sum(0).astype(float), (Z0 ** 2).sum(0)
        self.L_, self.m_ = start.L_.copy(), np.zeros(Xa.shape[1])
        self.psi_ = start.psi_.copy() if self.noise == "fa" else np.full(Xa.shape[1], max(self.psi_min, start.psi_.mean()))
        cells, prev, self.loglik_path_, self.converged_ = max(1.0, float(O.sum())), -np.inf, [], False
        for it in range(1, self.max_iter + 1):
            _, ll, A, B = self._estep(Z0, O, stats=True)
            self.loglik_path_.append(ll)
            if ll - prev < self.tol * cells:
                self.converged_ = True
                break
            prev = ll
            self._mstep(A, B, S, nobs)
        self.n_iter_, self.loglik_ = it, ll
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        a = self.active_
        Z = (X[:, a] - self.mu_[a]) / self.sigma_[a]
        O = ~np.isnan(Z)
        F = self._estep(np.where(O, Z, 0.0), O)[0]
        Zhat = np.zeros(X.shape)                                                 # constant columns: their value
        Zhat[:, a] = self.m_ + F @ self.L_.T
        return np.where(np.isnan(X), self.mu_ + Zhat * self.sigma_, X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


LAST_INFO = None          # extra information a method can report about its last run (EM iterations)


def m_emfa(X, seed, noise):
    global LAST_INFO
    mdl = EMFA(noise).fit(X)
    LAST_INFO = dict(iterations=mdl.n_iter_, k=mdl.k_, converged=mdl.converged_)
    return mdl.transform(X)


# =============================================================================
#  BASELINES
# =============================================================================
def m_mean(X, seed):
    X = np.asarray(X, float)
    return np.where(np.isnan(X), np.nanmean(X, axis=0), X)


def m_knn(X, seed):
    return KNNImputer(n_neighbors=5).fit_transform(X)


def m_mice(X, seed):
    return IterativeImputer(max_iter=10, tol=1e-3, random_state=seed).fit_transform(X)


def m_missforest(X, seed):
    est = RandomForestRegressor(n_estimators=20, max_features="sqrt", min_samples_leaf=3,
                                n_jobs=-1, random_state=seed)
    return IterativeImputer(estimator=est, max_iter=3, tol=1e-3, random_state=seed).fit_transform(X)


def m_softimpute(X, seed, max_iter=100, tol=1e-4):
    """SoftImpute (Mazumder et al., 2010): iterative soft-thresholded SVD on standardized data."""
    X = np.asarray(X, float)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    M = np.isnan(Z)
    Zf = np.where(M, 0.0, Z)
    lam = np.linalg.svd(Zf, compute_uv=False)[0] / 50.0
    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(Zf, full_matrices=False)
        Zhat = (U * np.maximum(s - lam, 0.0)) @ Vt
        Znew = np.where(M, Zhat, Z)
        delta = np.linalg.norm((Znew - Zf)[M]) / (np.linalg.norm(Zf[M]) + 1e-9)
        Zf = Znew
        if delta < tol:
            break
    return mu + Zf * sd


def m_gain(X, seed, n_iter=None, batch=128, hint_rate=0.9, alpha=100.0, lr=1e-3):
    """GAIN (Yoon et al., ICML 2018) — compact PyTorch re-implementation of the reference code."""
    import torch
    import torch.nn as nn
    n_iter = GAIN_ITERS if n_iter is None else n_iter
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    n, p = X.shape
    lo, hi = np.nanmin(X, 0), np.nanmax(X, 0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    Xn = (X - lo) / span
    M = (~np.isnan(Xn)).astype(np.float32)
    X0 = np.nan_to_num(Xn).astype(np.float32)
    Xt, Mt = torch.tensor(X0, device=device), torch.tensor(M, device=device)

    def mlp():
        return nn.Sequential(nn.Linear(2 * p, p), nn.ReLU(), nn.Linear(p, p), nn.ReLU(),
                             nn.Linear(p, p), nn.Sigmoid())

    G, D = mlp().to(device), mlp().to(device)
    optG = torch.optim.Adam(G.parameters(), lr=lr)
    optD = torch.optim.Adam(D.parameters(), lr=lr)
    bs, eps = min(batch, n), 1e-8
    for _ in range(n_iter):
        idx = torch.as_tensor(rng.choice(n, bs, replace=False), device=device)
        x, m = Xt[idx], Mt[idx]
        z = torch.rand(bs, p, device=device) * 0.01
        x_in = m * x + (1 - m) * z
        b = (torch.rand(bs, p, device=device) < hint_rate).float()
        h = m * b + 0.5 * (1 - b)
        # discriminator step
        g = G(torch.cat([x_in, m], 1)).detach()
        d = D(torch.cat([m * x + (1 - m) * g, h], 1))
        lossD = -torch.mean(m * torch.log(d + eps) + (1 - m) * torch.log(1 - d + eps))
        optD.zero_grad(); lossD.backward(); optD.step()
        # generator step
        g = G(torch.cat([x_in, m], 1))
        d = D(torch.cat([m * x + (1 - m) * g, h], 1))
        lossG = -torch.mean((1 - m) * torch.log(d + eps)) \
                + alpha * torch.mean((m * x - m * g) ** 2) / torch.mean(m)
        optG.zero_grad(); lossG.backward(); optG.step()
    with torch.no_grad():
        z = torch.rand(n, p, device=device) * 0.01
        g = G(torch.cat([Mt * Xt + (1 - Mt) * z, Mt], 1)).cpu().numpy()
    return (M * X0 + (1 - M) * g) * span + lo


def _hi_imputers():
    from hyperimpute.plugins.imputers import Imputers
    return Imputers()


def _standardized(fn):
    """Run a neural imputer on z-scored columns (as in the MIWAE / OT papers) and map the result back."""
    def wrapped(X, seed):
        X = np.asarray(X, float)
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
        sd[sd == 0] = 1.0
        return np.asarray(fn((X - mu) / sd, seed), float) * sd + mu
    return wrapped


@_standardized
def m_miwae(X, seed):
    """MIWAE (Mattei & Frellsen, ICML 2019) via the hyperimpute package; 128 hidden units as in the paper."""
    p = X.shape[1]
    plug = _hi_imputers().get("miwae", n_epochs=DEEP_EPOCHS, batch_size=256, latent_size=min(10, max(1, p // 2)),
                              n_hidden=128, K=20, random_state=seed)
    return plug.fit_transform(pd.DataFrame(X)).to_numpy(float)


@_standardized
def m_sinkhorn(X, seed):
    """Optimal-transport imputation with Sinkhorn divergences (Muzellec et al., ICML 2020) via hyperimpute."""
    plug = _hi_imputers().get("sinkhorn", n_epochs=DEEP_EPOCHS, batch_size=512, random_state=seed)
    return plug.fit_transform(pd.DataFrame(X)).to_numpy(float)


HI_INNER_ITER = int(os.environ.get("HI_INNER_ITER", "10"))


def m_hyperimpute(X, seed):
    """HyperImpute (Jarrett et al., ICML 2022): iterative imputation with automatic per-column model selection
    among linear, random-forest and gradient-boosting learners (a reduced candidate set to bound the cost)."""
    plug = _hi_imputers().get("hyperimpute", optimizer="simple",
                              classifier_seed=["logistic_regression", "random_forest", "xgboost"],
                              regression_seed=["linear_regression", "random_forest_regressor", "xgboost_regressor"],
                              n_inner_iter=HI_INNER_ITER, select_patience=3, random_state=seed)
    return plug.fit_transform(pd.DataFrame(np.asarray(X, float))).to_numpy(float)


def m_remasker(X, seed):
    """ReMasker (Du, Melis & Wang, ICLR 2024): official implementation, default configuration of the authors
    (embed 32, encoder depth 4, decoder depth 2, 4 heads, mask ratio 0.5, 600 epochs)."""
    import importlib
    if REMASKER_DIR not in sys.path:
        sys.path.insert(0, REMASKER_DIR)
    rm = importlib.import_module("remasker_impute")
    argv_backup = sys.argv
    sys.argv = ["remasker", "--max_epochs", str(REMASKER_EPOCHS), "--embed_dim", "32", "--depth", "4", "--decoder_depth", "2",
                "--num_heads", "4", "--mlp_ratio", "4", "--mask_ratio", "0.5", "--encode_func", "linear",
                "--batch_size", "64", "--seed", str(seed)]
    try:
        model = rm.ReMasker()
    finally:
        sys.argv = argv_backup
    import torch
    torch.manual_seed(seed)
    return np.asarray(model.fit_transform(pd.DataFrame(np.asarray(X, float))), float)


METHODS = OrderedDict([
    ("Mean",        m_mean),
    ("NDI-0",       lambda X, s: NDI("mean").fit_transform(X)),      # the method as originally described
    ("NDI-C",       lambda X, s: NDI("signed").fit_transform(X)),
    ("NDI-S",       lambda X, s: NDI("shrink").fit_transform(X)),
    ("GaussCM",     lambda X, s: GaussCM().fit_transform(X)),
    ("KNN",         m_knn),
    ("MICE",        m_mice),
    ("MissForest",  m_missforest),
    ("SoftImpute",  m_softimpute),
    ("GAIN",        m_gain),
    # ---- version 2 additions (the resume logic only computes what is not yet in results/)
    ("NDI-F1",      lambda X, s: NDIF(k=1, clip=False).fit_transform(X)),   # one estimated factor, no clipping
    ("NDI-F-noclip",lambda X, s: NDIF(clip=False).fit_transform(X)),        # Kaiser k, no clipping (ablation)
    ("NDI-F",       lambda X, s: NDIF().fit_transform(X)),                  # proposed: Kaiser k + clipping
    ("Copula-NDI-F",lambda X, s: CopulaWrap(NDIF(clip=False)).fit_transform(X)),
    # ---- version 4 additions: the same factor model fitted by maximum likelihood (EM)
    ("EM-FA",       lambda X, s: m_emfa(X, s, "fa")),                       # diagonal uniquenesses
    ("EM-PPCA",     lambda X, s: m_emfa(X, s, "ppca")),                     # isotropic noise
    # ---- version 3 additions: run only in DEEP_SETTINGS and on the natural datasets
    ("MIWAE",       m_miwae),
    ("Sinkhorn",    m_sinkhorn),
    ("HyperImpute", m_hyperimpute),
    ("ReMasker",    m_remasker),
])
try:
    import hyperimpute  # noqa: F401
except Exception:
    for _m in ("MIWAE", "Sinkhorn", "HyperImpute"):
        METHODS.pop(_m, None)
    log("hyperimpute not installed -> MIWAE, Sinkhorn and HyperImpute skipped (pip install hyperimpute)")
if "HyperImpute" in METHODS:
    # hyperimpute's internals rely on pandas chained assignment; under pandas >= 3 (copy-on-write) the
    # imputed values are silently discarded and the plugin returns the mean fill.  Detect that here.
    try:
        _rng = np.random.default_rng(0)
        _Xc = _rng.standard_normal((300, 2)) @ _rng.standard_normal((2, 6)) + 0.3 * _rng.standard_normal((300, 6))
        _Xm = _Xc.copy(); _Xm[_rng.random(_Xm.shape) < 0.3] = np.nan
        _out = _hi_imputers().get("ice").fit_transform(pd.DataFrame(_Xm)).to_numpy(float)
        _mean_fill = np.where(np.isnan(_Xm), np.nanmean(_Xm, 0), _Xm)
        if np.allclose(_out, _mean_fill):
            METHODS.pop("HyperImpute", None)
            log("!! hyperimpute returned the mean fill on a sanity check (pandas >= 3 copy-on-write) -> HyperImpute skipped; install pandas<3")
    except Exception as _e:
        METHODS.pop("HyperImpute", None)
        log(f"!! hyperimpute sanity check failed ({type(_e).__name__}) -> HyperImpute skipped")
if not os.path.exists(os.path.join(REMASKER_DIR, "remasker_impute.py")):
    METHODS.pop("ReMasker", None)
    log(f"ReMasker code not found in '{REMASKER_DIR}' -> ReMasker skipped (git clone https://github.com/tydusky/remasker.git)")
DEEP_METHODS = [m for m in DEEP_METHODS if m in METHODS]
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False
    METHODS.pop("GAIN")
    log("torch not available -> GAIN skipped (pip install torch to include it)")

REFERENCE_METHOD = "NDI-F"          # used for the pairwise Wilcoxon tests

# =============================================================================
#  DATASETS
# =============================================================================
def _get_file(fname):
    if not os.path.exists(fname):
        log(f"downloading {fname} from GitHub")
        urllib.request.urlretrieve(GITHUB_RAW + fname, fname)
    return fname


def _strip_strings(df):
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip()
    return df


def _onehot(df_cat):
    df_cat = df_cat.copy()
    for c in df_cat.columns:
        df_cat[c] = df_cat[c].fillna(df_cat[c].mode().iloc[0]).astype(str)
    return pd.get_dummies(df_cat, drop_first=True).to_numpy(float)


def load_ckd():
    df = _strip_strings(pd.read_csv(_get_file("kidney_disease.csv")).drop(columns="id"))
    num = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
    cat = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    y = (df["classification"].astype(str) == "ckd").astype(int).to_numpy()
    return dict(name="CKD", natural=True, X_num=df[num].reset_index(drop=True),
                X_cat=_onehot(df[cat]), y=y, source="UCI 336 (authors' GitHub copy)")


def load_heart():
    df = pd.read_csv(_get_file("heart.csv")).drop_duplicates().reset_index(drop=True)
    df.loc[df["thal"] == 0, "thal"] = np.nan       # out-of-range sentinels in this version
    df.loc[df["ca"] == 4, "ca"] = np.nan
    num = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    cat = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    return dict(name="Heart", natural=True, X_num=df[num].astype(float), X_cat=_onehot(df[cat]),
                y=df["target"].astype(int).to_numpy(), source="UCI 45 (1,025-row version, de-duplicated)")


def load_mice():
    df = pd.read_csv(_get_file("Data_Cortex_Nuclear.csv"))
    prot = df.columns[1:78].tolist()                      # 77 protein expression features
    y = pd.factorize(df["class"].astype(str))[0]
    return dict(name="Mice", natural=True, X_num=df[prot].astype(float), X_cat=None, y=y,
                source="UCI 342 (Genotype/Treatment/Behavior excluded: they define the label)")


def load_openml(name, data_id):
    from sklearn.datasets import fetch_openml
    for attempt in range(3):
        try:
            d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
            break
        except Exception as e:
            if attempt == 2:
                raise
            log(f"   OpenML {data_id}: {type(e).__name__}, retrying in 10 s")
            time.sleep(10)
    X = d.data.select_dtypes(include=[np.number]).copy()
    X = X.loc[:, X.nunique() > 1]                          # drop constant columns
    y = d.target
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok].astype(str)
    counts = y.value_counts()
    keep = y.isin(counts[counts >= 10].index)              # classes too small for 5-fold CV
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    if len(X) > N_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=N_CAP, stratify=y, random_state=0)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    y = pd.factorize(y)[0]
    openml_name = getattr(d, "details", {}).get("name", "?") if hasattr(d, "details") else "?"
    return dict(name=name, natural=False, X_num=X.astype(float), X_cat=None, y=y,
                source=f"OpenML {data_id} ({openml_name})")


def load_all():
    datasets = []
    for name, did in OPENML:
        try:
            ds = load_openml(name, did)
            datasets.append(ds)
            log(f"loaded {name:<18s} n={len(ds['y']):5d} p={ds['X_num'].shape[1]:3d} classes={len(np.unique(ds['y']))}  [{ds['source']}]")
        except Exception as e:
            log(f"!! could not load {name} (OpenML {did}): {type(e).__name__}: {str(e)[:80]} -> skipped")
    for loader in (load_ckd, load_heart, load_mice):
        try:
            ds = loader()
            datasets.append(ds)
            miss = ds["X_num"].isna().mean().mean()
            log(f"loaded {ds['name']:<18s} n={len(ds['y']):5d} p={ds['X_num'].shape[1]:3d} classes={len(np.unique(ds['y']))}  natural missing={miss:.1%}")
        except Exception as e:
            log(f"!! could not load {loader.__name__}: {type(e).__name__}: {str(e)[:80]} -> skipped")
    rows = [dict(dataset=d["name"], natural=d["natural"], n=len(d["y"]), p=d["X_num"].shape[1],
                 classes=len(np.unique(d["y"])), natural_missing_share=float(d["X_num"].isna().mean().mean()),
                 categorical_onehot_cols=0 if d["X_cat"] is None else d["X_cat"].shape[1], source=d["source"])
            for d in datasets]
    pd.DataFrame(rows).to_csv(DATA_CSV, index=False)
    return datasets


# =============================================================================
#  MISSINGNESS MECHANISMS  (MCAR / MAR / MNAR, logistic models as in Muzellec et al., 2020)
# =============================================================================
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _intercept_for_rate(logit, rate):
    lo, hi = -100.0, 100.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _sigmoid(logit + mid).mean() < rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_mask(X, mech, rate, rng):
    n, p = X.shape
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    mask = np.zeros((n, p), dtype=bool)
    if mech == "MCAR":
        mask = rng.random((n, p)) < rate
    elif mech == "MAR":
        n_obs = max(1, int(round(0.3 * p)))              # 30 % of the columns stay fully observed
        obs_cols = rng.choice(p, n_obs, replace=False)
        mis_cols = np.setdiff1d(np.arange(p), obs_cols)
        W = rng.standard_normal((n_obs, len(mis_cols)))
        logits = Z[:, obs_cols] @ W
        logits = logits / (logits.std(0) + 1e-9)
        r_adj = min(0.95, rate * p / len(mis_cols))       # so that the overall rate is `rate`
        for c, k in enumerate(mis_cols):
            b = _intercept_for_rate(logits[:, c], r_adj)
            mask[:, k] = rng.random(n) < _sigmoid(logits[:, c] + b)
    elif mech == "MNAR":                                   # self-masking logistic
        for j in range(p):
            logit = rng.choice([-1.0, 1.0]) * Z[:, j]
            b = _intercept_for_rate(logit, rate)
            mask[:, j] = rng.random(n) < _sigmoid(logit + b)
    else:
        raise ValueError(mech)
    for j in range(p):                                     # keep >= 2 observed cells per column
        if (~mask[:, j]).sum() < 2:
            mask[rng.choice(n, 2, replace=False), j] = False
    return mask


# =============================================================================
#  EVALUATION
# =============================================================================
def score(X_true, X_imp, mask, sd, lo, hi):
    if X_imp is None or np.isnan(X_imp).any():
        left = -1 if X_imp is None else int(np.isnan(X_imp).sum())
        return dict(rmse=np.nan, mae=np.nan, oob=np.nan, nan_left=left)
    cols = np.where(mask)[1]
    d = (X_imp[mask] - X_true[mask]) / sd[cols]
    oob = np.mean((X_imp[mask] < lo[cols]) | (X_imp[mask] > hi[cols]))
    return dict(rmse=float(np.sqrt(np.mean(d ** 2))), mae=float(np.mean(np.abs(d))), oob=float(oob), nan_left=0)


CLASSIFIERS = OrderedDict([
    ("LR",  lambda s: LogisticRegression(max_iter=2000)),
    ("KNN", lambda s: KNeighborsClassifier(n_neighbors=5)),
    ("RF",  lambda s: RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1)),
    ("SVC", lambda s: SVC()),
])


def downstream(X_num, X_cat, y, seed):
    X = X_num if X_cat is None else np.hstack([X_num, X_cat])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = {}
    for name, mk in CLASSIFIERS.items():
        r = cross_validate(make_pipeline(StandardScaler(), mk(seed)), X, y, cv=cv,
                           scoring=["accuracy", "f1_macro"], n_jobs=1)
        out[name] = (float(r["test_accuracy"].mean()), float(r["test_f1_macro"].mean()))
    return out


def rkey(r):
    """canonical string for a missing rate, stable across CSV round-trips"""
    return f"{float(r):.4f}"


def stable_seed(name, seed, rate, mech):
    h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    return [seed, h, int(round(float(rate) * 10_000)), len(mech)]


def append_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def done_keys(path, cols):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    if "rate" in cols:
        df["rate"] = df["rate"].map(rkey)
    return set(map(tuple, df[cols].astype(str).to_numpy()))


def run_method(fn, X_in, seed):
    global LAST_INFO
    LAST_INFO = None
    t0 = time.perf_counter()
    try:
        X_imp = np.asarray(fn(X_in, seed), float)
        err = ""
    except Exception as e:                       # a failing baseline must not kill the run
        X_imp, err = None, f"{type(e).__name__}: {str(e)[:60]}"
    return X_imp, time.perf_counter() - t0, err


# =============================================================================
#  MAIN EXPERIMENT
# =============================================================================
def run_imputation_benchmark(datasets):
    key_cols = ["dataset", "mechanism", "rate", "seed", "method"]
    done_r = done_keys(RMSE_CSV, key_cols)
    done_d = done_keys(DOWN_CSV, key_cols)
    total = sum((len(SETTINGS) if not d["natural"] else 1) * len(SEEDS) for d in datasets)
    cfg_i, t_start = 0, time.perf_counter()

    for ds in datasets:
        X_true = ds["X_num"].to_numpy(float)
        n, p = X_true.shape
        settings = [("MCAR", NATURAL_EXTRA_MASK)] if ds["natural"] else SETTINGS
        for mech, rate in settings:
            for seed in SEEDS:
                cfg_i += 1
                rng = np.random.default_rng(stable_seed(ds["name"], seed, rate, mech))
                observed = ~np.isnan(X_true)
                if ds["natural"]:
                    mask = observed & (rng.random((n, p)) < rate)
                else:
                    mask = make_mask(X_true, mech, rate, rng)
                X_in = X_true.copy()
                X_in[mask] = np.nan
                sd = np.nanstd(X_true, axis=0)
                sd[sd == 0] = 1.0
                lo, hi = np.nanmin(X_in, axis=0), np.nanmax(X_in, axis=0)
                do_down = (mech, rate) in DOWNSTREAM_SETTINGS and not ds["natural"]
                for mname, fn in METHODS.items():
                    if mname in DEEP_METHODS and (seed not in DEEP_SEEDS or (not ds["natural"] and (mech, rate) not in DEEP_SETTINGS)):
                        continue
                    key = (ds["name"], mech, rkey(rate), str(seed), mname)
                    if key in done_r and (not do_down or key in done_d):
                        continue
                    X_imp, dt, err = run_method(fn, X_in, seed)
                    sc = score(X_true, X_imp, mask, sd, lo, hi)
                    if key not in done_r:
                        append_row(RMSE_CSV, dict(dataset=ds["name"], natural=ds["natural"], n=n, p=p,
                                                  mechanism=mech, rate=rate, seed=seed, method=mname,
                                                  rmse=sc["rmse"], mae=sc["mae"], oob=sc["oob"],
                                                  nan_left=sc["nan_left"], time_s=dt, error=err))
                        if LAST_INFO:
                            append_row(INFO_CSV, dict(dataset=ds["name"], mechanism=mech, rate=rate, seed=seed,
                                                      method=mname, **LAST_INFO))
                    if do_down and key not in done_d and X_imp is not None and not np.isnan(X_imp).any():
                        for clf, (acc, f1) in downstream(X_imp, ds["X_cat"], ds["y"], seed).items():
                            append_row(DOWN_CSV, dict(dataset=ds["name"], natural=False, mechanism=mech,
                                                      rate=rate, seed=seed, method=mname, classifier=clf,
                                                      accuracy=acc, f1_macro=f1))
                el = time.perf_counter() - t_start
                log(f"[{cfg_i:3d}/{total}] {ds['name']:<18s} {mech:4s} {rate:.2f} seed={seed}  "
                    f"elapsed {el/60:5.1f} min, ETA {el/60/cfg_i*(total-cfg_i):5.1f} min")

    # --- downstream on the complete data (reference) and on the natural datasets ---
    for ds in datasets:
        for seed in SEEDS:
            if ds["natural"]:
                X_nat = ds["X_num"].to_numpy(float)
                rate = round(float(np.isnan(X_nat).mean()), 4)
                for mname, fn in METHODS.items():
                    key = (ds["name"], "natural", rkey(rate), str(seed), mname)
                    if key in done_d:
                        continue
                    X_imp, dt, err = run_method(fn, X_nat, seed)
                    if X_imp is None or np.isnan(X_imp).any():
                        continue
                    for clf, (acc, f1) in downstream(X_imp, ds["X_cat"], ds["y"], seed).items():
                        append_row(DOWN_CSV, dict(dataset=ds["name"], natural=True, mechanism="natural",
                                                  rate=rate, seed=seed, method=mname, classifier=clf,
                                                  accuracy=acc, f1_macro=f1))
                log(f"downstream (natural missingness) {ds['name']} seed={seed} done")
            else:
                key = (ds["name"], "none", rkey(0.0), str(seed), "COMPLETE")
                if key in done_d:
                    continue
                for clf, (acc, f1) in downstream(ds["X_num"].to_numpy(float), ds["X_cat"], ds["y"], seed).items():
                    append_row(DOWN_CSV, dict(dataset=ds["name"], natural=False, mechanism="none", rate=0.0,
                                              seed=seed, method="COMPLETE", classifier=clf,
                                              accuracy=acc, f1_macro=f1))
        log(f"downstream reference/natural for {ds['name']} done")


# =============================================================================
#  RUNTIME SCALING (synthetic factor-model data, MCAR 30 %)
# =============================================================================
SCALING_SKIP = {          # methods whose cost makes them impractical at these sizes
    "KNN": 5_000, "MissForest": 5_000, "MICE": 20_000,
    "MIWAE": 1_000, "Sinkhorn": 1_000, "HyperImpute": 1_000, "ReMasker": 1_000,
}


def run_scaling():
    done = done_keys(SCALE_CSV, ["n", "p", "method"])
    for n in SCALING_N:
        for p in SCALING_P:
            rng = np.random.default_rng(1)
            F = rng.standard_normal((n, 3))
            L = rng.standard_normal((3, p))
            X_true = F @ L + 0.5 * rng.standard_normal((n, p))
            mask = rng.random((n, p)) < 0.3
            X_in = X_true.copy()
            X_in[mask] = np.nan
            sd = X_true.std(0)
            for mname, fn in METHODS.items():
                if (str(n), str(p), mname) in done:
                    continue
                if n > SCALING_SKIP.get(mname, np.inf):
                    append_row(SCALE_CSV, dict(n=n, p=p, method=mname, time_s=np.nan, rmse=np.nan, status="skipped (too costly)"))
                    continue
                X_imp, dt, err = run_method(fn, X_in, 0)
                rmse = np.nan if X_imp is None else float(np.sqrt(np.mean(((X_imp[mask] - X_true[mask]) / sd[np.where(mask)[1]]) ** 2)))
                append_row(SCALE_CSV, dict(n=n, p=p, method=mname, time_s=dt, rmse=rmse, status=err or "ok"))
                if LAST_INFO:
                    append_row(INFO_CSV, dict(dataset=f"scaling n={n} p={p}", mechanism="MCAR", rate=0.3, seed=0,
                                              method=mname, **LAST_INFO))
                log(f"scaling n={n:>7d} p={p:3d} {mname:<11s} {dt:8.2f} s  rmse={rmse:.3f}")


# =============================================================================
#  WARM-START EXPERIMENT: MissForest initialised with the column mean vs with NDI-F
# =============================================================================
WARM_CSV = os.path.join(OUT_DIR, "warmstart_long.csv")
WARM_SETTINGS = [("MCAR", .30)]


def missforest_loop(X, X_init, seed, sd, X_true, mask, max_iter=8):
    """Plain MissForest (Stekhoven & Buehlmann 2012) with an explicit initial fill and the original
    stopping rule (stop as soon as the change between successive imputations increases, and return
    the previous imputation).  Returns the imputed matrix, the number of iterations run and the
    z-RMSE after every iteration (the last entry is the returned imputation)."""
    M = np.isnan(X)
    Xf = X_init.copy()
    cols = [j for j in np.argsort(M.sum(0)) if M[:, j].any()]
    hist, prev_delta, Xprev = [], np.inf, None
    for it in range(max_iter):
        Xold = Xf.copy()
        for j in cols:
            others = np.arange(X.shape[1]) != j
            rf = RandomForestRegressor(n_estimators=20, max_features="sqrt", min_samples_leaf=3, n_jobs=-1, random_state=seed)
            rf.fit(Xf[~M[:, j]][:, others], X[~M[:, j], j])
            Xf[M[:, j], j] = rf.predict(Xf[M[:, j]][:, others])
        d = (Xf[mask] - X_true[mask]) / sd[np.where(mask)[1]]
        hist.append(float(np.sqrt(np.mean(d ** 2))))
        delta = float(np.mean((((Xf - Xold) / sd)[M]) ** 2))
        if delta > prev_delta:                      # change started to increase -> keep the previous one
            hist.pop(); Xf = Xold
            break
        prev_delta = delta
    return Xf, len(hist), hist


def run_warmstart(datasets):
    done = done_keys(WARM_CSV, ["dataset", "mechanism", "rate", "seed", "init"])
    for ds in datasets:
        if ds["natural"]:
            continue
        X_true = ds["X_num"].to_numpy(float)
        n, p = X_true.shape
        for mech, rate in WARM_SETTINGS:
            for seed in SEEDS:
                rng = np.random.default_rng(stable_seed(ds["name"], seed, rate, mech))
                mask = make_mask(X_true, mech, rate, rng)
                X_in = X_true.copy(); X_in[mask] = np.nan
                sd = np.nanstd(X_true, axis=0); sd[sd == 0] = 1.0
                for init_name, init_fn in [("mean", lambda A: m_mean(A, seed)), ("NDI-F", lambda A: NDIF().fit_transform(A))]:
                    key = (ds["name"], mech, rkey(rate), str(seed), init_name)
                    if key in done:
                        continue
                    t0 = time.perf_counter()
                    X0 = init_fn(X_in)
                    d0 = (X0[mask] - X_true[mask]) / sd[np.where(mask)[1]]
                    Xf, iters, hist = missforest_loop(X_in, X0, seed, sd, X_true, mask)
                    dt = time.perf_counter() - t0
                    best = min(hist)
                    within1 = next(i + 1 for i, h in enumerate(hist) if h <= best * 1.01)
                    append_row(WARM_CSV, dict(dataset=ds["name"], mechanism=mech, rate=rate, seed=seed, init=init_name,
                                              rmse_init=float(np.sqrt(np.mean(d0 ** 2))), rmse_iter1=hist[0],
                                              rmse_final=hist[-1], iterations=iters, iters_to_within_1pct=within1,
                                              time_s=dt, rmse_history=";".join(f"{h:.4f}" for h in hist)))
                log(f"warm start {ds['name']:<18s} {mech} {rate:.2f} seed={seed} done")


# =============================================================================
#  STATISTICS AND SUMMARY TABLES
# =============================================================================
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}


def q_nemenyi(k, alpha=0.05):
    """Critical value of the Nemenyi test: studentized range q(alpha, k, inf) / sqrt(2) (Demsar, 2006)."""
    try:
        from scipy.stats import studentized_range
        return float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))
    except Exception:
        return _Q05.get(k, 3.391)


def friedman_report(block, title):
    """block: DataFrame (rows = datasets, columns = methods) of a lower-is-better metric."""
    block = block.dropna(axis=1, how="any")
    lines = [f"\n### {title}   (N = {len(block)} datasets, k = {block.shape[1]} methods)"]
    if len(block) < 3 or block.shape[1] < 3:
        lines.append("   not enough complete blocks for the Friedman test")
        return "\n".join(lines), None
    ranks = block.rank(axis=1, method="average")
    avg = ranks.mean().sort_values()
    chi2, pval = stats.friedmanchisquare(*[block[c].to_numpy() for c in block.columns])
    k, N = block.shape[1], len(block)
    cd = q_nemenyi(k) * math.sqrt(k * (k + 1) / (6.0 * N))
    lines.append(f"   Friedman chi2 = {chi2:.2f}, p = {pval:.2e};  Nemenyi CD (alpha=0.05) = {cd:.3f}")
    best = avg.index[0]
    for m, r in avg.items():
        flag = "" if m == best else ("  (not significantly worse than the best)" if r - avg.iloc[0] < cd else "  *")
        lines.append(f"   {m:<12s} average rank {r:5.2f}{flag}")
    lines.append("   * = significantly worse than the best-ranked method (Nemenyi, alpha = 0.05)")
    return "\n".join(lines), avg


def wilcoxon_report(block, ref):
    lines = [f"\n### Wilcoxon signed-rank, {ref} vs each method (blocks = dataset x setting, Holm-corrected)"]
    if ref not in block.columns:
        return "\n".join(lines + ["   reference method not present"])
    rows = []
    for m in block.columns:
        if m == ref:
            continue
        pair = block[[ref, m]].dropna()
        if len(pair) < 5:
            continue
        diff = pair[ref] - pair[m]
        wins, losses = int((diff < 0).sum()), int((diff > 0).sum())
        try:
            pv = stats.wilcoxon(pair[ref], pair[m]).pvalue if (diff != 0).any() else 1.0
        except Exception:
            pv = np.nan
        rows.append((m, len(pair), wins, losses, float(diff.mean()), pv))
    rows.sort(key=lambda r: r[5])
    mtests = len(rows)
    for i, (m, nb, w, l, md, pv) in enumerate(rows):
        p_holm = min(1.0, pv * (mtests - i)) if not np.isnan(pv) else np.nan
        lines.append(f"   vs {m:<12s} blocks={nb:3d}  {ref} better in {w:3d}, worse in {l:3d};  "
                     f"mean diff (z-RMSE) = {md:+.4f};  p = {pv:.2e}, Holm p = {p_holm:.2e}")
    return "\n".join(lines)


def summarize():
    out = []
    if not os.path.exists(RMSE_CSV):
        log("no rmse_long.csv found — nothing to summarize")
        return
    rm = pd.read_csv(RMSE_CSV).drop_duplicates(subset=["dataset", "mechanism", "rate", "seed", "method"], keep="last")
    rm["setting"] = rm["mechanism"] + "-" + (rm["rate"] * 100).round().astype(int).astype(str) + "%"
    rm.loc[rm["natural"] == True, "setting"] = "natural+MCAR-" + str(int(NATURAL_EXTRA_MASK * 100)) + "%"
    synth = rm[rm["natural"] == False]

    # --- Table: mean z-RMSE / MAE per setting and method (over datasets and seeds) + average rank
    per_ds = synth.groupby(["setting", "dataset", "method"], as_index=False)[["rmse", "mae", "oob", "time_s"]].mean()
    tab = per_ds.groupby(["setting", "method"]).agg(rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
                                                     mae_mean=("mae", "mean"), oob_mean=("oob", "mean"),
                                                     time_mean_s=("time_s", "mean")).reset_index()
    ranks = []
    for s, g in per_ds.groupby("setting"):
        blk = g.pivot(index="dataset", columns="method", values="rmse")
        r = blk.rank(axis=1).mean().rename("avg_rank").reset_index()
        r["setting"] = s
        ranks.append(r)
    tab = tab.merge(pd.concat(ranks), on=["setting", "method"], how="left").sort_values(["setting", "rmse_mean"])
    tab.to_csv(os.path.join(OUT_DIR, "summary_rmse.csv"), index=False)
    out.append("## Mean z-RMSE by setting (lower is better)\n" +
               tab.pivot(index="method", columns="setting", values="rmse_mean").round(3).to_string())
    out.append("\n## Average rank by setting (lower is better)\n" +
               tab.pivot(index="method", columns="setting", values="avg_rank").round(2).to_string())

    # --- Table: per-dataset z-RMSE for every setting
    for s, g in per_ds.groupby("setting"):
        g.pivot(index="dataset", columns="method", values="rmse").round(3).to_csv(
            os.path.join(OUT_DIR, f"per_dataset_rmse_{s.replace('%', '')}.csv"))

    # --- natural datasets (extra masking)
    nat = rm[rm["natural"] == True]
    if len(nat):
        t = nat.groupby(["dataset", "method"])["rmse"].mean().unstack().round(3)
        t.to_csv(os.path.join(OUT_DIR, "natural_rmse.csv"))
        out.append("\n## z-RMSE on the naturally-incomplete datasets (extra MCAR masking of observed cells)\n" + t.to_string())

    # --- out-of-range share
    oob = rm.groupby("method")["oob"].mean().sort_values().round(4)
    oob.to_csv(os.path.join(OUT_DIR, "summary_out_of_range.csv"))
    out.append("\n## Share of imputed values outside the observed range of the column (all settings)\n" + oob.to_string())

    # --- Friedman / Nemenyi / Wilcoxon
    rep = []
    for s, g in per_ds.groupby("setting"):
        txt, _ = friedman_report(g.pivot(index="dataset", columns="method", values="rmse"), f"Setting {s}")
        rep.append(txt)
    per_ds["block"] = per_ds["dataset"] + "|" + per_ds["setting"]
    allblk = per_ds.pivot(index="block", columns="method", values="rmse")
    txt, _ = friedman_report(allblk, "All settings pooled (blocks = dataset x setting; methods with a complete set of blocks)")
    rep.append(txt)
    rep.append(wilcoxon_report(allblk, REFERENCE_METHOD))
    deep_names = [f"{m}-{int(r*100)}%" for m, r in DEEP_SETTINGS]
    sub = per_ds[per_ds.setting.isin(deep_names)]
    if len(sub):
        subblk = sub.pivot(index="block", columns="method", values="rmse")
        txt, _ = friedman_report(subblk, f"Pooled over {', '.join(deep_names)} (blocks = dataset x setting; all methods incl. deep baselines)")
        rep.append(txt)
        rep.append(wilcoxon_report(subblk, REFERENCE_METHOD).replace("blocks = dataset x setting", "blocks = dataset x 30 % settings"))
    with open(os.path.join(OUT_DIR, "friedman_nemenyi_wilcoxon.txt"), "w") as f:
        f.write("\n".join(rep))
    out.append("\n## Statistical tests\n" + "\n".join(rep))

    # --- errors
    errs = rm[rm["error"].fillna("") != ""]
    if len(errs):
        out.append("\n## Method failures\n" + errs.groupby(["method", "error"]).size().to_string())

    # --- downstream
    if os.path.exists(DOWN_CSV):
        dn = pd.read_csv(DOWN_CSV).drop_duplicates(subset=["dataset", "mechanism", "rate", "seed", "method", "classifier"], keep="last")
        dn["setting"] = dn["mechanism"] + "-" + (dn["rate"] * 100).round().astype(int).astype(str) + "%"
        syn = dn[dn["natural"] == False].copy()
        syn.loc[syn["method"] == "COMPLETE", "setting"] = "complete data"
        t = syn.groupby(["setting", "classifier", "method"])[["accuracy", "f1_macro"]].mean().reset_index()
        t.to_csv(os.path.join(OUT_DIR, "summary_downstream.csv"), index=False)
        acc = syn.groupby(["setting", "method"])["accuracy"].mean().unstack(0).round(4)
        out.append("\n## Downstream accuracy, mean over datasets, seeds and the 4 classifiers\n" + acc.to_string())
        comp = syn[syn["method"] == "COMPLETE"].groupby("dataset")["accuracy"].mean()
        ref_txt = "\n## Complete-data reference accuracy per dataset (mean over classifiers)\n" + comp.round(4).to_string()
        out.append(ref_txt)
        for dsn, g in dn[dn["natural"] == True].groupby("dataset"):
            t = g.groupby(["method", "classifier"])["accuracy"].mean().unstack().round(4)
            t["mean"] = t.mean(axis=1).round(4)
            t.to_csv(os.path.join(OUT_DIR, f"natural_downstream_{dsn}.csv"))
            out.append(f"\n## Downstream accuracy on {dsn} (its own missing values imputed; 5-fold CV, mean over seeds)\n" + t.to_string())

    # --- warm start
    if os.path.exists(WARM_CSV):
        wm = pd.read_csv(WARM_CSV).drop_duplicates(subset=["dataset", "mechanism", "rate", "seed", "init"], keep="last")
        t = wm.groupby("init").agg(rmse_init=("rmse_init", "mean"), rmse_after_1_iter=("rmse_iter1", "mean"),
                                   rmse_final=("rmse_final", "mean"), iterations=("iterations", "mean"),
                                   iters_to_within_1pct=("iters_to_within_1pct", "mean"), time_s=("time_s", "mean")).round(4)
        t.to_csv(os.path.join(OUT_DIR, "summary_warmstart.csv"))
        out.append("\n## MissForest warm start (own loop, 20 trees, Stekhoven stopping rule, max 8 iterations, MCAR 30 %): mean over datasets and seeds\n" + t.to_string())
        per = wm.pivot_table(index="dataset", columns="init", values=["iterations", "time_s", "rmse_final"]).round(3)
        per.to_csv(os.path.join(OUT_DIR, "warmstart_per_dataset.csv"))

    # --- scaling
    if os.path.exists(SCALE_CSV):
        sc = pd.read_csv(SCALE_CSV)
        t = sc.pivot_table(index="method", columns=["p", "n"], values="time_s")
        t.round(2).to_csv(os.path.join(OUT_DIR, "summary_scaling.csv"))
        out.append("\n## Wall-clock seconds on synthetic data (MCAR 30 %); blank = skipped as too costly\n" + t.round(2).to_string())

    with open(os.path.join(OUT_DIR, "SUMMARY.md"), "w") as f:
        f.write("# NDI benchmark — summary\n\n```\n" + "\n".join(out) + "\n```\n")
    print("\n".join(out))


def write_env():
    info = dict(python=platform.python_version(), platform=platform.platform(), cpu_count=os.cpu_count(),
                numpy=np.__version__, pandas=pd.__version__, quick=QUICK, seeds=SEEDS, settings=SETTINGS,
                n_cap=N_CAP, gain_iters=GAIN_ITERS, methods=list(METHODS))
    try:
        import sklearn, scipy
        info["sklearn"], info["scipy"] = sklearn.__version__, scipy.__version__
    except Exception:
        pass
    if HAVE_TORCH:
        import torch
        info["torch"] = torch.__version__
        info["gain_device"] = "cuda: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    path = os.path.join(OUT_DIR, "env.json")
    if os.path.exists(path):                  # extending existing results: keep the record of the first run
        path = os.path.join(OUT_DIR, time.strftime("env_resumed_%Y%m%d-%H%M%S.json"))
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)


def main():
    if "--summarize" in sys.argv:
        summarize()
        return
    log(f"=== NDI benchmark v4 {'(QUICK smoke test)' if QUICK else '(full run)'} -> {OUT_DIR}/  methods: {len(METHODS)} ===")
    write_env()
    datasets = load_all()
    if not datasets:
        log("no datasets could be loaded — check the network connection"); return
    run_imputation_benchmark(datasets)
    if SCALING_N:
        run_scaling()
    run_warmstart(datasets)
    summarize()
    zip_path = shutil.make_archive(OUT_DIR, "zip", OUT_DIR)
    log(f"=== finished. All results in {OUT_DIR}/ and {zip_path} ===")


if __name__ == "__main__":
    main()
