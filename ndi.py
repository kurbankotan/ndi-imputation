# -*- coding: utf-8 -*-
"""
ndi.py — Normal Deviate Imputation: NDI-0, NDI-C, NDI-S (class NDI), NDI-F (class NDIF), the same factor
model fitted by maximum likelihood with EM (class EMFA) and the Gaussian conditional mean (class GaussCM).
Extracted verbatim from ndi_benchmark.py, the script that produced every result in the paper. Requires numpy
and pandas (scipy only for CopulaWrap).

    from ndi import NDIF
    imp = NDIF()                               # NDI-F: Kaiser k, clipping to the observed range (recommended)
    X_train_imp = imp.fit_transform(X_train)   # numpy array, NaN = missing, numerical columns only
    X_test_imp  = imp.transform(X_test)        # uses the training statistics (mu, sigma, loadings, uniquenesses)

    from ndi import NDI
    NDI("shrink")   # NDI-S      NDI("signed")  # NDI-C      NDI("mean")  # NDI-0 (the original rule)

    from ndi import NDIFCV
    NDIFCV()        # NDI-F-V: k chosen by internal validation (v4.1; ~10x the cost of NDI-F, still one pass, no iteration)

    from ndi import EMFA
    EMFA("fa")      # EM-FA: maximum-likelihood fit of the NDI-F model by EM, started from NDI-F (more accurate, ~100x slower)
    EMFA("fa", max_iter=3, tol=0.0)   # three EM iterations from the NDI-F start: ~80 % of EM's gain at ~4x the cost of NDI-F

Reference: Kırışoğlu, S. & Kotan, K. Normal Deviate Imputation (NDI-F): A One-Pass Alternative to EM for
Factor-Model Imputation of Numerical Tabular Data (under review).
"""
import numpy as np
import pandas as pd

__all__ = ["NDI", "NDIF", "NDIFCV", "EMFA", "GaussCM", "CopulaWrap", "pairwise_corr"]



def pairwise_corr(X, chunk=20_000):
    """Pairwise-complete Pearson correlation matrix computed with four matrix products (identical to
    pandas.DataFrame.corr() up to floating-point rounding, and much faster for large p: the pandas routine
    loops over the column pairs in Cython).  Version 4.1; the main benchmark was timed with the pandas call."""
    X = np.asarray(X, float)
    n, p = X.shape
    N = np.zeros((p, p)); Sx = np.zeros((p, p)); Sxx = np.zeros((p, p)); Sxy = np.zeros((p, p))
    for s in range(0, n, chunk):                      # chunked to bound memory
        Xc = X[s:s + chunk]
        O = (~np.isnan(Xc)).astype(float)
        X0 = np.where(O > 0, Xc, 0.0)
        N += O.T @ O                                  # co-observed counts
        Sx += X0.T @ O                                # Sx[j, k] = sum of x_j over the rows where j and k are observed
        Sxx += (X0 * X0).T @ O
        Sxy += X0.T @ X0
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = Sx / N; my = mx.T
        cov = Sxy / N - mx * my
        vx = Sxx / N - mx ** 2; vy = vx.T
        R = cov / np.sqrt(vx * vy)
    R[~np.isfinite(R)] = 0.0
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -1.0, 1.0)

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
            R = pairwise_corr(X)                           # pairwise-complete Pearson r
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
        R = pairwise_corr(X)
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
        R = pairwise_corr(X)
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
        chunk = max(256, min(self.chunk, int(2e7 // (k * k))))              # bound the n_c x k x k workspace (~160 MB)
        for s in range(0, X.shape[0], chunk):                            # chunked to bound memory
            Oc = O[s:s + chunk].astype(float)
            G = np.einsum("ij,jk,jl->ikl", Oc, Lw, Lw, optimize=True) + np.eye(k)   # n_c x k x k
            F = np.linalg.solve(G, B[s:s + chunk][..., None])[..., 0]               # posterior factor means
            Zhat[s:s + chunk] = F @ L.T
        out = np.where(O, X, self.mu_ + Zhat * self.sigma_)
        if self.clip:
            out = np.where(O, out, np.clip(out, self.lo_, self.hi_))
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class NDIFCV(NDIF):
    """NDI-F-V: NDI-F with the number of factors chosen by internal validation (version 4.1).

    A validation set is formed by hiding `val_share` of the observed cells of at most `val_rows` (seeded) rows;
    the pairwise-complete correlation matrix of the remaining cells is factored once, every candidate k imputes
    the hidden cells with the first k principal factors (same eigenvectors, psi recomputed), and the k with the
    lowest validation z-RMSE is refitted on all observed cells.  Deterministic given `seed`; two correlation
    passes and one closed-form imputation of the validation rows per candidate; no iteration, no model fitting.
    Candidates: 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100 (those below p and below n/5).  About ten times the
    cost of NDI-F on datasets of a few thousand rows, about twice at n = 100,000 (the validation is capped at
    5,000 rows)."""
    CANDIDATES = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100)

    def __init__(self, val_share=0.10, val_rows=5000, seed=0, candidates=CANDIDATES, ridge=0.05, clip=True, chunk=8192):
        super().__init__(k="kaiser", max_k=10, ridge=ridge, clip=clip, chunk=chunk)
        self.val_share, self.val_rows, self.seed, self.candidates = val_share, val_rows, seed, candidates

    def _factors(self, X):
        R = pairwise_corr(X)
        w, V = np.linalg.eigh(R)
        w = np.clip(w, self.ridge, None)
        order = np.argsort(w)[::-1]
        return w[order], V[:, order]

    def _set_k(self, w, V, k):
        self.k_ = k
        self.L_ = V[:, :k] * np.sqrt(w[:k])
        self.psi_ = np.clip(1.0 - (self.L_ ** 2).sum(axis=1), self.ridge, None)

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        n, p = X.shape
        self.mu_ = np.nanmean(X, axis=0); self.sigma_ = np.nanstd(X, axis=0); self.sigma_[self.sigma_ == 0] = 1.0
        self.lo_, self.hi_ = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
        obs = ~np.isnan(X)
        rng = np.random.default_rng(self.seed)
        rows = np.sort(rng.choice(n, self.val_rows, replace=False)) if n > self.val_rows else np.arange(n)
        hide_sub = obs[rows] & (rng.random((len(rows), p)) < self.val_share)     # hidden cells, within the validation rows
        Xtr = X.copy(); Xtr[rows[np.where(hide_sub)[0]], np.where(hide_sub)[1]] = np.nan
        cands = [k for k in self.candidates if k <= p - 1 and k <= max(1, n // 5)] or [1]
        w, V = self._factors(Xtr)
        Xv, truth = Xtr[rows], X[rows][hide_sub]
        sd = self.sigma_[np.where(hide_sub)[1]]
        self.val_rmse_ = {}
        clip, self.clip = self.clip, False
        for k in cands:
            self._set_k(w, V, k)
            Xi = self.transform(Xv)
            self.val_rmse_[k] = float(np.sqrt(np.mean(((Xi[hide_sub] - truth) / sd) ** 2)))
        self.clip = clip
        self.k_cv_ = min(self.val_rmse_, key=self.val_rmse_.get)
        w, V = self._factors(X)                                   # final fit on all observed cells
        self._set_k(w, V, self.k_cv_)
        return self


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
        chunk = max(256, min(self.chunk, int(2e7 // ((k + 1) * (k + 1)))))   # bound the per-chunk k x k workspaces
        for s in range(0, n, chunk):
            o = O[s:s + chunk]
            of = o.astype(float)
            z = Z0[s:s + chunk]
            r = np.where(o, z - m0, 0.0)                                   # centred observed cells
            M = ((of * ipsi) @ LL).reshape(-1, k, k) + np.eye(k)            # I + L_O' Psi_O^-1 L_O
            b = (r * ipsi) @ L                                              # L_O' Psi_O^-1 r_O
            G = np.linalg.inv(M)                                            # posterior covariance of f
            f = np.einsum("nkl,nl->nk", G, b)                               # posterior mean of f
            F[s:s + chunk] = f
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
