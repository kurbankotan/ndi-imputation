# -*- coding: utf-8 -*-
"""Figures of the additional experiments, in the style of make_figures.py (600 dpi, no in-figure titles):
figure7_scaling_p.png, figure8_em_iterations.png, figure9_small_n.png (numbered as in the manuscript).  Usage: python make_extra_figures.py RESULTS_DIR FIG_DIR"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results_extra")
FIG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures")
C = {"NDI-F": "#2a78d6", "NDI-F-V": "#7a3fc4", "NDI-0": "#eb6834", "Mean": "#1baf7a", "MissForest": "#eda100", "MICE": "#e87ba4",
     "GaussCM": "#008300", "SoftImpute": "#e34948", "NDI-S": "#52514e", "KNN": "#898781", "EM-FA": "#138d90", "EM-PPCA": "#898781"}
MK = {"NDI-F": "o", "NDI-F-V": "D", "NDI-S": "h", "NDI-0": "s", "Mean": "^", "MissForest": "D", "MICE": "v", "GaussCM": "P", "SoftImpute": "*", "KNN": "<", "EM-FA": "d", "EM-PPCA": "p"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "savefig.dpi": 600, "savefig.facecolor": "white"})
setting = lambda d: d.mechanism + "-" + (d.rate * 100).round().astype(int).astype(str)

# ------------------------------------------------------------------ Fig 7: EM from the NDI-F start, stopped early
f = os.path.join(OUT, "em_iterations_long.csv")
if os.path.exists(f):
    d = pd.read_csv(f); d["setting"] = setting(d)
    a = d.groupby(["dataset", "setting", "variant"], as_index=False).agg(rmse=("rmse", "mean"), time_s=("time_s", "mean"), n_iter=("n_iter", "mean"))
    W = a.pivot(index=["dataset", "setting"], columns="variant", values="rmse"); TW = a.pivot(index=["dataset", "setting"], columns="variant", values="time_s")
    caps = [0, 1, 2, 3, 5, 10, 20, 50]; names = ["NDI-F-noclip"] + [f"EM-FA@{c}" for c in caps[1:]]; conv = "EM-FA@conv:ndif"
    med_it = d[d.variant == conv].n_iter.median()                       # per run, as in the text
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    ax = axes[0]
    y = [W[v].mean() for v in names]
    ax.plot(caps, y, "-", color=C["EM-FA"], lw=1.6, marker=MK["EM-FA"], ms=5, label="EM-FA from the NDI-F start, $m$ iterations", zorder=3)
    ax.axhline(W["NDI-F"].mean(), color=C["NDI-F"], ls="--", lw=1.2, label="NDI-F (one pass)")
    ax.axhline(W[conv].mean(), color=MUTED, ls=":", lw=1.3, label=f"EM-FA converged (median {med_it:.0f} iterations)")
    ax.set_xscale("symlog", linthresh=1); ax.set_xticks(caps); ax.set_xticklabels([str(c) for c in caps]); ax.set_xlim(-0.3, 70)
    ax.set_xlabel("EM iterations $m$ after the one-pass start"); ax.set_ylabel(f"z-RMSE, mean over {len(W)} blocks")
    ax.legend(fontsize=6.8, loc="center right"); ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, va="top", fontweight="bold", color=INK)
    ax = axes[1]
    t = [TW[v].mean() / TW["NDI-F"].mean() for v in names]
    ax.plot(t, y, "-", color=C["EM-FA"], lw=1.6, marker=MK["EM-FA"], ms=5, zorder=3)
    for c, xx, yy in zip(caps, t, y):
        if c in (1, 3, 10): ax.annotate(f"$m$ = {c}", (xx, yy), textcoords="offset points", xytext=(5, 4), fontsize=7, color=INK2)
        if c == 50: ax.annotate(f"$m$ = {c}", (xx, yy), textcoords="offset points", xytext=(-4, -12), fontsize=7, color=INK2)
    ax.annotate("start ($m$ = 0)", (t[0], y[0]), textcoords="offset points", xytext=(7, -2), fontsize=7, color=INK2)
    ax.scatter([TW[conv].mean() / TW["NDI-F"].mean()], [W[conv].mean()], s=40, color=MUTED, marker="s", zorder=4)
    ax.annotate("converged", (TW[conv].mean() / TW["NDI-F"].mean(), W[conv].mean()), textcoords="offset points", xytext=(-20, 7), fontsize=7, color=INK2)
    ax.scatter([1], [W["NDI-F"].mean()], s=46, color=C["NDI-F"], marker="o", zorder=4)
    ax.annotate("NDI-F", (1, W["NDI-F"].mean()), textcoords="offset points", xytext=(5, -10), fontsize=7, color=C["NDI-F"])
    ax.set_xscale("log"); ax.set_xlabel("Time relative to NDI-F (log scale)"); ax.set_ylabel("z-RMSE")
    for a_ in axes: a_.set_ylim(top=max(y) + 0.005)
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, va="top", fontweight="bold", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figure8_em_iterations.png")); plt.close(fig); print("figure8 (EM iterations) written")

# ------------------------------------------------------------------ Fig 8: time against p (synthetic)
f = os.path.join(OUT, "scaling_p_long.csv")
if os.path.exists(f):
    d = pd.read_csv(f); d = d[d.patterns == "mcar"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
    for ax, n, lab in zip(axes, [10_000, 100_000], ["(a)", "(b)"]):
        g = d[d.n == n]
        for m in ["Mean", "NDI-S", "NDI-F", "NDI-F-V", "EM-FA", "EM-PPCA", "GaussCM", "SoftImpute"]:
            gg = g[(g.method == m) & g.time_s.notna()].sort_values("p")
            if len(gg): ax.plot(gg.p, gg.time_s, "-", color=C.get(m, MUTED), marker=MK.get(m, "o"), ms=5, lw=1.5, label=m)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([50, 100, 200, 500]); ax.set_xticklabels(["50", "100", "200", "500"]); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(f"Columns $p$ ($n$ = {n:,}, 30 % MCAR)"); ax.text(0.02, 0.98, lab, transform=ax.transAxes, va="top", fontweight="bold", color=INK)
    axes[0].set_ylabel("Wall-clock seconds (log scale)"); axes[1].legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figure7_scaling_p.png")); plt.close(fig); print("figure7 (scaling with p) written")

# ------------------------------------------------------------------ Fig 9: the small-sample regime
f = os.path.join(OUT, "small_n_long.csv")
if os.path.exists(f):
    d = pd.read_csv(f)
    W = d.groupby(["dataset", "n_sub", "method"]).rmse.mean().unstack()
    m_by_n = W.groupby(level="n_sub").mean()
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for m in ["Mean", "NDI-F", "NDI-F-V", "GaussCM", "EM-FA", "SoftImpute", "MissForest", "MICE"]:
        if m in m_by_n: ax.plot(m_by_n.index, m_by_n[m], "-", color=C.get(m, MUTED), marker=MK.get(m, "o"), ms=5, lw=1.5, label=m)
    ax.set_xscale("log"); ax.set_xticks([50, 100, 200, 500]); ax.set_xticklabels(["50", "100", "200", "500"]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Rows $n$ (sub-sampled; 30 % MCAR)"); ax.set_ylabel(f"z-RMSE, mean over {W.index.get_level_values(0).nunique()} datasets")
    ax.set_ylim(top=min(1.6, float(m_by_n.max().max()) + 0.05)); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figure9_small_n.png")); plt.close(fig); print("figure9 written")
