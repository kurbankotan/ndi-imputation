# -*- coding: utf-8 -*-
"""Numbers and figure of the million-row experiments used in the manuscript (Section 4.7 (vii)-(viii), Section 5.11,
Tables 18-19, A6-A7, Fig. 10).  Every number of that text is read from the file written here.
Usage: python large/make_large_paper.py RESULTS_DIR [OUT_DIR]   ->  large_paper.json, figure10_large.png"""
import os, sys, json, re
import numpy as np, pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_large")
OUT = sys.argv[2] if len(sys.argv) > 2 else R
fl = lambda v: None if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)) else float(v)
P = {}
ORDER = ["MiniBooNE", "YearPredictionMSD", "covertype", "KDDCup99", "SUSY", "HEPMASS", "HIGGS"]
SETTINGS = ["MCAR-30", "MAR-30", "MNAR-30"]
setting = lambda d: d.mechanism + "-" + (d.rate * 100).round().astype(int).astype(str)

# ------------------------------------------------------------------ ACC: error and time against n
a = pd.read_csv(os.path.join(R, "acc_long.csv")); a = a[a.dataset != "synthetic"].copy(); a["setting"] = setting(a)
nfull = a.groupby("dataset").n.max().to_dict(); p_of = a.groupby("dataset").p.max().to_dict()
info = pd.read_csv(os.path.join(R, "datasets_large.csv")).drop_duplicates("dataset").set_index("dataset") if os.path.exists(os.path.join(R, "datasets_large.csv")) else None
P["datasets"] = [dict(dataset=d, n=int(nfull[d]), p=int(p_of[d]), classes=(int(info.loc[d, "classes"]) if info is not None and d in info.index and str(info.loc[d, "classes"]) not in ("", "nan") else None)) for d in ORDER]
g = a.groupby(["dataset", "n", "setting", "method"], as_index=False).agg(rmse=("rmse", "mean"), t=("time_s", "mean"), k=("k", "mean"), it=("em_iterations", "mean"), seeds=("seed", "nunique"))
top = lambda d: min(1_000_000, nfull[d])
gb = g[g.n.isin([10_000, 50_000, 100_000]) | (g.n == g.dataset.map(top))]
W = gb.pivot_table(index=["dataset", "n", "setting"], columns="method", values="rmse")
M9 = ["Mean", "NDI-S", "NDI-F", "NDI-F-V", "EM-FA@1", "EM-FA@3", "EM-FA", "EM-PPCA", "GaussCM"]
Wc = W[M9].dropna()
rk = Wc.rank(axis=1).mean()
acc = dict(blocks=int(len(Wc)), sizes=[10_000, 50_000, 100_000, "1e6 or full"], methods=M9,
           rank={m: float(rk[m]) for m in M9}, order=sorted(M9, key=lambda m: rk[m]),
           best={m: int(v) for m, v in Wc.idxmin(axis=1).value_counts().items()},
           by_setting={s: {m: float(Wc.xs(s, level="setting")[m].mean()) for m in M9} for s in SETTINGS},
           best_by_setting={s: {m: int(v) for m, v in Wc.xs(s, level="setting").idxmin(axis=1).value_counts().items()} for s in SETTINGS},
           blocks_per_setting=int(len(Wc.xs("MCAR-30", level="setting"))))
pairs = {}
for m1, m2 in [("NDI-F-V", "NDI-F"), ("GaussCM", "NDI-F-V"), ("GaussCM", "NDI-F"), ("EM-FA", "GaussCM"), ("EM-FA@3", "NDI-F"), ("EM-FA@1", "NDI-F"),
               ("EM-FA", "EM-FA@3"), ("NDI-F-V", "EM-FA@3"), ("GaussCM", "EM-FA@3"), ("EM-FA", "NDI-F")]:
    d_ = Wc[m1] - Wc[m2]
    pairs[f"{m1} vs {m2}"] = dict(wins=int((d_ < 0).sum()), losses=int((d_ > 0).sum()), mean_diff=float(d_.mean()), median_diff=float(d_.median()))
acc["pairs"] = pairs
# stability across sizes (MCAR): largest change of the dataset-mean error between sizes
mc = Wc.xs("MCAR-30", level="setting")
size_lab = lambda d, n: "top" if n == top(d) and n not in (10_000, 50_000, 100_000) else n
acc["mcar_by_size"] = {str(n): {m: float(v) for m, v in mc.xs(n, level="n")[M9].mean().items()} for n in [10_000, 50_000, 100_000]}
spread = {}
for m in M9:
    per_ds = mc[m].groupby(level="dataset").agg(lambda x: x.max() - x.min())
    spread[m] = dict(median=float(per_ds.median()), max=float(per_ds.max()))
acc["spread_over_sizes_mcar"] = spread
# SoftImpute where it ran (at most 2.5e7 cells)
Ws = W[["SoftImpute", "NDI-F", "NDI-F-V", "GaussCM", "EM-FA"]].dropna()
acc["softimpute"] = dict(blocks=int(len(Ws)), rank={m: float(v) for m, v in Ws.rank(axis=1).mean().items()},
                         mean={m: float(v) for m, v in Ws.mean().items()})
P["acc"] = acc

# full tables (30 % MCAR): error and time
FULLM = ["Mean", "NDI-S", "NDI-F", "NDI-F-V", "EM-FA@1", "EM-FA@3", "EM-FA", "EM-PPCA", "GaussCM", "SoftImpute"]
full = {}
for d in ORDER:
    x = g[(g.dataset == d) & (g.n == nfull[d]) & (g.setting == "MCAR-30")].set_index("method")
    full[d] = dict(n=int(nfull[d]), p=int(p_of[d]), rmse={m: fl(x.rmse.get(m)) for m in FULLM}, time={m: fl(x.t.get(m)) for m in FULLM},
                   k_ndifv=fl(x.k.get("NDI-F-V")), k_ndif=fl(x.k.get("NDI-F")))
P["full"] = full
P["full_ratio_gauss_ndif"] = {d: full[d]["time"]["GaussCM"] / full[d]["time"]["NDI-F"] for d in ORDER}
# time at 1e6 rows (datasets with at least 1e6 rows), 30 % MCAR
t6 = g[(g.n == 1_000_000) & (g.setting == "MCAR-30")].pivot_table(index="dataset", columns="method", values="t")
P["time_1e6"] = {m: dict(min=fl(t6[m].min()), max=fl(t6[m].max()), median=fl(t6[m].median()), datasets=int(t6[m].notna().sum())) for m in t6.columns}
it6 = g[(g.n == 1_000_000) & g.method.isin(["EM-FA", "EM-PPCA"])].groupby("method").it.agg(["min", "median", "max"])
P["em_iterations_1e6"] = {m: {k: float(v) for k, v in row.items()} for m, row in it6.iterrows()}
# heavy tails (KDD Cup 99): blocks in which a method was worse than mean substitution
k = g[g.dataset == "KDDCup99"].pivot_table(index=["n", "setting"], columns="method", values="rmse")
P["kdd_worse_than_mean"] = {m: dict(blocks=int(k[m].notna().sum()), worse=int((k[m] > k["Mean"]).sum()), max=fl(k[m].max())) for m in k.columns if m != "Mean"}

# ------------------------------------------------------------------ DEEP: 50,000-row samples, seed 0
dd = pd.read_csv(os.path.join(R, "deep_long.csv")); dd = dd[(dd.dataset != "synthetic") & dd.rmse.notna()].copy(); dd["setting"] = setting(dd)
cheap = a[(a.seed == 0) & (a.n == 50_000)]
both = pd.concat([cheap, dd], ignore_index=True)
S = both.pivot_table(index=["dataset", "setting"], columns="method", values="rmse"); ST = both.pivot_table(index=["dataset", "setting"], columns="method", values="time_s")
M12 = ["Mean", "NDI-F", "NDI-F-V", "EM-FA@3", "EM-FA", "GaussCM", "SoftImpute", "GAIN@50000", "MIWAE@200", "Sinkhorn@2000", "HyperImpute", "ReMasker@100"]
X = S[M12].dropna(); rk = X.rank(axis=1).mean()
deep = dict(blocks=int(len(X)), methods=M12, rank={m: float(rk[m]) for m in M12}, order=sorted(M12, key=lambda m: rk[m]),
            best={m: int(v) for m, v in X.idxmin(axis=1).value_counts().items()},
            by_setting={s: {m: float(X.xs(s, level="setting")[m].mean()) for m in M12} for s in SETTINGS},
            median={m: float(X[m].median()) for m in M12}, time={m: float(ST[m].mean()) for m in M12},
            worse_than_mean={m: int((X[m] > X["Mean"]).sum()) for m in M12})
dp = {}
for m1, m2 in [("ReMasker@100", "GaussCM"), ("ReMasker@100", "EM-FA"), ("ReMasker@100", "NDI-F-V"), ("ReMasker@100", "NDI-F"), ("HyperImpute", "GaussCM"),
               ("Sinkhorn@2000", "NDI-F"), ("GAIN@50000", "NDI-F"), ("MIWAE@200", "NDI-F"), ("GAIN@50000", "GAIN@2000"), ("Sinkhorn@2000", "Sinkhorn@200")]:
    d_ = (S[m1] - S[m2]).dropna()
    dp[f"{m1} vs {m2}"] = dict(blocks=int(len(d_)), wins=int((d_ < 0).sum()), losses=int((d_ > 0).sum()), mean_diff=float(d_.mean()), median_diff=float(d_.median()))
deep["pairs"] = dp
deep["remasker_minus_gauss"] = {f"{d}|{s}": float(v) for (d, s), v in (S["ReMasker@100"] - S["GaussCM"]).items()}
deep["per_block"] = {f"{d}|{s}": {m: fl(v) for m, v in row.items()} for (d, s), row in S.iterrows()}
deep["budgets"] = {"GAIN": [2000, 50000], "MIWAE": [200], "Sinkhorn": [200, 2000], "ReMasker": [100]}
deep["miwae_retried"] = int(len(pd.read_csv(os.path.join(R, "deep_long_errors.csv")))) if os.path.exists(os.path.join(R, "deep_long_errors.csv")) else 0
P["deep"] = deep

# ------------------------------------------------------------------ ONLINE: streaming
o = pd.read_csv(os.path.join(R, "online_long.csv")); o = o[o.dataset != "synthetic"]
V = ["Mean frozen", "NDI-F frozen", "NDI-F updated", "NDI-F refit", "NDI-F-V frozen", "EM-FA frozen"]
upd = {}
logf = os.path.join(R, "log_acc_online.txt")
if os.path.exists(logf):
    for line in open(logf, encoding="utf-8", errors="replace"):
        m = re.search(r"ONLINE (\S+)\s+block\s+(\d+)/(\d+) rows_seen=\s*([\d,]+) update ([\d.]+) s", line)
        if m: upd.setdefault(m.group(1), {})[int(m.group(2))] = float(m.group(5))
onl = {}
for d in ORDER:
    x = o[o.dataset == d]
    if x.empty: continue
    w = x.pivot_table(index="block", columns="variant", values="rmse"); t = x.pivot_table(index="block", columns="variant", values="time_s")
    rows_seen = x.groupby("block").rows_seen.first()
    onl[d] = dict(blocks=int(len(w)), block_rows=int(x.block_rows.median()), rmse={v: fl(w[v].mean()) for v in V if v in w},
                  max_abs_updated_minus_refit=fl((w["NDI-F updated"] - w["NDI-F refit"]).abs().max()),
                  update_s=(float(np.mean(list(upd[d].values()))) if d in upd and upd[d] else None),
                  impute_s=fl(t["NDI-F updated"].mean()), refit_first_s=fl(t["NDI-F refit"].iloc[0]), refit_last_s=fl(t["NDI-F refit"].iloc[-1]),
                  rows_seen_last=int(rows_seen.iloc[-1]))
P["online"] = onl
P["online_used"] = [d for d in ORDER if d in onl and onl[d]["blocks"] >= 5]

# ------------------------------------------------------------------ DOWN
w = pd.read_csv(os.path.join(R, "downstream_long.csv")); w = w[w.dataset != "synthetic"].copy(); w["setting"] = setting(w)
D = w.pivot_table(index=["dataset", "setting"], columns="method", values="accuracy")
imp = [c for c in D.columns if c != "complete"]
sp = D[imp].max(axis=1) - D[imp].min(axis=1); gap = D["complete"] - D[imp].max(axis=1)
P["down"] = dict(blocks=int(len(D)), spread_mean=float(sp.mean()), spread_max=float(sp.max()), gap_mean=float(gap.mean()),
                 best={m: int(v) for m, v in D[imp].idxmax(axis=1).value_counts().items()},
                 rank={m: float(v) for m, v in D[imp].rank(axis=1, ascending=False).mean().items()}, methods=imp)

env = os.path.join(R, "env_large.json")
if os.path.exists(env): P["env"] = json.load(open(env))
json.dump(P, open(os.path.join(OUT, "large_paper.json"), "w"), indent=1)

# ------------------------------------------------------------------ Fig. 10 (style of make_figures.py: 600 dpi, no titles)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
C = {"NDI-F": "#2a78d6", "NDI-F-V": "#7a3fc4", "NDI-S": "#52514e", "Mean": "#1baf7a", "MICE": "#e87ba4", "GaussCM": "#008300",
     "SoftImpute": "#e34948", "EM-FA": "#138d90", "EM-PPCA": "#898781", "EM-FA@1": "#7cc4c6", "EM-FA@3": "#44a7aa"}
MK = {"NDI-F": "o", "NDI-F-V": "D", "NDI-S": "h", "Mean": "^", "MICE": "v", "GaussCM": "P", "SoftImpute": "*", "EM-FA": "d", "EM-PPCA": "p", "EM-FA@1": "x", "EM-FA@3": "s"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.0, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "savefig.dpi": 600, "savefig.facecolor": "white"})
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
ax = axes[0]; d0 = "HIGGS"
ends = []                                   # direct labels at the line ends, as in Fig. 6 (pushed apart in log space)
for m in ["Mean", "NDI-S", "NDI-F", "NDI-F-V", "EM-FA@1", "EM-FA@3", "GaussCM", "EM-PPCA", "EM-FA", "SoftImpute", "MICE"]:
    x = g[(g.dataset == d0) & (g.setting == "MCAR-30") & (g.method == m)].sort_values("n")
    if len(x):
        ax.plot(x.n, x.t, "-", color=C[m], marker=MK[m], ms=3.4, lw=1.2)
        ends.append([m, float(x.n.iloc[-1]), float(np.log10(x.t.iloc[-1]))])
for xe in sorted(set(e[1] for e in ends)):
    grp = sorted([e for e in ends if e[1] == xe], key=lambda e: e[2]); gap = 0.15
    ys = [e[2] for e in grp]
    for _ in range(50):                      # simple repel: keep at least `gap` decades between neighbours
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gap:
                mid = (ys[i] + ys[i - 1]) / 2; ys[i - 1], ys[i] = mid - gap / 2, mid + gap / 2
    for e, y in zip(grp, ys):
        fx, fy = {"SoftImpute": (1.15, 1.55)}.get(e[0], (1.35, 1.0))          # SoftImpute ends inside the bundle: label above it
        ax.annotate(e[0], xy=(e[1], 10 ** e[2]), xytext=(e[1] * fx, 10 ** y * fy), fontsize=6.0, color=C[e[0]], va="center",
                    arrowprops=None if abs(y - e[2]) < 0.03 else dict(arrowstyle="-", color=C[e[0]], lw=0.5, alpha=0.6))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(6e3, 2.5e8); ax.set_xticks([1e4, 1e5, 1e6, 1e7])
ax.set_xlabel(f"Rows $n$ ({d0}, 30 % MCAR)"); ax.set_ylabel("Wall-clock seconds (log scale)")
ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", fontweight="bold", color=INK)
# streaming: running mean of the block errors over the blocks imputed so far (prequential)
for ax, d1, lab in [(axes[1], "HIGGS", "(b) HIGGS"), (axes[2], "KDDCup99", "(c) KDD Cup 99")]:
    x = o[o.dataset == d1]
    for v, c, ls, lw in [("Mean frozen", C["Mean"], "-", 1.1), ("NDI-F frozen", C["NDI-F"], "--", 1.2), ("NDI-F updated", C["NDI-F"], "-", 1.6),
                         ("NDI-F-V frozen", C["NDI-F-V"], "--", 1.1), ("EM-FA frozen", C["EM-FA"], ":", 1.4)]:
        xx = x[x.variant == v].sort_values("block")
        if len(xx): ax.plot(xx.rows_seen / 1e6, xx.rmse.expanding().mean(), ls, color=c, lw=lw, label=v.replace("updated", "updated (= refit)"))
    ax.set_xlabel("Rows seen (millions)")
    top_ = d1 == "HIGGS"
    ax.text(0.03, 0.97 if top_ else 0.03, lab, transform=ax.transAxes, ha="left", va="top" if top_ else "bottom", fontweight="bold", color=INK)
axes[1].set_ylabel("Running mean of the block z-RMSE")
emk = o[(o.dataset == "KDDCup99") & (o.variant == "EM-FA frozen")].rmse
axes[2].set_ylim(0, 2.2)
axes[2].annotate(f"EM-FA frozen: off scale (mean {emk.mean():.1f})", xy=(0.03, 0.12), xycoords="axes fraction", ha="left", va="bottom", fontsize=6.0, color=C["EM-FA"])
lo1 = o[(o.dataset == "HIGGS")].groupby("variant").rmse.mean()
axes[1].set_ylim(min(lo1.min() - 0.02, 0.85), 1.03)
axes[1].legend(fontsize=6.0, loc="center right", borderaxespad=0.2)
fig.tight_layout(w_pad=0.8); fig.savefig(os.path.join(OUT, "figure10_large.png")); plt.close(fig)
print("written:", os.path.join(OUT, "large_paper.json"), os.path.join(OUT, "figure10_large.png"))
