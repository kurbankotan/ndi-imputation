# -*- coding: utf-8 -*-
"""Summaries and figures of the million-row experiments.  Usage: python large/make_large_tables.py [RESULTS_DIR] [FIG_DIR]
Writes summary_acc.csv, summary_time.csv, summary_deep.csv, summary_online.csv, summary_downstream.csv, large_tables.json and
figL1_error_vs_n.png, figL2_time_vs_n.png, figL3_online.png."""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results_large")
FIG = sys.argv[2] if len(sys.argv) > 2 else OUT
f32 = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
T = {}
C = {"NDI-F": "#2a78d6", "NDI-F-V": "#7a3fc4", "NDI-S": "#52514e", "Mean": "#1baf7a", "MissForest": "#eda100", "MICE": "#e87ba4", "GaussCM": "#008300",
     "SoftImpute": "#e34948", "KNN": "#898781", "EM-FA": "#138d90", "EM-PPCA": "#898781", "EM-FA@1": "#5fb3b5", "EM-FA@3": "#2f9ea1",
     "GAIN": "#4b4bd6", "MIWAE": "#b04ab0", "Sinkhorn": "#c76a1f", "ReMasker": "#2f6b2f", "HyperImpute": "#9b2d2d"}
MK = {"NDI-F": "o", "NDI-F-V": "D", "NDI-S": "h", "Mean": "^", "MissForest": "D", "MICE": "v", "GaussCM": "P", "SoftImpute": "*", "KNN": "<", "EM-FA": "d",
      "EM-PPCA": "p", "EM-FA@1": ".", "EM-FA@3": "s"}

acc = os.path.join(OUT, "acc_long.csv")
if os.path.exists(acc):
    d = pd.read_csv(acc); d["setting"] = d.mechanism + "-" + (d.rate * 100).round().astype(int).astype(str)
    if (d.dataset != "synthetic").any(): d = d[d.dataset != "synthetic"]            # smoke-test rows, if any
    g = d.groupby(["dataset", "p", "n", "setting", "method"], as_index=False).agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), time_s=("time_s", "mean"),
                                                                                  seeds=("seed", "nunique"), k=("k", "mean"), em_iterations=("em_iterations", "mean"), error=("error", "first"))
    g.to_csv(os.path.join(OUT, "summary_acc.csv"), index=False)
    W = g.pivot_table(index=["dataset", "n", "setting"], columns="method", values="rmse")
    TW = g.pivot_table(index=["dataset", "n", "setting"], columns="method", values="time_s")
    W.round(4).to_csv(os.path.join(OUT, "summary_rmse_wide.csv")); TW.round(3).to_csv(os.path.join(OUT, "summary_time.csv"))
    T["acc"] = {"|".join(map(str, idx)): {m: f32(v) for m, v in row.items()} for idx, row in W.iterrows()}
    T["time"] = {"|".join(map(str, idx)): {m: f32(v) for m, v in row.items()} for idx, row in TW.iterrows()}
    # paired comparisons at every size (MCAR-30)
    pairs = {}
    for m1, m2 in [("NDI-F-V", "NDI-F"), ("EM-FA@3", "NDI-F"), ("EM-FA", "NDI-F"), ("GaussCM", "NDI-F"), ("EM-FA", "NDI-F-V"), ("GaussCM", "NDI-F-V"), ("SoftImpute", "NDI-F-V"), ("MICE", "NDI-F-V")]:
        if m1 in W and m2 in W:
            ok = W[[m1, m2]].dropna(); diff = ok[m1] - ok[m2]
            pairs[f"{m1} vs {m2}"] = dict(blocks=int(len(ok)), wins=int((diff < 0).sum()), losses=int((diff > 0).sum()), mean_diff=float(diff.mean()))
    T["pairs"] = pairs
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "axes.grid": True, "grid.color": "#e1e0d9", "savefig.dpi": 300})
    ds = list(dict.fromkeys(g.dataset)); nd = len(ds)
    for kind, col, ylab, fn in [("rmse", "rmse", "z-RMSE (30 % MCAR)", "figL1_error_vs_n.png"), ("time", "time_s", "Wall-clock seconds (log scale)", "figL2_time_vs_n.png")]:
        fig, axes = plt.subplots(1, nd, figsize=(2.6 * nd, 2.8), sharey=(kind == "time"), squeeze=False); axes = axes[0]
        for ax, name in zip(axes, ds):
            gg = g[(g.dataset == name) & (g.setting == "MCAR-30")]
            for m in ["Mean", "NDI-S", "NDI-F", "NDI-F-V", "EM-FA@3", "EM-FA", "EM-PPCA", "GaussCM", "SoftImpute", "MICE", "MissForest", "KNN"]:
                x = gg[gg.method == m].sort_values("n")
                if len(x): ax.plot(x.n, x[col], "-", color=C.get(m, "#888"), marker=MK.get(m, "o"), ms=4, lw=1.3, label=m)
            ax.set_xscale("log"); ax.set_title(f"{name} (p = {int(gg.p.iloc[0]) if len(gg) else '?'})", fontsize=8.5); ax.set_xlabel("rows n")
            if kind == "time": ax.set_yscale("log")
            else: ax.set_ylim(top=min(ax.get_ylim()[1], 1.35))                 # diverged imputations (MICE on KDD) would flatten the panel
        axes[0].set_ylabel(ylab); axes[-1].legend(fontsize=6.5, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, fn)); plt.close(fig)

deep = os.path.join(OUT, "deep_long.csv")
if os.path.exists(deep) and os.path.exists(acc):
    dd = pd.read_csv(deep); dd["setting"] = dd.mechanism + "-" + (dd.rate * 100).round().astype(int).astype(str)
    dd = dd[dd.rmse.notna()]
    if (dd.dataset != "synthetic").any(): dd = dd[dd.dataset != "synthetic"]
    cheap = d[(d.seed == 0) & d.n.isin(dd.n.unique())]
    both = pd.concat([cheap, dd], ignore_index=True)
    S = both.pivot_table(index=["dataset", "n", "setting"], columns="method", values="rmse"); ST = both.pivot_table(index=["dataset", "n", "setting"], columns="method", values="time_s")
    S.round(4).to_csv(os.path.join(OUT, "summary_deep.csv")); ST.round(2).to_csv(os.path.join(OUT, "summary_deep_time.csv"))
    T["deep"] = {"|".join(map(str, idx)): {m: f32(v) for m, v in row.items()} for idx, row in S.iterrows()}
    T["deep_time"] = {"|".join(map(str, idx)): {m: f32(v) for m, v in row.items()} for idx, row in ST.iterrows()}

onl = os.path.join(OUT, "online_long.csv")
if os.path.exists(onl):
    o = pd.read_csv(onl)
    if (o.dataset != "synthetic").any(): o = o[o.dataset != "synthetic"]
    s = o.groupby(["dataset", "variant"], as_index=False).agg(rmse=("rmse", "mean"), time_per_block_s=("time_s", "mean"), blocks=("block", "nunique"))
    s.to_csv(os.path.join(OUT, "summary_online.csv"), index=False)
    T["online"] = {ds: {r.variant: dict(rmse=f32(r.rmse), time_per_block_s=f32(r.time_per_block_s), blocks=int(r.blocks)) for r in gg.itertuples()} for ds, gg in s.groupby("dataset")}
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ds = list(dict.fromkeys(o.dataset)); nd = len(ds)
    fig, axes = plt.subplots(1, nd, figsize=(2.6 * nd, 2.8), squeeze=False); axes = axes[0]
    for ax, name in zip(axes, ds):
        gg = o[o.dataset == name]
        for v, c, ls in [("Mean frozen", C["Mean"], "-"), ("NDI-F frozen", C["NDI-F"], "--"), ("NDI-F updated", C["NDI-F"], "-"), ("NDI-F refit", "#000", ":"), ("NDI-F-V frozen", C["NDI-F-V"], "--"), ("EM-FA frozen", C["EM-FA"], "--")]:
            x = gg[gg.variant == v].sort_values("block")
            if len(x): ax.plot(x.rows_seen, x.rmse, ls, color=c, lw=1.3, label=v)
        ax.set_title(name, fontsize=8.5); ax.set_xlabel("rows seen before the block")
    axes[0].set_ylabel("z-RMSE of the block (30 % MCAR)"); axes[-1].legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figL3_online.png")); plt.close(fig)

down = os.path.join(OUT, "downstream_long.csv")
if os.path.exists(down):
    w = pd.read_csv(down); w["setting"] = w.mechanism + "-" + (w.rate * 100).round().astype(int).astype(str)
    if (w.dataset != "synthetic").any(): w = w[w.dataset != "synthetic"]
    D = w.pivot_table(index=["dataset", "setting"], columns="method", values="accuracy"); D.round(4).to_csv(os.path.join(OUT, "summary_downstream.csv"))
    T["downstream"] = {"|".join(map(str, idx)): {m: f32(v) for m, v in row.items()} for idx, row in D.iterrows()}

env = os.path.join(OUT, "env_large.json")
if os.path.exists(env): T["env"] = json.load(open(env))
json.dump(T, open(os.path.join(OUT, "large_tables.json"), "w"), indent=1)
print("sections:", list(T.keys()))
