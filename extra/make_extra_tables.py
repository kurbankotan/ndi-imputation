# -*- coding: utf-8 -*-
"""Summaries of the additional experiments (results_extra/*.csv) -> extra_tables.json (read by the manuscript generator)."""
import os, re, json, sys
import numpy as np, pandas as pd
from scipy import stats
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_extra")
T = {}
f32 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)

def setting_col(d):
    d["setting"] = d.mechanism + "-" + (d.rate * 100).round().astype(int).astype(str); return d

def ranks(block, cols):
    r = block[cols].rank(axis=1); return {c: float(r[c].mean()) for c in cols}

def nemenyi_cd(k, N, alpha=0.05):
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6 * N)))

# ----------------------------------------------------------------------------- machine ratio
tc = os.path.join(OUT, "timing_check_extra.txt")
if os.path.exists(tc):
    t = [float(x) for x in re.findall(r"([0-9.]+) s", open(tc).readline())[:2]]
    T["timing_check"] = dict(this_machine_s=t[0], paper_machine_s=t[1], ratio=t[0] / t[1])
tp = os.path.join(OUT, "timing_paths.json")                 # same reference computation with the paper's code path (pandas correlation), idle machine
if os.path.exists(tp): T.setdefault("timing_check", {}).update(json.load(open(tp)))
env = os.path.join(OUT, "env_extra.json")
if os.path.exists(env): T["env"] = json.load(open(env))

# ----------------------------------------------------------------------------- EM: early stopping and starts
f = os.path.join(OUT, "em_iterations_long.csv")
if os.path.exists(f):
    d = setting_col(pd.read_csv(f))
    a = d.groupby(["dataset", "setting", "variant"], as_index=False).agg(rmse=("rmse", "mean"), time_s=("time_s", "mean"), n_iter=("n_iter", "mean"), loglik=("loglik", "mean"))
    W = a.pivot(index=["dataset", "setting"], columns="variant", values="rmse")
    TW = a.pivot(index=["dataset", "setting"], columns="variant", values="time_s")
    IT = a.pivot(index=["dataset", "setting"], columns="variant", values="n_iter")
    LL = a.pivot(index=["dataset", "setting"], columns="variant", values="loglik")
    caps = [1, 2, 3, 5, 10, 20, 50]
    capv = [f"EM-FA@{c}" for c in caps]; conv = "EM-FA@conv:ndif"
    gain = W["NDI-F-noclip"] - W[conv]
    em = dict(n_blocks=int(len(W)), n_datasets=int(d.dataset.nunique()), datasets=sorted(set(d.dataset)), caps=caps, variants={})
    for v in ["NDI-F", "NDI-F-noclip"] + capv + [conv, "EM-FA@conv:pca", "EM-FA@conv:random"]:
        share = ((W["NDI-F-noclip"] - W[v]) / gain)[gain > 0]
        runs_v = d[d.variant == v].n_iter                      # iterations: per run (as in the main text), not per seed-averaged block
        em["variants"][v] = dict(rmse=float(W[v].mean()), time_s=float(TW[v].mean()), time_rel_ndif=float(TW[v].mean() / TW["NDI-F"].mean()),
                                 iters_median=float(runs_v.median()), iters_q90=float(runs_v.quantile(.9)), iters_max=float(runs_v.max()),
                                 gain_share_median=float(share.median()), gain_share_mean=float(share.clip(-2, 2).mean()),
                                 better_than_ndif=int((W[v] < W["NDI-F"]).sum()), better_than_conv=int((W[v] < W[conv]).sum()),
                                 worse_than_start=int((W[v] > W["NDI-F-noclip"]).sum()),
                                 by_setting={s: float(g[v].mean()) for s, g in W.groupby(level="setting")})
    em["gain_positive_blocks"] = int((gain > 0).sum())
    em["conv_worse_than_ndif_blocks"] = int((W[conv] > W["NDI-F"]).sum())
    em["best_cap"] = min(capv, key=lambda v: W[v].mean())
    em["conv_minus_best_cap"] = float(W[conv].mean() - W[em["best_cap"]].mean())
    # iterations needed per block to reach a share of the converged gain (from the cap grid; convergence if never)
    reach = {}
    for frac in (0.5, 0.8, 0.9):
        need = []
        for idx, row in W.iterrows():
            g = row["NDI-F-noclip"] - row[conv]
            if not g > 0: continue
            for c, v in zip(caps, capv):
                if (row["NDI-F-noclip"] - row[v]) / g >= frac: need.append(c); break
            else: need.append(int(IT.loc[idx, conv]))
        reach[str(frac)] = dict(median=float(np.median(need)), q75=float(np.percentile(need, 75)), q90=float(np.percentile(need, 90)))
    em["iterations_to_reach"] = reach
    for s in ["pca", "random"]:
        g = LL[conv] - LL[f"EM-FA@conv:{s}"]
        em[f"loglik_ndif_minus_{s}"] = dict(mean=float(g.mean()), min=float(g.min()), max=float(g.max()), ndif_higher_blocks=int((g > 0).sum()))
    # the three starts, per run: log-likelihood gap per observed cell, error gap, iterations
    cells = (d.n * d.p * (1 - d.rate)).groupby([d.dataset, d.setting, d.seed]).first()
    r = d.pivot_table(index=["dataset", "setting", "seed"], columns="variant", values=["loglik", "rmse", "n_iter"])
    starts = {}
    for s in ["pca", "random"]:
        dll = (r[("loglik", conv)] - r[("loglik", f"EM-FA@conv:{s}")]) / cells.reindex(r.index)
        drm = r[("rmse", conv)] - r[("rmse", f"EM-FA@conv:{s}")]
        starts[s] = dict(ll_per_cell_abs_max=float(dll.abs().max()), ll_per_cell_mean=float(dll.mean()), rmse_diff_mean=float(drm.mean()),
                         rmse_diff_abs_max=float(drm.abs().max()), ndif_lower_error_runs=int((drm < 0).sum()), runs=int(len(drm)),
                         iters_median=float(r[("n_iter", f"EM-FA@conv:{s}")].median()))
    starts["ndif"] = dict(iters_median=float(r[("n_iter", conv)].median()))
    em["starts"] = starts
    em["conv_worse_than_ndif_datasets"] = sorted(set(i[0] for i in W.index[W[conv] > W["NDI-F"]]))
    em["cap10_worse_than_ndif_blocks"] = int((W["EM-FA@10"] > W["NDI-F"]).sum())
    em["per_dataset"] = {ds: {v: float(g[v].mean()) for v in ["NDI-F", "NDI-F-noclip", "EM-FA@1", "EM-FA@3", "EM-FA@10", conv]} for ds, g in W.groupby(level="dataset")}
    T["em"] = em

# ----------------------------------------------------------------------------- K sensitivity
f = os.path.join(OUT, "k_sensitivity_long.csv")
if os.path.exists(f):
    d = setting_col(pd.read_csv(f))
    a = d.groupby(["dataset", "setting", "variant"], as_index=False).agg(rmse=("rmse", "mean"), k_used=("k_used", "mean"), time_s=("time_s", "mean"))
    W = a.pivot(index=["dataset", "setting"], columns="variant", values="rmse")
    KU = a.pivot(index=["dataset", "setting"], columns="variant", values="k_used")
    TW = a.pivot(index=["dataset", "setting"], columns="variant", values="time_s")
    order = [v for v in ["k=1", "k=2", "k=3", "k=5", "k=Kaiser", "k=10", "k=20", "k=Kaiser,max_k=p-1", "floor=0.01", "floor=0.1", "floor=0.2", "k=validated"] if v in W]
    rk = ranks(W.reset_index(), order)
    ks = dict(n_blocks=int(len(W)), n_datasets=int(d.dataset.nunique()), datasets=sorted(set(d.dataset)), order=order, variants={})
    for v in order:
        ks["variants"][v] = dict(rmse=float(W[v].mean()), rank=rk[v], k_used_mean=float(KU[v].mean()), time_s=float(TW[v].mean()),
                                 by_setting={s: float(g[v].mean()) for s, g in W.groupby(level="setting")},
                                 better_than_kaiser=int((W[v] < W["k=Kaiser"]).sum()), worse_than_kaiser=int((W[v] > W["k=Kaiser"]).sum()))
    fixed = [v for v in ["k=1", "k=2", "k=3", "k=5", "k=10", "k=20"] if v in W]
    best_fixed = W[fixed].min(axis=1)
    ks["oracle_fixed_k"] = dict(rmse=float(best_fixed.mean()), kaiser_minus_oracle=float((W["k=Kaiser"] - best_fixed).mean()),
                                validated_minus_oracle=float((W["k=validated"] - best_fixed).mean()) if "k=validated" in W else None)
    ks["best_fixed_k_by_dataset"] = {ds: fixed[int(np.argmin([g[v].mean() for v in fixed]))] for ds, g in W.groupby(level="dataset")}
    ks["kaiser_k_by_dataset"] = {ds: float(g["k=Kaiser"].mean()) for ds, g in KU.groupby(level="dataset")}
    if "k=validated" in W:
        ks["validated_k_by_dataset"] = {ds: float(g["k=validated"].mean()) for ds, g in KU.groupby(level="dataset")}
        ks["validated_vs_kaiser"] = dict(better=int((W["k=validated"] < W["k=Kaiser"]).sum()), worse=int((W["k=validated"] > W["k=Kaiser"]).sum()),
                                         wilcoxon_p=float(stats.wilcoxon(W["k=validated"], W["k=Kaiser"]).pvalue),
                                         time_ratio=float(TW["k=validated"].mean() / TW["k=Kaiser"].mean()))
    T["k"] = ks

# ----------------------------------------------------------------------------- RATE: 70 / 80 %
f = os.path.join(OUT, "high_rate_long.csv")
if os.path.exists(f):
    d = pd.read_csv(f)
    hr = dict(datasets=sorted(set(d.dataset)), n_datasets=int(d.dataset.nunique()), by_rate={})
    for rate, g in d.groupby("rate"):
        a = g.groupby(["dataset", "method"], as_index=False).agg(rmse=("rmse", "mean"), time_s=("time_s", "mean"))
        W = a.pivot(index="dataset", columns="method", values="rmse")
        methods = [m for m in ["Mean", "NDI-0", "NDI-S", "NDI-F", "NDI-F-V", "GaussCM", "EM-FA", "EM-PPCA", "KNN", "MICE", "MissForest", "SoftImpute"] if m in W]
        ok = W.dropna(); rk = ranks(ok.reset_index(), methods)
        key = f"{rate:.1f}"
        hr["by_rate"][key] = dict(n_datasets=int(len(ok)), methods={m: dict(rmse=float(W[m].mean()), rank=rk[m], better_than_ndif=int((W[m] < W["NDI-F"]).sum()),
                                  time_s=float(a[a.method == m].time_s.mean())) for m in methods},
                                  order=sorted(methods, key=lambda m: rk[m]), CD=nemenyi_cd(len(methods), len(ok)),
                                  friedman_p=float(stats.friedmanchisquare(*[ok[m].to_numpy() for m in methods]).pvalue) if len(ok) >= 3 else None)
    T["rate"] = hr

# ----------------------------------------------------------------------------- SMALLN
f = os.path.join(OUT, "small_n_long.csv")
if os.path.exists(f):
    d = pd.read_csv(f)
    sn = dict(datasets=sorted(set(d.dataset)), n_datasets=int(d.dataset.nunique()), p_by_dataset={ds: int(p) for ds, p in d.groupby("dataset").p.max().items()}, by_n={})
    for n_sub, g in d.groupby("n_sub"):
        a = g.groupby(["dataset", "method"], as_index=False).agg(rmse=("rmse", "mean"), time_s=("time_s", "mean"))
        W = a.pivot(index="dataset", columns="method", values="rmse")
        methods = [m for m in ["Mean", "NDI-S", "NDI-F", "NDI-F-V", "GaussCM", "EM-FA", "EM-PPCA", "KNN", "MICE", "MissForest", "SoftImpute"] if m in W]
        ok = W.dropna(); rk = ranks(ok.reset_index(), methods)
        sn["by_n"][str(n_sub)] = dict(n_datasets=int(len(ok)), methods={m: dict(rmse=float(W[m].mean()), rank=rk[m], better_than_ndif=int((W[m] < W["NDI-F"]).sum()),
                                      worse_than_mean=int((W[m] > W["Mean"]).sum()), time_s=float(a[a.method == m].time_s.mean())) for m in methods},
                                      order=sorted(methods, key=lambda m: rk[m]))
    T["smalln"] = sn

# ----------------------------------------------------------------------------- SCALEP
f = os.path.join(OUT, "scaling_p_long.csv")
if os.path.exists(f):
    d = pd.read_csv(f)
    T["scalep"] = [dict(n=int(r.n), p=int(r.p), patterns=r.patterns, n_patterns=int(r.n_patterns), method=r.method, time_s=f32(r.time_s), rmse=f32(r.rmse),
                        status=r.status, em_iterations=f32(getattr(r, "em_iterations", np.nan))) for r in d.itertuples()]
    sc = {(r["n"], r["p"], r["patterns"], r["method"]): r for r in T["scalep"]}
    T["scalep_lookup_note"] = "key: n, p, patterns, method"

# ----------------------------------------------------------------------------- LARGE
f = os.path.join(OUT, "large_long.csv")
if os.path.exists(f):
    d = setting_col(pd.read_csv(f))
    a = d.groupby(["dataset", "n", "p", "setting", "method"], as_index=False).agg(rmse=("rmse", "mean"), time_s=("time_s", "mean"), error=("error", "first"), em_iterations=("em_iterations", "mean"))
    lg = dict(datasets={}, n_datasets=int(d.dataset.nunique()))
    for (ds, n, p), g in a.groupby(["dataset", "n", "p"]):
        lg["datasets"][ds] = dict(n=int(n), p=int(p), settings={s: {m: dict(rmse=f32(r.rmse), time_s=f32(r.time_s), error=(None if pd.isna(r.error) else str(r.error)), em_iterations=f32(r.em_iterations))
                                                                   for m, r in gg.set_index("method").iterrows()} for s, gg in g.groupby("setting")})
    W = a.pivot(index=["dataset", "setting"], columns="method", values="rmse")
    always = [m for m in ["Mean", "NDI-0", "NDI-S", "NDI-F", "NDI-F-k20", "NDI-F-kall", "NDI-F-V", "GaussCM", "EM-FA", "EM-FA-k20", "EM-PPCA", "SoftImpute", "KNN", "MICE", "MissForest"] if m in W and W[m].notna().all()]
    rk = ranks(W.reset_index(), always)
    lg["always"] = dict(methods=always, n_blocks=int(len(W)), ranks=rk, rmse={m: float(W[m].mean()) for m in always},
                        better_than_ndif={m: int((W[m] < W["NDI-F"]).sum()) for m in always}, better_than_gauss={m: int((W[m] < W["GaussCM"]).sum()) for m in always} if "GaussCM" in always else None,
                        order=sorted(always, key=lambda m: rk[m]))
    # paired comparisons over the blocks in which both methods ran
    pairs = {}
    for m1, m2 in [("NDI-F-V", "NDI-F"), ("NDI-F-V", "GaussCM"), ("NDI-F-k20", "NDI-F"), ("NDI-F-kall", "NDI-F"), ("EM-FA", "NDI-F"), ("GaussCM", "NDI-F"),
                   ("EM-FA-k20", "EM-FA"), ("SoftImpute", "NDI-F-V"), ("MICE", "NDI-F-V"), ("EM-FA", "GaussCM")]:
        if m1 in W and m2 in W:
            ok = W[[m1, m2]].dropna(); diff = ok[m1] - ok[m2]
            pairs[f"{m1} vs {m2}"] = dict(blocks=int(len(ok)), wins=int((diff < 0).sum()), losses=int((diff > 0).sum()), mean_diff=float(diff.mean()),
                                          wilcoxon_p=float(stats.wilcoxon(ok[m1], ok[m2]).pvalue) if len(ok) >= 5 and (diff != 0).any() else None,
                                          datasets_better=sorted(set(i[0] for i in ok.index[diff < 0])))
    lg["pairs"] = pairs
    # MAR and MNAR only: who re-estimates the column statistics wins on strongly correlated data
    Wx = W[W.index.get_level_values("setting").isin(["MAR-30", "MNAR-30"])]
    ran = [m for m in ["Mean", "NDI-S", "NDI-F", "NDI-F-V", "GaussCM", "EM-FA", "EM-FA-k20", "EM-PPCA", "SoftImpute", "MICE"] if m in Wx and Wx[m].notna().all()]
    best = Wx[ran].idxmin(axis=1)
    mm = dict(blocks=int(len(Wx)), methods=ran, best_counts={m: int((best == m).sum()) for m in ran}, pairs={})
    for m1, m2 in [("EM-FA", "NDI-F-V"), ("MICE", "NDI-F-V"), ("SoftImpute", "NDI-F-V"), ("GaussCM", "NDI-F-V"), ("EM-FA", "NDI-F"), ("MICE", "EM-FA"), ("NDI-F-V", "NDI-F")]:
        if m1 in Wx and m2 in Wx:
            ok = Wx[[m1, m2]].dropna(); diff = ok[m1] - ok[m2]
            mm["pairs"][f"{m1} vs {m2}"] = dict(blocks=int(len(ok)), wins=int((diff < 0).sum()), losses=int((diff > 0).sum()), mean_diff=float(diff.mean()),
                                                wilcoxon_p=float(stats.wilcoxon(ok[m1], ok[m2]).pvalue) if len(ok) >= 5 and (diff != 0).any() else None)
    mm["by_block"] = {f"{d}|{s}": {m: f32(v) for m, v in row.items()} for (d, s), row in Wx[ran].iterrows()}
    lg["marmnar"] = mm
    lg["n_settings_by_dataset"] = {ds: int(len(g)) for ds, g in W.groupby(level="dataset")}
    # MCAR-30 only, all methods that ran there (NaN = not run)
    W30 = W.xs("MCAR-30", level="setting")
    lg["mcar30"] = {ds: {m: f32(v) for m, v in row.items()} for ds, row in W30.iterrows()}
    TW30 = a[a.setting == "MCAR-30"].pivot(index="dataset", columns="method", values="time_s")
    lg["mcar30_time"] = {ds: {m: f32(v) for m, v in row.items()} for ds, row in TW30.iterrows()}
    T["large"] = lg
    info = os.path.join(OUT, "large_datasets.csv")
    if os.path.exists(info):
        T["large_datasets"] = [dict(dataset=r.dataset, n=int(r.n), p=int(r.p), classes=int(r.classes), source=r.source) for r in pd.read_csv(info).drop_duplicates("dataset").itertuples()]

json.dump(T, open(os.path.join(OUT, "extra_tables.json"), "w"), indent=1)
print("sections:", list(T.keys()))
