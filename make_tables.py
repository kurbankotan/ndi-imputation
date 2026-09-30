"""Derive every manuscript table from the benchmark's long-format CSVs -> tables.json"""
import json, math, os
import numpy as np, pandas as pd
from scipy import stats

r = pd.read_csv("rmse_long.csv"); d = pd.read_csv("downstream_long.csv"); sc = pd.read_csv("scaling_long.csv")
ds = pd.read_csv("datasets.csv")
METHODS = ["Mean", "NDI-0", "NDI-C", "NDI-S", "NDI-F1", "NDI-F-noclip", "Copula-NDI-F", "NDI-F", "GaussCM", "EM-FA", "EM-PPCA",
           "KNN", "MICE", "MissForest", "SoftImpute", "GAIN"]
SETTINGS = ["MCAR-10", "MCAR-30", "MCAR-50", "MAR-30", "MNAR-30"]
out = {}

r["setting"] = r.mechanism + "-" + (r.rate * 100).round().astype(int).astype(str)
syn = r[r.natural == False].copy()
per = syn.groupby(["setting", "dataset", "method"]).agg(rmse=("rmse", "mean"), mae=("mae", "mean"),
                                                        oob=("oob", "mean"), time_s=("time_s", "mean")).reset_index()

# ---- Table: datasets
out["datasets"] = ds.to_dict(orient="records")

# ---- Table: z-RMSE mean ± sd across datasets, and average rank, per setting
tab = {}
for s in SETTINGS:
    blk = per[per.setting == s].pivot(index="dataset", columns="method", values="rmse")[METHODS]
    rk = blk.rank(axis=1).mean()
    tab[s] = {m: dict(mean=float(blk[m].mean()), sd=float(blk[m].std()), rank=float(rk[m])) for m in METHODS}
# pooled
blk_all = per.assign(block=per.dataset + "|" + per.setting).pivot(index="block", columns="method", values="rmse")[METHODS]
rk_all = blk_all.rank(axis=1).mean()
tab["pooled"] = {m: dict(mean=float(blk_all[m].mean()), sd=float(blk_all[m].std()), rank=float(rk_all[m])) for m in METHODS}
out["rmse_by_setting"] = tab

# ---- Friedman / Nemenyi per setting and pooled
from scipy.stats import studentized_range
def q_nemenyi(k, alpha=0.05):
    """studentized range q(alpha, k, inf) / sqrt(2) (Demsar, 2006); 3.354 for k = 14"""
    return float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))
fr = {}
for s in SETTINGS + ["pooled"]:
    blk = blk_all if s == "pooled" else per[per.setting == s].pivot(index="dataset", columns="method", values="rmse")[METHODS]
    chi2, p = stats.friedmanchisquare(*[blk[c].to_numpy() for c in blk.columns])
    k, N = blk.shape[1], len(blk)
    cd = q_nemenyi(k) * math.sqrt(k * (k + 1) / (6.0 * N))
    fr[s] = dict(chi2=float(chi2), p=float(p), N=int(N), k=int(k), CD=float(cd), ranks={m: float(v) for m, v in blk.rank(axis=1).mean().items()})
out["friedman"] = fr

# ---- Wilcoxon NDI-S vs others (pooled blocks), Holm
ref = "NDI-F"; rows = []
for m in METHODS:
    if m == ref: continue
    diff = blk_all[ref] - blk_all[m]
    pv = stats.wilcoxon(blk_all[ref], blk_all[m]).pvalue
    rows.append(dict(method=m, wins=int((diff < 0).sum()), losses=int((diff > 0).sum()), mean_diff=float(diff.mean()), p=float(pv)))
rows.sort(key=lambda x: x["p"])
for i, row in enumerate(rows):
    row["p_holm"] = min(1.0, row["p"] * (len(rows) - i))
# enforce monotonicity of Holm-adjusted p-values
for i in range(1, len(rows)):
    rows[i]["p_holm"] = max(rows[i]["p_holm"], rows[i - 1]["p_holm"])
out["wilcoxon"] = rows

# ---- MCAR rate trend
out["mcar_trend"] = {m: {str(rt): float(v) for rt, v in syn[(syn.mechanism == "MCAR") & (syn.method == m)].groupby("rate").rmse.mean().items()} for m in METHODS}

# ---- per-dataset MCAR-30 (with p)
b = per[per.setting == "MCAR-30"].pivot(index="dataset", columns="method", values="rmse")[METHODS]
b.insert(0, "p", ds.set_index("dataset").loc[b.index, "p"]); b.insert(1, "n", ds.set_index("dataset").loc[b.index, "n"])
b = b.sort_values("p")
out["per_dataset_mcar30"] = [dict(dataset=i, **{k: (float(v) if k not in ("p", "n") else int(v)) for k, v in row.items()}) for i, row in b.iterrows()]
out["spearman_p_vs_gap"] = {
    "NDI-0 minus Mean": float(stats.spearmanr(b["p"], b["NDI-0"] - b["Mean"]).statistic),
    "NDI-S minus Mean": float(stats.spearmanr(b["p"], b["NDI-S"] - b["Mean"]).statistic),
    "NDI-S minus GaussCM": float(stats.spearmanr(b["p"], b["NDI-S"] - b["GaussCM"]).statistic)}
out["ndi0_beats_mean_mcar30"] = list(b.index[b["NDI-0"] < b["Mean"]])

# ---- natural datasets: extra-masking RMSE
nat = r[r.natural == True].groupby(["dataset", "method"]).rmse.mean().unstack()[METHODS]
out["natural_rmse"] = {i: {m: float(v) for m, v in row.items()} for i, row in nat.iterrows()}

# ---- out-of-range share (all settings incl. natural), and mean time
out["oob"] = {m: float(v) for m, v in r.groupby("method").oob.mean().items()}
out["time_mean"] = {m: float(v) for m, v in r.groupby("method").time_s.mean().items()}
out["time_median"] = {m: float(v) for m, v in r.groupby("method").time_s.median().items()}

# ---- downstream: synthetic settings (mean over datasets, seeds, classifiers) + accuracy retained vs complete
d["setting"] = d.mechanism + "-" + (d.rate * 100).round().astype(int).astype(str)
syn_d = d[d.natural == False]
comp = syn_d[syn_d.method == "COMPLETE"].groupby(["dataset", "classifier"]).accuracy.mean().rename("acc_complete")
dd = syn_d[syn_d.method != "COMPLETE"].merge(comp, on=["dataset", "classifier"])
dd["retained"] = dd.accuracy / dd.acc_complete
down = {}
for s in ["MCAR-30", "MAR-30", "MNAR-30"]:
    g = dd[dd.setting == s]
    down[s] = {m: dict(acc=float(g[g.method == m].accuracy.mean()), f1=float(g[g.method == m].f1_macro.mean()),
                       retained=float(g[g.method == m].retained.mean())) for m in METHODS}
down["complete"] = dict(acc=float(syn_d[syn_d.method == "COMPLETE"].accuracy.mean()), f1=float(syn_d[syn_d.method == "COMPLETE"].f1_macro.mean()))
# per classifier at MCAR-30
down["by_classifier_MCAR-30"] = {c: {m: float(v) for m, v in dd[(dd.setting == "MCAR-30") & (dd.classifier == c)].groupby("method").accuracy.mean().items()} for c in ["LR", "KNN", "RF", "SVC"]}
down["complete_by_classifier"] = {c: float(v) for c, v in syn_d[syn_d.method == "COMPLETE"].groupby("classifier").accuracy.mean().items()}
out["downstream"] = down

# ---- downstream on natural datasets (method x classifier)
natd = {}
for dsn, g in d[d.natural == True].groupby("dataset"):
    t = g.groupby(["method", "classifier"]).accuracy.mean().unstack()
    natd[dsn] = {m: {**{c: float(t.loc[m, c]) for c in t.columns}, "mean": float(t.loc[m].mean())} for m in METHODS + ["MIWAE", "Sinkhorn", "HyperImpute", "ReMasker"] if m in t.index}
out["natural_downstream"] = natd

# ---- scaling
out["scaling"] = [dict(n=int(a.n), p=int(a.p), method=a.method, time_s=(None if pd.isna(a.time_s) else float(a.time_s)),
                       rmse=(None if pd.isna(a.rmse) else float(a.rmse)), status=a.status) for a in sc.itertuples()]

wm = pd.read_csv("warmstart_long.csv")
g = wm.groupby("init").agg(rmse_init=("rmse_init", "mean"), rmse_iter1=("rmse_iter1", "mean"), rmse_final=("rmse_final", "mean"),
                           iterations=("iterations", "mean"), within1=("iters_to_within_1pct", "mean"), time_s=("time_s", "mean"))
out["warmstart"] = {i: {k: float(v) for k, v in row.items()} for i, row in g.iterrows()}
pv = wm.pivot(index=["dataset", "seed"], columns="init", values="time_s")
out["warmstart_time_saving"] = float(1 - (pv["NDI-F"] / pv["mean"]).mean())
pf = wm.pivot(index=["dataset", "seed"], columns="init", values="rmse_final")
out["warmstart_final_wins"] = dict(ndif_better=int((pf["NDI-F"] < pf["mean"]).sum()), mean_better=int((pf["NDI-F"] > pf["mean"]).sum()))
out["warmstart_wilcoxon_time_p"] = float(stats.wilcoxon(pv["NDI-F"], pv["mean"]).pvalue)
out["warmstart_wilcoxon_rmse_p"] = float(stats.wilcoxon(pf["NDI-F"], pf["mean"]).pvalue)
# ---- deep / AutoML baselines: 30 % settings, seed 0 for ALL methods (same masks), N = 15 datasets
DEEP = ["MIWAE", "Sinkhorn", "HyperImpute", "ReMasker"]
ALL18 = METHODS + DEEP
S30 = ["MCAR-30", "MAR-30", "MNAR-30"]
s0 = syn[(syn.seed == 0) & (syn.setting.isin(S30))]
p0 = s0.pivot_table(index=["setting", "dataset"], columns="method", values="rmse")[ALL18]
deep_tab = {}
for st in S30:
    blk = p0.loc[st]
    rk = blk.rank(axis=1).mean()          # NaN rows (HyperImpute failures) rank NaN -> ignored in mean
    deep_tab[st] = {m: dict(mean=float(blk[m].mean()), sd=float(blk[m].std()), rank=float(rk[m]), n=int(blk[m].notna().sum())) for m in ALL18}
out["deep_by_setting"] = deep_tab
# Friedman over the 3 x 30 % settings with all 18 methods; blocks with a failed method are dropped
blk_all18 = p0.dropna(axis=0, how="any")
chi2, p = stats.friedmanchisquare(*[blk_all18[c].to_numpy() for c in blk_all18.columns])
k, N = blk_all18.shape[1], len(blk_all18)
q = q_nemenyi(k)
cd = q * math.sqrt(k * (k + 1) / (6.0 * N))
out["friedman_deep"] = dict(chi2=float(chi2), p=float(p), N=int(N), k=int(k), CD=float(cd), q=q,
                            ranks={m: float(v) for m, v in blk_all18.rank(axis=1).mean().items()},
                            dropped_blocks=[f"{i[1]} ({i[0]})" for i in p0.index if i not in blk_all18.index])
rows = []
for m in ALL18:
    if m == "NDI-F": continue
    pair = p0[["NDI-F", m]].dropna()
    diff = pair["NDI-F"] - pair[m]
    pv = stats.wilcoxon(pair["NDI-F"], pair[m]).pvalue
    rows.append(dict(method=m, blocks=int(len(pair)), wins=int((diff < 0).sum()), losses=int((diff > 0).sum()), mean_diff=float(diff.mean()), p=float(pv)))
rows.sort(key=lambda x: x["p"])
for i, row in enumerate(rows):
    row["p_holm"] = min(1.0, row["p"] * (len(rows) - i))
for i in range(1, len(rows)):
    rows[i]["p_holm"] = max(rows[i]["p_holm"], rows[i - 1]["p_holm"])
out["wilcoxon_deep"] = rows
# natural datasets incl. deep methods (mean over available seeds)
nat18 = r[r.natural == True].groupby(["dataset", "method"]).rmse.mean().unstack()[ALL18]
out["natural_rmse18"] = {i: {m: (None if pd.isna(v) else float(v)) for m, v in row.items()} for i, row in nat18.iterrows()}
# times & scaling entries for deep methods
out["time_mean"].update({m: float(v) for m, v in r.groupby("method").time_s.mean().items() if m in DEEP})
out["hyperimpute_failures"] = r[(r.method == "HyperImpute") & (r.error.fillna("") != "")][["dataset", "mechanism"]].apply(lambda x: f"{x.dataset} ({x.mechanism})", axis=1).tolist()
out["deep_outliers"] = {m: {f"{d}": float(v) for (d, v) in p0.loc["MCAR-30"][m].items() if v > 1.5} for m in DEEP}

# downstream accuracy at seed 0 for all 18 methods (30 % settings; mean over 15 datasets x 4 classifiers)
dsl = pd.read_csv("downstream_long.csv")
d0 = dsl[(dsl.seed == 0) & (dsl.natural == False) & (dsl.rate == 0.3)]
out["downstream_deep"] = {f"{mech}-30": {m: dict(acc=float(g.accuracy.mean()), f1=float(g.f1_macro.mean())) for m, g in d0[d0.mechanism == mech].groupby("method")} for mech in ["MCAR", "MAR", "MNAR"]}
# HyperImpute / NDI-F on the MCAR-30 datasets excluding the two HyperImpute outliers, for the text
sub = [d for d in p0.loc["MCAR-30"].index if d not in ("ecoli", "yeast")]
out["mcar30_excl_outliers"] = {m: float(p0.loc["MCAR-30"].loc[sub, m].mean()) for m in ALL18}
# mean per-run time of the deep methods on the synthetic settings only (n <= 2000), for the cost text
out["time_mean_syn"] = {m: float(v) for m, v in syn.groupby("method").time_s.mean().items()}
# ---- EM iterations of EM-FA / EM-PPCA (benchmark runs and scaling runs)
if os.path.exists("method_info_long.csv"):
    mi = pd.read_csv("method_info_long.csv")
    mi["scaling"] = mi.dataset.astype(str).str.startswith("scaling")
    out["em_iterations"] = {m: dict(median=float(g.iterations.median()), mean=float(g.iterations.mean()),
                                    q90=float(g.iterations.quantile(0.9)), max=int(g.iterations.max()),
                                    converged=float(g.converged.astype(str).str.lower().eq("true").mean()), runs=int(len(g)),
                                    k_median=float(g.k.median()))
                            for m, g in mi[~mi.scaling].groupby("method")}
    out["em_iterations_scaling"] = {f"{m}|{row.dataset}": int(row.iterations) for m, g in mi[mi.scaling].groupby("method") for row in g.itertuples()}
# ---- speed of the machine that ran the EM methods (second run), relative to the machine of the main run
if os.path.exists("timing_check.txt"):
    import re
    _t = [float(x) for x in re.findall(r"([0-9.]+) s", open("timing_check.txt").readline())[:2]]
    out["second_run_machine_ratio"] = _t[0] / _t[1]          # NDI-F on the 100,000 x 50 scaling matrix
json.dump(out, open("tables.json", "w"), indent=1)
print("tables.json written")
print(json.dumps(out["rmse_by_setting"]["pooled"], indent=0)[:600])
print(out["wilcoxon"])
print(out["downstream"]["MCAR-30"]["NDI-S"], out["downstream"]["complete"])
