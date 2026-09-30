"""Manuscript figures from tables.json (print, 300 dpi). Palette: validated reference palette."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

T = json.load(open("tables.json"))
C = {"NDI-F": "#2a78d6", "NDI-0": "#eb6834", "Mean": "#1baf7a", "MissForest": "#eda100", "MICE": "#e87ba4",
     "GaussCM": "#008300", "GAIN": "#4a3aa7", "SoftImpute": "#e34948", "NDI-S": "#52514e", "KNN": "#898781", "NDI-C": "#898781",
     "NDI-F1": "#898781", "NDI-F-noclip": "#898781", "Copula-NDI-F": "#898781",
     "MIWAE": "#898781", "Sinkhorn": "#898781", "HyperImpute": "#898781", "ReMasker": "#898781",
     "EM-FA": "#138d90", "EM-PPCA": "#898781"}
MK = {"NDI-F": "o", "NDI-S": "h", "NDI-0": "s", "Mean": "^", "MissForest": "D", "MICE": "v", "GaussCM": "P", "GAIN": "X", "SoftImpute": "*", "KNN": "<", "NDI-C": ">", "EM-FA": "d", "EM-PPCA": "p"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "savefig.dpi": 300, "savefig.facecolor": "white"})
METHODS = ["Mean", "NDI-0", "NDI-C", "NDI-S", "NDI-F1", "NDI-F-noclip", "Copula-NDI-F", "NDI-F", "GaussCM", "EM-FA", "EM-PPCA",
           "KNN", "MICE", "MissForest", "SoftImpute", "GAIN"]
MAIN = ["Mean", "NDI-0", "NDI-C", "NDI-S", "NDI-F", "GaussCM", "EM-FA", "EM-PPCA", "KNN", "MICE", "MissForest", "SoftImpute", "GAIN"]
DEEP = ["MIWAE", "Sinkhorn", "HyperImpute", "ReMasker"]
PARETO = MAIN + DEEP


# ------------------------------------------------------------------ Fig 1: accuracy-cost Pareto
def fig_pareto():
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = {m: max(T["time_mean"][m], 3e-4) for m in PARETO}
    y = {m: np.mean([T["deep_by_setting"][st][m]["mean"] for st in ["MCAR-30", "MAR-30", "MNAR-30"]]) for m in PARETO}
    fam = lambda m: m.startswith("NDI")
    pts = sorted(((x[m], y[m], m) for m in PARETO))
    front, best = [], np.inf
    for xi, yi, m in pts:
        if yi < best:
            front.append((xi, yi, m)); best = yi
    ax.plot([f[0] for f in front], [f[1] for f in front], ls="--", lw=1.2, color=INK2, zorder=1, label="Pareto front")
    for m in PARETO:
        ax.scatter(x[m], y[m], s=46, color=C["NDI-F"] if fam(m) else MUTED, edgecolor="white", linewidth=0.8, zorder=3,
                   marker="o", label="NDI family (this study)" if m == "NDI-F" else ("baselines" if m == "Mean" else None))
    off = {"Mean": (6, -9), "NDI-0": (6, 5), "NDI-C": (6, 5), "NDI-S": (-40, 6), "NDI-F": (-6, -14), "GaussCM": (-20, -14), "KNN": (6, 4),
           "MICE": (6, 5), "MissForest": (6, -10), "SoftImpute": (-30, 8), "GAIN": (-6, -13), "MIWAE": (6, 5), "Sinkhorn": (6, -10),
           "HyperImpute": (6, 5), "ReMasker": (-52, -12), "EM-FA": (-42, -4), "EM-PPCA": (-4, -14)}
    for m in PARETO:
        ax.annotate(m, (x[m], y[m]), xytext=off[m], textcoords="offset points", fontsize=8,
                    color=INK if fam(m) else INK2, fontweight="bold" if m == "NDI-F" else "normal")
    ax.set_xscale("log"); ax.set_xlim(2e-4, 2e3)
    ax.set_xlabel("Mean imputation time per dataset (s, log scale; n ≤ 2,000)")
    ax.set_ylabel("z-RMSE, mean over 15 datasets × 3 settings (30 %)")
    ax.set_title("Imputation error against cost (lower-left is better)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig("figure5_pareto.png"); plt.close(fig)


def fig_cd_deep():
    fr = T["friedman_deep"]; ranks = fr["ranks"]; cd = fr["CD"]
    order = sorted(ranks, key=ranks.get)
    fig, ax = plt.subplots(figsize=(7.0, 4.0 + 0.25 * max(0, (len(order) + 1) // 2 - 9)))
    lo, hi = 1, len(order)
    ax.set_xlim(lo - 0.4, hi + 0.4); ax.set_ylim(-1.45 - 0.5 * ((len(order) + 1) // 2 - 1), 1.6); ax.axis("off")
    ax.plot([lo, hi], [0, 0], color=INK, lw=1)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [0, 0.15], color=INK, lw=1); ax.text(t, 0.25, str(t), ha="center", va="bottom", fontsize=8, color=INK)
    ax.text((lo + hi) / 2, 0.95, f"Average rank over {fr['N']} dataset × setting blocks (30 % settings, seed 0; 1 = best)", ha="center", fontsize=8.5, color=INK2)
    ax.plot([lo, lo + cd], [1.35, 1.35], color=INK, lw=1.5); ax.plot([lo, lo], [1.25, 1.45], color=INK, lw=1); ax.plot([lo + cd, lo + cd], [1.25, 1.45], color=INK, lw=1)
    ax.text(lo + cd + 0.15, 1.35, f"CD = {cd:.2f} (Nemenyi, α = 0.05)", va="center", fontsize=8, color=INK)
    half = (len(order) + 1) // 2
    for i, m in enumerate(order):
        r = ranks[m]
        if i < half:
            y = -1.05 - i * 0.5; xt = lo - 0.35
            ax.plot([r, r, xt], [0, y, y], color=C[m] if m in ("NDI-F", "NDI-0") else INK2, lw=1)
            ax.text(xt - 0.05, y, f"{m} ({r:.2f})", ha="right", va="center", fontsize=7.5, color=INK, fontweight="bold" if m == "NDI-F" else "normal")
        else:
            j = i - half; y = -1.05 - (len(order) - half - 1 - j) * 0.5; xt = hi + 0.35
            ax.plot([r, r, xt], [0, y, y], color=C[m] if m in ("NDI-F", "NDI-0") else INK2, lw=1)
            ax.text(xt + 0.05, y, f"({r:.2f}) {m}", ha="left", va="center", fontsize=7.5, color=INK, fontweight="bold" if m == "NDI-F" else "normal")
    rs = [ranks[m] for m in order]; cliques = []
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and rs[j + 1] - rs[i] < cd: j += 1
        if j > i and not any(a <= i and b >= j for a, b in cliques): cliques.append((i, j))
    for k, (a, b) in enumerate(cliques):
        y = -0.22 - 0.13 * k
        ax.plot([rs[a] - 0.04, rs[b] + 0.04], [y, y], color=INK, lw=3, solid_capstyle="butt")
    fig.tight_layout(); fig.savefig(f"figure4_cd_{len(order)}_methods.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig 2: dimension dependence of NDI-0
def fig_dimension():
    rows = T["per_dataset_mcar30"]
    short = {"MagicTelescope": "Magic", "wine-quality-red": "wine-q-red", "ionosphere": "ionosph.", "qsar-biodeg": "qsar", "banknote": "banknote", "diabetes": "diabetes"}
    names = [f"{short.get(r['dataset'], r['dataset'])}\n(p={r['p']})" for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w, xs = 0.21, np.arange(len(rows))
    for i, m in enumerate(["Mean", "NDI-0", "NDI-S", "NDI-F"]):
        vals = [r[m] for r in rows]
        ax.bar(xs + (i - 1.5) * w, vals, width=w - 0.02, color=C[m], label=m, zorder=3, linewidth=0)
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":", zorder=2)
    ax.text(len(rows) - 0.6, 1.015, "z-RMSE = 1 (variance of the column)", ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("z-RMSE, MCAR 30 %"); ax.set_ylim(0, 1.45)
    ax.set_title("MCAR 30 %, datasets ordered by number of columns p")
    ax.grid(axis="x", visible=False); ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig("figure2_dimension.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig 3: robustness to the missing rate
def fig_rate():
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    rates = [0.1, 0.3, 0.5]
    ends = {}
    for m in ["MissForest", "MICE", "GAIN", "EM-FA", "NDI-F", "NDI-S", "Mean", "NDI-0"]:
        ys = [T["mcar_trend"][m][str(r)] for r in rates]
        ax.plot(rates, ys, marker=MK[m], ms=6, lw=1.8, color=C[m], label=m, zorder=3, markeredgecolor="white", markeredgewidth=0.7)
        ends[m] = ys[-1]
    # push end labels apart (min gap in data units)
    lab = sorted(ends.items(), key=lambda kv: kv[1]); gap = 0.022; pos = {}
    for i, (m, y) in enumerate(lab):
        pos[m] = y if i == 0 else max(y, pos[lab[i - 1][0]] + gap)
    for m, y in ends.items():                    # a thin leader line where a label had to be moved away from its line
        moved = abs(pos[m] - y) > 0.004
        ax.annotate(m, (rates[-1], y), xytext=(rates[-1] + 0.035, pos[m]), textcoords="data", va="center", fontsize=8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=C[m], lw=0.7, shrinkA=0, shrinkB=2) if moved else None)
    ax.set_xticks(rates); ax.set_xticklabels(["10 %", "30 %", "50 %"]); ax.set_xlim(0.07, 0.64)
    ax.set_xlabel("Share of cells removed (MCAR)"); ax.set_ylabel("z-RMSE, mean over 15 datasets")
    ax.set_title("Degradation with the missing rate")
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.60), fontsize=7.5, ncol=2)
    fig.tight_layout(); fig.savefig("figure3_missing_rate.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig 4: critical-difference diagram (pooled)
def fig_cd():
    fr = T["friedman"]["pooled"]; ranks = fr["ranks"]; cd = fr["CD"]
    order = sorted(ranks, key=ranks.get)
    fig, ax = plt.subplots(figsize=(7.0, 3.9 + 0.25 * max(0, (len(order) + 1) // 2 - 7)))
    lo, hi = 1, len(order)
    ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(-1.45 - 0.5 * ((len(order) + 1) // 2 - 1), 1.6); ax.axis("off")
    ax.plot([lo, hi], [0, 0], color=INK, lw=1)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [0, 0.15], color=INK, lw=1); ax.text(t, 0.25, str(t), ha="center", va="bottom", fontsize=8, color=INK)
    ax.text((lo + hi) / 2, 0.95, f"Average rank over {fr['N']} dataset × setting blocks (1 = best)", ha="center", fontsize=8.5, color=INK2)
    # CD bar
    ax.plot([lo, lo + cd], [1.35, 1.35], color=INK, lw=1.5); ax.plot([lo, lo], [1.25, 1.45], color=INK, lw=1); ax.plot([lo + cd, lo + cd], [1.25, 1.45], color=INK, lw=1)
    ax.text(lo + cd + 0.1, 1.35, f"CD = {cd:.2f} (Nemenyi, α = 0.05)", va="center", fontsize=8, color=INK)
    # method stems: left half descends on the left, right half on the right
    half = (len(order) + 1) // 2
    for i, m in enumerate(order):
        r = ranks[m]
        if i < half:
            y = -1.05 - i * 0.5; xt = lo - 0.25
            ax.plot([r, r, xt], [0, y, y], color=C[m] if m in ("NDI-F", "NDI-0") else INK2, lw=1)
            ax.text(xt - 0.05, y, f"{m} ({r:.2f})", ha="right", va="center", fontsize=7.5, color=INK, fontweight="bold" if m == "NDI-F" else "normal")
        else:
            j = i - half; y = -1.05 - (len(order) - half - 1 - j) * 0.5; xt = hi + 0.25
            ax.plot([r, r, xt], [0, y, y], color=C[m] if m in ("NDI-F", "NDI-0") else INK2, lw=1)
            ax.text(xt + 0.05, y, f"({r:.2f}) {m}", ha="left", va="center", fontsize=7.5, color=INK, fontweight="bold" if m == "NDI-F" else "normal")
    # cliques: maximal groups with max-min rank < CD
    rs = [ranks[m] for m in order]; cliques = []
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and rs[j + 1] - rs[i] < cd: j += 1
        if j > i and not any(a <= i and b >= j for a, b in cliques): cliques.append((i, j))
    for k, (a, b) in enumerate(cliques):
        y = -0.22 - 0.13 * k
        ax.plot([rs[a] - 0.04, rs[b] + 0.04], [y, y], color=INK, lw=3, solid_capstyle="butt")
    fig.tight_layout(); fig.savefig(f"figure1_cd_{len(order)}_methods.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig 5: runtime scaling (p = 50)
def fig_scaling():
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    rows = [r for r in T["scaling"] if r["p"] == 50 and r["time_s"] is not None]
    for m in ["Mean", "NDI-F", "EM-FA", "GaussCM", "SoftImpute", "MICE", "GAIN", "MissForest", "KNN"]:
        pts = sorted((r["n"], r["time_s"]) for r in rows if r["method"] == m)
        if not pts: continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=MK[m], ms=6, lw=1.8, color=C[m], label=m, zorder=3, markeredgecolor="white", markeredgewidth=0.7)
        ax.annotate(m, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(7e2, 4.5e5)
    ax.set_xlabel("Rows n (p = 50, 30 % MCAR, synthetic factor-model data)"); ax.set_ylabel("Wall-clock seconds (2 CPU cores)")
    ax.set_title("Runtime scaling; a line that stops was not run at larger n")
    ax.legend(loc="lower right", fontsize=7.5, ncol=2)
    fig.tight_layout(); fig.savefig("figure6_scaling.png"); plt.close(fig)


fig_pareto(); fig_dimension(); fig_rate(); fig_cd(); fig_scaling(); fig_cd_deep()
print("figures written")
