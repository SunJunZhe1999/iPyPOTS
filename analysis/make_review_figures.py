#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate ten composite, multi-panel review figures for the cost-aware routing paper.

Each figure carries several panels and a deliberately varied chart type, with no
heatmap/matrix anywhere. Writes one PDF per figure into analysis/figures_review/ plus a
merged _all_figures.pdf. Does NOT touch the manuscript; the approved panels get
integrated later.

Run: LD_LIBRARY_PATH=$HOME/anaconda3/lib python analysis/make_review_figures.py
"""
from __future__ import annotations
import os, re, glob, sys, collections
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut

ROOT = Path("/home/Aboya_25R9803/projects/lab/SUN/iPyPOTS")
CAMP = ROOT / "output/imputation/cuda/paper_campaign"
OUT = ROOT / "analysis/figures_review"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12, "font.weight": "bold", "text.color": "black",
    "axes.labelsize": 12, "axes.titlesize": 12, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.labelcolor": "black", "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.color": "black", "ytick.color": "black", "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 2.0, "axes.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False,
})

# ---- consistent identity ----
C_DEEP, C_LLM, C_ROUTER, C_ORACLE, C_SINGLE, C_ROUTED = "#0288D1", "#D32F2F", "#1B5E20", "#000000", "#F57C00", "#1565C0"
GOOD, BAD = "#2E7D32", "#C62828"
MODEL_COLOR = {"locf":"#4C72B0","mean":"#55A868","median":"#8172B3",
               "saits":"#C44E52","tefn":"#CCB974","tslanet":"#64B5CD","gpt4ts":"#DD8452"}
FAMILY_OF = {"locf":"classical","mean":"classical","median":"classical",
             "saits":"deep","tefn":"deep","tslanet":"deep","gpt4ts":"llm"}
MK = {"classical":"o","deep":"s","llm":"D"}
LS = {"classical":"-","deep":"--","llm":":"}
PRETTY = {"mean":"Mean","median":"Median","locf":"LOCF","saits":"SAITS","tefn":"TEFN","tslanet":"TSLANet","gpt4ts":"GPT4TS"}
SHORT = {"household_power":"HHP","appliances_energy":"App","opsd_germany":"OPSD","citylearn_zone5":"City",
         "etth1":"ETTh1","ettm1":"ETTm1","solar":"Solar","eld":"ELD","physionet_2012":"Phys"}
ORDER = ["locf","mean","median","saits","tefn","tslanet","gpt4ts"]
GROUP_COLOR = {"Statistical":"#4C72B0","Temporal":"#55A868","Spectral":"#8172B3",
               "Causal-structural":"#C44E52","Causal-confounding":"#DD8452"}
SHORTG = {"Statistical":"Stat","Temporal":"Temp","Spectral":"Spec","Causal-structural":"C-str","Causal-confounding":"C-conf"}
SAVED = []

def mcolor(m): return MODEL_COLOR.get(m, "#444444")
def mmark(m): return MK[FAMILY_OF.get(m, "classical")]
def mls(m): return LS[FAMILY_OF.get(m, "classical")]
def model_lines(models):
    return [Line2D([0],[0], color=mcolor(m), marker=mmark(m), ls=mls(m), lw=2, mec="black", label=PRETTY[m]) for m in models]

def feat_group(name):
    if name.startswith("ctcar_stat_"): return "Statistical"
    if name.startswith("ctcar_temporal_"): return "Temporal"
    if name.startswith("ctcar_spectral_"): return "Spectral"
    if name.startswith("ctcar_causal_"): return "Causal-structural"
    return "Causal-confounding"  # proxy graph/confounder + action/outcome index features

def feat_short(name):
    for p in ("ctcar_stat_","ctcar_temporal_","ctcar_spectral_","ctcar_causal_","ctcar_proxy_","ctcar_"):
        if name.startswith(p): return name[len(p):]
    return name

def legend_outside(ax, handles=None, ncol=1, anchor=(1.02, 1.0), loc="upper left", fontsize=8):
    kw = dict(loc=loc, bbox_to_anchor=anchor, frameon=False, ncol=ncol, fontsize=fontsize)
    ax.legend(handles=handles, **kw) if handles is not None else ax.legend(**kw)

def vlabels(ax, bars, fmt="{:.0f}", min_h=1e-9, color="white", rot=90, fontsize=8):
    for b in bars:
        h = b.get_height()
        if abs(h) <= min_h: continue
        ax.text(b.get_x()+b.get_width()/2, b.get_y()+h/2, fmt.format(h),
                ha="center", va="center", rotation=rot, color=color, fontsize=fontsize, fontweight="bold")

def pareto_frontier(points):
    front, best = [], np.inf
    for x, y in sorted(points):
        if y < best - 1e-12: front.append((x, y)); best = y
    return front

def nemenyi_cd(k, n, alpha=0.05):
    q05 = {2:1.960,3:2.343,4:2.569,5:2.728,6:2.850,7:2.949,8:3.031,9:3.102,10:3.164}
    return q05.get(k, 3.164) * np.sqrt(k*(k+1)/(6.0*n))

def save(fig, name, title):
    fig.savefig(OUT/f"{name}.pdf"); plt.close(fig)   # PDF only, never PNG
    SAVED.append((name, title)); print("  wrote", name)

# ---- data ----
df = pd.read_csv(CAMP/"metrics_summary.csv"); df["model"] = df["model"].str.lower(); df = df[df.model != "timemixerpp"]
df["missing_rate"] = df["missing_rate"].astype(float)
feats = pd.read_csv(CAMP/"ctcar_features.csv"); feats["missing_rate"] = feats["missing_rate"].astype(float)
MODELS = [m for m in ORDER if m in df.model.unique()]
datasets = sorted(df.dataset.unique())
FC = [c for c in feats.columns if c not in ("dataset","missing_rate","seed","n_steps","n_features") and pd.api.types.is_numeric_dtype(feats[c])]
FF = feats[["dataset","missing_rate"]+FC].drop_duplicates(subset=["dataset","missing_rate"])
PIV = df.pivot_table(index=["dataset","missing_rate","seed"], columns="model", values="MAE")
MAE_CAP = float(np.nanpercentile(df["MAE"].values, 97))
RATES = sorted(df.missing_rate.unique())

def cost_epochs():
    pat = re.compile(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)')
    dur, ep, curves = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for rl in glob.glob(f"{CAMP}/*/*/mr*/seed*/run.log"):
        m = rl.split("/")[-5]
        try: t = open(rl, errors="replace").read()
        except Exception: continue
        ts = pat.findall(t)
        if len(ts) >= 2:
            d = (datetime.strptime(ts[-1], "%Y-%m-%d %H:%M:%S")-datetime.strptime(ts[0], "%Y-%m-%d %H:%M:%S")).total_seconds()
            if d >= 0: dur[m].append(d)
        e = re.findall(r'Epoch (\d+)', t)
        if e: ep[m].append(int(e[-1]))
        vc = [(int(a), float(b)) for a, b in re.findall(r'Epoch (\d+).*?validation (?:MSE|loss): ([\d.]+)', t)]
        if vc: curves[m].append(vc)
    return ({m: float(np.mean(v)) for m, v in dur.items() if v},
            {m: float(np.mean(v)) for m, v in ep.items() if v}, curves)
COST, EPOCHS, CURVES = cost_epochs()

def router(models, want_pred=False):
    d = df[df.model.isin(models)]
    piv = d.pivot_table(index=["dataset","missing_rate","seed"], columns="model", values="MAE")
    comp = piv.dropna(how="any").reset_index(); comp["missing_rate"] = comp["missing_rate"].astype(float)
    comp["best"] = comp[models].idxmin(axis=1)
    m = comp.merge(FF, on=["dataset","missing_rate"], how="inner")
    X, y, g = m[FC].fillna(0).values, m["best"].values, m["dataset"].values
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
    def cv(sp, gr=None):
        out = dict(acc=[], routed=[], oracle=[], pred=[], true=[], ds=[], rate=[])
        for tr, te in (sp.split(X, y, gr) if gr is not None else sp.split(X)):
            if len(np.unique(y[tr])) < 2: pr = [pd.Series(y[tr]).mode()[0]]*len(te)
            else: rf.fit(X[tr], y[tr]); pr = rf.predict(X[te])
            for i, ti in enumerate(te):
                row = m.iloc[ti]
                out["acc"].append(int(pr[i] == y[ti])); out["routed"].append(float(row[pr[i]]))
                out["oracle"].append(float(min(row[x] for x in models)))
                out["pred"].append(pr[i]); out["true"].append(y[ti])
                out["ds"].append(row["dataset"]); out["rate"].append(float(row["missing_rate"]))
        return out
    sb = piv.mean().idxmin(); single = float(comp[sb].mean())
    loo = cv(LeaveOneOut()); logo = cv(LeaveOneGroupOut(), g)
    rf.fit(X, y); imp = rf.feature_importances_
    res = dict(single=single, sb=sb, loo_acc=np.mean(loo["acc"]), loo=float(np.mean(loo["routed"])),
               logo_acc=np.mean(logo["acc"]), logo=float(np.mean(logo["routed"])), oracle=float(np.mean(loo["oracle"])), imp=imp)
    if want_pred:
        res.update(pred=loo["pred"], true=loo["true"], loo_routed=loo["routed"], loo_oracle=loo["oracle"],
                   loo_ds=loo["ds"], loo_rate=loo["rate"],
                   logo_pred=logo["pred"], logo_true=logo["true"], logo_routed=logo["routed"],
                   logo_oracle=logo["oracle"], logo_ds=logo["ds"])
    return res

POOL6 = ["mean","median","locf","saits","tefn","tslanet"]
R6 = router(POOL6, want_pred=True)
R7 = router(MODELS, want_pred=True)

_base = df[df.model.isin(MODELS)].pivot_table(index=["dataset","missing_rate","seed"], columns="model", values="MAE").dropna(how="any").reset_index()
_base["missing_rate"] = _base["missing_rate"].astype(float); _base["best"] = _base[MODELS].idxmin(axis=1)
ABL_MM = _base.merge(FF, on=["dataset","missing_rate"], how="inner"); ABL_Y = ABL_MM["best"].values; ABL_G = ABL_MM["dataset"].values
GROUPS_MAIN = ["Statistical","Temporal","Spectral","Causal-structural","Causal-confounding"]
GIDX = {G: [i for i, c in enumerate(FC) if feat_group(c) == G] for G in GROUPS_MAIN}

def group_router(keep_idx, kind="logo"):
    cols = [FC[i] for i in sorted(keep_idx)]
    if not cols: return float("nan")
    X = ABL_MM[cols].fillna(0).values
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
    splits = LeaveOneOut().split(X) if kind == "loo" else LeaveOneGroupOut().split(X, ABL_Y, ABL_G)
    routed = []
    for tr, te in splits:
        if len(np.unique(ABL_Y[tr])) < 2: pr = [pd.Series(ABL_Y[tr]).mode()[0]]*len(te)
        else: rf.fit(X[tr], ABL_Y[tr]); pr = rf.predict(X[te])
        for i, ti in enumerate(te): routed.append(float(ABL_MM.iloc[ti][pr[i]]))
    return float(np.mean(routed))

ALLIDX = list(range(len(FC)))
ABL_FULL = group_router(ALLIDX, "logo")
ABL_REMOVE = {"Full": ABL_FULL}
for G in GROUPS_MAIN:
    ABL_REMOVE["-"+SHORTG[G]] = group_router([i for i in ALLIDX if i not in set(GIDX[G])], "logo")

def arrows(ax, res, title):
    full = res["Full"]; items = [(k, v) for k, v in res.items() if k != "Full"]
    ax.axvline(full, ls="--", color="gray", lw=1.3, zorder=0)
    ax.text(full, len(items)-0.35, f"Full\n{full:.3f}", rotation=0, va="top", ha="center", fontsize=7, color="gray")
    for i, (k, v) in enumerate(items):
        helps = v < full; col = GOOD if helps else BAD
        ax.annotate("", xy=(v, i), xytext=(full, i), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
        ax.scatter(v, i, color=col, edgecolor="black", zorder=3, s=48)
        ax.text(v, i+0.32, f"{v-full:+.3f}", ha="center", va="bottom", fontsize=7, color=col, fontweight="bold")
    ax.set_yticks(range(len(items))); ax.set_yticklabels([k for k, _ in items]); ax.invert_yaxis()
    ax.set_xlabel("Routed MAE"); ax.set_title(title)
    ax.legend(handles=[Line2D([0],[0], color=GOOD, lw=2.5, label="Lowers MAE (helps)"),
                       Line2D([0],[0], color=BAD, lw=2.5, label="Raises MAE (hurts)")], frameon=False, fontsize=7, loc="lower right")

# ============================ 10 composite figures ============================
def fig1_heterogeneity():
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)
    fig.suptitle("Fig. 1  No free lunch: model behaviour is dataset dependent", fontsize=13)
    data = [df[df.model == m]["MAE"].clip(upper=MAE_CAP).values for m in MODELS]
    parts = axa.violinplot(data, showextrema=False, widths=0.85)
    for pc, m in zip(parts["bodies"], MODELS):
        pc.set_facecolor(mcolor(m)); pc.set_alpha(0.8); pc.set_edgecolor("black"); pc.set_linewidth(0.8)
    axa.scatter(range(1, len(MODELS)+1), [np.median(df[df.model == m]["MAE"]) for m in MODELS], color="white", edgecolor="black", zorder=5, s=34)
    axa.set_xticks(range(1, len(MODELS)+1)); axa.set_xticklabels([PRETTY[m] for m in MODELS], rotation=25, ha="right")
    axa.set_ylabel("MAE"); axa.set_ylim(0, MAE_CAP); axa.set_title("(a) MAE distribution per model")
    xs = np.arange(len(datasets))
    for m in MODELS:
        ys = []
        for d in datasets:
            means = {mm: df[(df.model == mm) & (df.dataset == d)]["MAE"].mean() for mm in MODELS}
            avail = sorted([mm for mm in MODELS if not np.isnan(means[mm])], key=lambda mm: means[mm])
            ys.append(avail.index(m)+1 if m in avail else np.nan)
        axb.plot(xs, ys, marker=mmark(m), ls=mls(m), color=mcolor(m), lw=2, ms=6)
    axb.set_yticks(range(1, len(MODELS)+1)); axb.invert_yaxis()
    axb.set_xticks(xs); axb.set_xticklabels([SHORT[d] for d in datasets], rotation=30, ha="right")
    axb.set_ylabel("Rank (1 = best)"); axb.set_title("(b) Model rank across datasets")
    win = PIV[MODELS].idxmin(axis=1)
    wd = PIV.reset_index()[["dataset"]].copy(); wd["win"] = win.values
    tab = wd.groupby("dataset")["win"].value_counts().unstack().reindex(index=datasets).fillna(0)
    bottom = np.zeros(len(datasets))
    for m in [mm for mm in MODELS if mm in tab.columns]:
        vals = tab[m].values
        vlabels(axc, axc.bar([SHORT[d] for d in datasets], vals, bottom=bottom, color=mcolor(m), edgecolor="black", width=0.82)); bottom += vals
    axc.set_ylabel("# scenario wins"); axc.set_ylim(0, max(bottom.max(), 1)*1.12)
    axc.set_xticks(range(len(datasets))); axc.set_xticklabels([SHORT[d] for d in datasets], rotation=30, ha="right")
    axc.set_title("(c) Per-dataset winning model"); legend_outside(axc, handles=model_lines(MODELS))
    save(fig, "F01_heterogeneity", "Heterogeneity: violin, rank, wins")

def fig2_cost():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    fig.suptitle("Fig. 2  Cost and accuracy span two orders of magnitude", fontsize=13)
    pts = []
    for m in MODELS:
        x, y = COST.get(m, np.nan), df[df.model == m]["MAE"].mean()
        if np.isnan(x): continue
        axa.scatter(x, y, s=70+9*EPOCHS.get(m, 0), color=mcolor(m), edgecolor="black", marker=mmark(m), zorder=3)
        axa.annotate(PRETTY[m], (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, fontweight="bold")
        pts.append((x, y))
    fr = pareto_frontier(pts)
    axa.plot([p[0] for p in fr], [p[1] for p in fr], color="#555555", lw=1.5, zorder=2, label="Pareto front")
    axa.axhline(R7["oracle"], ls=":", color=C_ORACLE, lw=1.6, label="Oracle")
    axa.axhline(R7["logo"], ls="--", color=C_ROUTER, lw=1.6, label="Router (LOGO)")
    axa.set_xscale("log"); axa.set_xlabel("Cost per scenario (s, log)"); axa.set_ylabel("Mean MAE")
    axa.set_title("(a) Accuracy vs cost (marker size = epochs)"); axa.legend(frameon=False, fontsize=8, loc="upper left")
    Bs = sorted({COST[m] for m in MODELS if m in COST}); Bx, orac, sing = [], [], []
    for B in Bs:
        feas = [m for m in MODELS if COST.get(m, 1e9) <= B+1e-6]
        if not feas: continue
        Bx.append(B); orac.append(PIV[feas].min(axis=1).mean()); sing.append(min(PIV[m].mean() for m in feas))
    axb.step(Bx, orac, where="post", color=C_ROUTER, marker="o", label="Oracle within budget")
    axb.step(Bx, sing, where="post", color=C_SINGLE, marker="s", label="Single-best within budget")
    axb.set_xscale("log"); axb.set_xlabel("Compute budget B (s, log)"); axb.set_ylabel("Mean MAE")
    axb.set_title("(b) Best achievable accuracy under a budget"); axb.legend(frameon=False, fontsize=8)
    save(fig, "F02_cost", "Cost-accuracy Pareto + budget frontier")

def fig3_routing_gains():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    fig.suptitle("Fig. 3  Routing recovers most of the oracle gain", fontsize=13)
    grp = ["Single", "LOO", "LOGO", "Oracle"]; x = np.arange(4); w = 0.38
    v6 = [R6["single"], R6["loo"], R6["logo"], R6["oracle"]]; v7 = [R7["single"], R7["loo"], R7["logo"], R7["oracle"]]
    vlabels(axa, axa.bar(x-w/2, v6, w, color=C_DEEP, edgecolor="black", label="Deep+classical (6)"), fmt="{:.3f}")
    vlabels(axa, axa.bar(x+w/2, v7, w, color=C_LLM, edgecolor="black", label="+ GPT4TS (7)"), fmt="{:.3f}")
    axa.set_xticks(x); axa.set_xticklabels(grp); axa.set_ylabel("Mean MAE"); axa.set_ylim(0, max(v6+v7)*1.20)
    axa.set_title("(a) Router vs baselines vs oracle"); axa.legend(frameon=False, fontsize=8, loc="upper right")
    sb = R6["sb"]
    sbd = PIV[sb].groupby(level="dataset").mean(); ocd = PIV[MODELS].min(axis=1).groupby(level="dataset").mean()
    rdf = pd.DataFrame({"ds": R6["loo_ds"], "r": R6["loo_routed"]}).groupby("ds")["r"].mean()
    order = sorted(datasets, key=lambda d: sbd.get(d, np.nan), reverse=True)
    for i, d in enumerate(order):
        s, o, r = sbd.get(d, np.nan), ocd.get(d, np.nan), rdf.get(d, np.nan)
        axb.plot([o, s], [i, i], color="lightgray", lw=3, zorder=1)
        axb.scatter(s, i, color=C_SINGLE, edgecolor="black", zorder=3, s=46, label="Single-best" if i == 0 else None)
        axb.scatter(o, i, color=C_ROUTER, edgecolor="black", zorder=3, s=46, label="Oracle" if i == 0 else None)
        if r == r: axb.scatter(r, i, color=C_ROUTED, edgecolor="black", marker="D", zorder=4, s=42, label="Routed (LOO)" if i == 0 else None)
    axb.set_yticks(range(len(order))); axb.set_yticklabels([SHORT[d] for d in order])
    axb.set_xlabel("MAE"); axb.set_title(f"(b) Per-dataset gap (6-pool, single-best={PRETTY[sb]})")
    axb.margins(x=0.12); axb.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "F03_routing_gains", "Routing gains + per-dataset gap")

def fig4_significance():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    fig.suptitle("Fig. 4  Statistical comparison of the model pool", fontsize=13)
    comp = PIV[MODELS].dropna(how="any"); ranks = comp.rank(axis=1).mean(axis=0); order = ranks.sort_values().index.tolist()
    cd = nemenyi_cd(len(MODELS), len(comp)); fr = stats.friedmanchisquare(*[comp[m].values for m in MODELS])
    for i, m in enumerate(order):
        axa.errorbar(ranks[m], i, xerr=cd/2, fmt=mmark(m), color=mcolor(m), ecolor="gray", elinewidth=2, capsize=3, ms=10, mec="black", zorder=3)
    axa.set_yticks(range(len(order))); axa.set_yticklabels([PRETTY[m] for m in order]); axa.invert_yaxis()
    axa.set_xlabel("Mean rank (lower = better)"); axa.set_title(f"(a) Mean ranks $\\pm$ CD/2 (Friedman $p={fr.pvalue:.1e}$)")
    axa.text(0.98, 0.96, f"Nemenyi CD={cd:.2f}, N={len(comp)}", transform=axa.transAxes, ha="right", va="top", fontsize=8)
    sb = R7["sb"]; rows = []
    for m in MODELS:
        if m == sb: continue
        sub = PIV[[m, sb]].dropna()
        try: p = stats.wilcoxon(sub[m], sub[sb]).pvalue
        except Exception: p = np.nan
        rows.append((m, int((sub[m] < sub[sb]).sum()), int((sub[m] > sub[sb]).sum()), p))
    rows.sort(key=lambda r: r[1]-r[2])
    for i, (m, wbet, wwor, p) in enumerate(rows):
        axb.barh(i, wbet, color=GOOD, edgecolor="black"); axb.barh(i, -wwor, color=BAD, edgecolor="black")
        axb.text(wbet+1, i, (f"p={p:.0e}" if p == p else ""), va="center", fontsize=7)
    axb.axvline(0, color="black", lw=1)
    axb.set_yticks(range(len(rows))); axb.set_yticklabels([PRETTY[m] for m, *_ in rows])
    axb.set_xlabel(f"# scenarios worse  |  better than {PRETTY[sb]}")
    axb.set_title(f"(b) Wilcoxon vs single-best ({PRETTY[sb]})")
    axb.legend(handles=[Patch(facecolor=GOOD, edgecolor="black", label="better"), Patch(facecolor=BAD, edgecolor="black", label="worse")], frameon=False, fontsize=8, loc="lower right")
    save(fig, "F04_significance", "Critical difference + Wilcoxon")

def fig5_robustness():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    fig.suptitle("Fig. 5  Robustness to the missing rate", fontsize=13)
    for m in MODELS:
        mu, se = [], []
        for r in RATES:
            v = df[(df.model == m) & (np.isclose(df.missing_rate, r))]["MAE"].values
            mu.append(np.nanmean(v)); se.append(np.nanstd(v)/max(np.sqrt(len(v)), 1))
        mu, se = np.array(mu), np.array(se)
        axa.plot(RATES, mu, marker=mmark(m), ls=mls(m), color=mcolor(m), lw=2); axa.fill_between(RATES, mu-se, mu+se, color=mcolor(m), alpha=0.12)
    axa.set_xticks(RATES); axa.set_xlabel("Missing rate"); axa.set_ylabel("Mean MAE ($\\pm$ SE)"); axa.set_title("(a) Per-model error vs missing rate")
    legend_outside(axa, handles=model_lines(MODELS))
    sb = R6["sb"]; orac_s = PIV[MODELS].min(axis=1); rr = pd.DataFrame({"rate": R6["loo_rate"], "r": R6["loo_routed"]})
    sing = [df[(df.model == sb) & (np.isclose(df.missing_rate, r))]["MAE"].mean() for r in RATES]
    orac = [orac_s[np.isclose(orac_s.index.get_level_values("missing_rate").astype(float), r)].mean() for r in RATES]
    routed = [rr[np.isclose(rr.rate, r)]["r"].mean() for r in RATES]
    axb.plot(RATES, sing, marker="s", color=C_SINGLE, label=f"Single-best ({PRETTY[sb]})")
    axb.plot(RATES, routed, marker="D", color=C_ROUTED, label="Routed (LOO)")
    axb.plot(RATES, orac, marker="o", color=C_ROUTER, label="Oracle")
    axb.set_xticks(RATES); axb.set_xlabel("Missing rate"); axb.set_ylabel("Mean MAE")
    axb.set_title("(b) Routing vs baselines (6-pool)"); axb.legend(frameon=False, fontsize=8)
    save(fig, "F05_robustness", "Robustness to missing rate")

def fig6_importance():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    fig.suptitle("Fig. 6  Which C-TCAR features the router relies on", fontsize=13)
    imp = R7["imp"]; topo = np.argsort(imp)[::-1][:12][::-1]
    for yi, idx in enumerate(topo):
        g = feat_group(FC[idx]); axa.hlines(yi, 0, imp[idx], color=GROUP_COLOR[g], lw=2.6, zorder=1)
        axa.scatter(imp[idx], yi, color=GROUP_COLOR[g], edgecolor="black", zorder=3, s=52)
    axa.set_yticks(range(len(topo))); axa.set_yticklabels([feat_short(FC[i]) for i in topo], fontsize=7)
    axa.set_xlabel("Random-forest importance"); axa.set_title("(a) Top-12 individual features")
    present = [g for g in GROUP_COLOR if any(feat_group(FC[i]) == g for i in topo)]
    axa.legend(handles=[Patch(facecolor=GROUP_COLOR[g], edgecolor="black", label=g) for g in present], frameon=False, fontsize=7, loc="lower right")
    gsum = {}
    for i, c in enumerate(FC): gsum[feat_group(c)] = gsum.get(feat_group(c), 0.0) + imp[i]
    gs = sorted(gsum.items(), key=lambda kv: kv[1])
    for yi, (g, v) in enumerate(gs):
        axb.hlines(yi, 0, v, color=GROUP_COLOR[g], lw=3, zorder=1); axb.scatter(v, yi, color=GROUP_COLOR[g], edgecolor="black", zorder=3, s=70)
        axb.text(v+0.005, yi, f"{v:.2f}", va="center", fontsize=8, fontweight="bold")
    axb.set_yticks(range(len(gs))); axb.set_yticklabels([g for g, _ in gs])
    axb.set_xlim(0, max(v for _, v in gs)*1.18); axb.set_xlabel("Summed importance"); axb.set_title("(b) Importance by feature group")
    save(fig, "F06_importance", "Feature importance (top features + groups)")

def fig7_ablation():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.2, 4.3), constrained_layout=True)
    fig.suptitle("Fig. 7  The cross-dataset router overfits its feature set", fontsize=13)
    arrows(axa, ABL_REMOVE, "(a) Remove one feature group (LOGO)")
    imp = R7["imp"]; oi = np.argsort(imp)[::-1]
    ks = [5, 10, 20, 40, len(FC)]
    loo_c = [group_router(oi[:k], "loo") for k in ks]
    logo_c = [group_router(oi[:k], "logo") for k in ks]
    axb.plot(ks, loo_c, marker="o", color=C_DEEP, lw=2.2, label="LOO (within pool)")
    axb.plot(ks, logo_c, marker="s", color=C_LLM, lw=2.2, label="LOGO (cross dataset)")
    axb.axhline(R7["single"], ls=":", color=C_SINGLE, lw=1.6, label="Single-best")
    axb.set_xlabel("Number of top features used"); axb.set_ylabel("Routed MAE")
    axb.set_title("(b) Robust in-pool, overfits cross-dataset"); axb.legend(frameon=False, fontsize=8, loc="center right")
    save(fig, "F07_ablation", "Ablation: removal arrows + feature-count curve")

def fig8_selection():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    fig.suptitle("Fig. 8  Router selection quality", fontsize=13)
    for vals, col, lab in [([R6["loo_acc"], R6["logo_acc"]], C_DEEP, "Deep+classical (6)"),
                           ([R7["loo_acc"], R7["logo_acc"]], C_LLM, "+ GPT4TS (7)")]:
        axa.plot([0, 1], vals, marker="o", color=col, lw=2.6, ms=10, mec="black", label=lab)
        for xx, vv in zip([0, 1], vals): axa.text(xx, vv+0.025, f"{vv:.2f}", ha="center", fontsize=9, fontweight="bold")
    axa.set_xticks([0, 1]); axa.set_xticklabels(["LOO", "LOGO"]); axa.set_xlim(-0.3, 1.3); axa.set_ylim(0, 1.12)
    axa.set_ylabel("Selection accuracy"); axa.set_title("(a) Generalization gap: LOO vs LOGO"); axa.legend(frameon=False, fontsize=8, loc="lower left")
    rt = np.array(R6["logo_routed"]); orc = np.array(R6["logo_oracle"]); corr = np.array([p == t for p, t in zip(R6["logo_pred"], R6["logo_true"])])
    lim = max(rt.max(), orc.max())*1.05
    axb.plot([0, lim], [0, lim], ls="--", color="gray", lw=1.2, zorder=1)
    axb.scatter(orc[corr], rt[corr], color=GOOD, edgecolor="black", s=30, alpha=0.8, label="Correct pick", zorder=3)
    axb.scatter(orc[~corr], rt[~corr], color=C_SINGLE, edgecolor="black", s=44, label="Misrouted", zorder=4)
    axb.set_xlim(0, lim); axb.set_ylim(0, lim); axb.set_aspect("equal", adjustable="box")
    axb.set_xlabel("Oracle MAE"); axb.set_ylabel("Routed MAE")
    axb.set_title(f"(b) Routed vs oracle (6-pool, LOGO, regret {np.mean(rt-orc):.3f})"); axb.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "F08_selection", "Selection accuracy + routed-vs-oracle")

def fig9_errors():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    fig.suptitle("Fig. 9  Where cross-dataset routing errs and what it costs", fontsize=13)
    lp, lt, lds = R6["logo_pred"], R6["logo_true"], R6["logo_ds"]
    accs = []
    for d in datasets:
        idx = [j for j, dd in enumerate(lds) if dd == d]
        if idx: accs.append((d, float(np.mean([lp[j] == lt[j] for j in idx])), len(idx)))
    accs.sort(key=lambda r: r[1])
    for i, (d, a, n) in enumerate(accs):
        col = GOOD if a >= 0.8 else (C_SINGLE if a >= 0.5 else BAD)
        axa.hlines(i, 0, a, color=col, lw=2.6, zorder=1); axa.scatter(a, i, color=col, edgecolor="black", s=70, zorder=3)
        axa.text(a+0.02, i, f"{a:.2f}", va="center", fontsize=8, fontweight="bold")
    axa.set_yticks(range(len(accs))); axa.set_yticklabels([SHORT[d] for d, _, _ in accs])
    axa.set_xlim(0, 1.15); axa.set_xlabel("LOGO selection accuracy"); axa.set_title("(a) Accuracy per held-out dataset (6-pool)")
    for ks, lab, col in [(("loo_routed", "loo_oracle"), "LOO (within pool)", C_DEEP),
                         (("logo_routed", "logo_oracle"), "LOGO (cross dataset)", C_LLM)]:
        reg = np.sort(np.array(R6[ks[0]]) - np.array(R6[ks[1]]))
        axb.step(reg, np.arange(1, len(reg)+1)/len(reg), where="post", color=col, lw=2.2, label=lab)
    axb.axvline(0, color="gray", ls="--", lw=1)
    axb.set_xlabel("Routing regret (routed $-$ oracle MAE)"); axb.set_ylabel("Cumulative fraction of scenarios")
    axb.set_title("(b) Regret: LOO vs LOGO (6-pool)"); axb.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "F09_errors", "Per-dataset LOGO accuracy + regret ECDF")

def fig10_training():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.0, 4.2), constrained_layout=True)
    fig.suptitle("Fig. 10  Training dynamics of the deep imputers", fontsize=13)
    trained = ["saits", "tefn", "tslanet", "gpt4ts"]; conv = {}
    for m in trained:
        if m not in CURVES: continue
        bylen = collections.defaultdict(list)
        for c in CURVES[m]:
            for e, v in c: bylen[e].append(v)
        es = sorted(bylen); mean = np.array([np.mean(bylen[e]) for e in es])
        axa.plot(es, mean, color=mcolor(m), ls=mls(m), lw=2, label=PRETTY[m])
        tot = mean[0] - mean[-1]; target = mean[0] - 0.9*tot
        ce = next((e for e, val in zip(es, mean) if val <= target), es[-1]); conv[m] = ce
        axa.scatter(ce, mean[es.index(ce)], color=mcolor(m), edgecolor="black", marker=mmark(m), zorder=4, s=48)
    axa.set_yscale("log"); axa.set_xlabel("Epoch"); axa.set_ylabel("Validation loss (log)")
    axa.set_title("(a) Convergence (marker = 90% of total drop)"); axa.legend(frameon=False, fontsize=8)
    ms = [m for m in trained if m in conv]
    for i, m in enumerate(ms):
        axb.hlines(i, 0, conv[m], color=mcolor(m), lw=2.6, zorder=1); axb.scatter(conv[m], i, color=mcolor(m), edgecolor="black", marker=mmark(m), s=82, zorder=3)
        axb.text(conv[m]+0.6, i, f"{conv[m]}", va="center", fontsize=9, fontweight="bold")
    axb.set_yticks(range(len(ms))); axb.set_yticklabels([PRETTY[m] for m in ms]); axb.invert_yaxis()
    axb.set_xlim(0, max(conv.values())*1.25); axb.set_xlabel("Epochs to 90% of total loss drop"); axb.set_title("(b) Convergence speed")
    save(fig, "F10_training", "Learning curves + convergence speed")

for fn in [fig1_heterogeneity, fig2_cost, fig3_routing_gains, fig4_significance, fig5_robustness,
           fig6_importance, fig7_ablation, fig8_selection, fig9_errors, fig10_training]:
    try: fn()
    except Exception as ex:
        import traceback; print("  FAILED", fn.__name__, repr(ex)[:160]); traceback.print_exc()

import fitz
merged = fitz.open()
for name, _ in SAVED:
    merged.insert_pdf(fitz.open(OUT/f"{name}.pdf"))
merged.save(OUT/"_all_figures.pdf"); merged.close()
print(f"\n{len(SAVED)} figure PDFs + _all_figures.pdf in {OUT}")
