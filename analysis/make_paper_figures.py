#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication-quality figures for the cost-aware imputation-routing paper.

Adapts the house style of analysis/generate_figures.py (600 DPI serif, bold black
text, colorblind palette, clean spines) to THIS project's results. Reads the
benchmark metrics, the C-TCAR features, and the real per-run training time parsed
from run.log timestamps, then writes PDF figures into the manuscript figures/ dir.

Run:  LD_LIBRARY_PATH=$HOME/anaconda3/lib python analysis/make_paper_figures.py
"""
from __future__ import annotations
import os, re, glob, argparse, collections
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12, "font.weight": "bold", "text.color": "black",
    "axes.labelsize": 13, "axes.titlesize": 14, "axes.labelweight": "bold",
    "axes.titleweight": "bold", "axes.labelcolor": "black",
    "xtick.labelsize": 10, "ytick.labelsize": 10, "xtick.color": "black", "ytick.color": "black",
    "legend.fontsize": 10, "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5, "lines.linewidth": 2.0,
    "axes.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
})
C_BASE, C_DEEP, C_LLM, C_ROUTER, C_ORACLE, C_SINGLE = "#888888", "#0288D1", "#D32F2F", "#1B5E20", "#000000", "#F57C00"
PRETTY = {"mean":"Mean","median":"Median","locf":"LOCF","saits":"SAITS","tefn":"TEFN",
          "tslanet":"TSLANet","gpt4ts":"GPT4TS","timemixerpp":"TimeMixer++"}
SHORT_DS = {"household_power":"HHP","appliances_energy":"App","opsd_germany":"OPSD","citylearn_zone5":"City",
            "etth1":"ETTh1","ettm1":"ETTm1","solar":"Solar","eld":"ELD","physionet_2012":"Phys"}
FAMILY = {"mean":C_BASE,"median":C_BASE,"locf":C_BASE,"saits":C_DEEP,"tefn":C_DEEP,"tslanet":C_DEEP,"gpt4ts":C_LLM}
ORDER = ["locf","mean","median","saits","tefn","tslanet","gpt4ts"]


def per_model_cost_epochs(root):
    pat = re.compile(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)')
    dur, ep = collections.defaultdict(list), collections.defaultdict(list)
    for rl in glob.glob(f"{root}/*/*/mr*/seed*/run.log"):
        m = rl.split("/")[-5]
        try: txt = open(rl, errors="replace").read()
        except Exception: continue
        ts = pat.findall(txt)
        if len(ts) >= 2:
            d = (datetime.strptime(ts[-1], "%Y-%m-%d %H:%M:%S") - datetime.strptime(ts[0], "%Y-%m-%d %H:%M:%S")).total_seconds()
            if d >= 0: dur[m].append(d)
        e = re.findall(r'Epoch (\d+)', txt)
        if e: ep[m].append(int(e[-1]))
    cost = {m: float(np.mean(v)) for m, v in dur.items() if v}
    epochs = {m: float(np.mean(v)) for m, v in ep.items() if v}
    return cost, epochs


def router_eval(df, feats, models):
    d = df[df.model.isin(models)]
    piv = d.pivot_table(index=["dataset","missing_rate","seed"], columns="model", values="MAE")
    comp = piv.dropna(how="any").reset_index(); comp["missing_rate"] = comp["missing_rate"].astype(float)
    comp["best"] = comp[models].idxmin(axis=1)
    fc = [c for c in feats.columns if c not in ("dataset","missing_rate","seed","n_steps","n_features") and pd.api.types.is_numeric_dtype(feats[c])]
    ff = feats[["dataset","missing_rate"]+fc].drop_duplicates(subset=["dataset","missing_rate"])
    m = comp.merge(ff, on=["dataset","missing_rate"], how="inner")
    X, y, g = m[fc].fillna(0).values, m["best"].values, m["dataset"].values
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
    def cv(sp, gr=None):
        routed, orac = [], []
        for tr, te in (sp.split(X, y, gr) if gr is not None else sp.split(X)):
            if len(np.unique(y[tr])) < 2: pr = [pd.Series(y[tr]).mode()[0]]*len(te)
            else: rf.fit(X[tr], y[tr]); pr = rf.predict(X[te])
            for i, ti in enumerate(te):
                row = m.iloc[ti]; routed.append(float(row[pr[i]])); orac.append(float(min(row[x] for x in models)))
        return float(np.mean(routed)), float(np.mean(orac))
    sb = piv.mean().idxmin(); single = float(comp[sb].mean())
    loo, logo = cv(LeaveOneOut()), cv(LeaveOneGroupOut(), g)
    return dict(single=single, loo=loo[0], logo=logo[0], oracle=loo[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/imputation/cuda/paper_campaign")
    ap.add_argument("--figs", default="../intervenenet/figures")
    args = ap.parse_args()
    root, figs = Path(args.root), Path(args.figs); figs.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root/"metrics_summary.csv"); df["model"] = df["model"].str.lower(); df = df[df.model != "timemixerpp"]
    feats = pd.read_csv(root/"ctcar_features.csv"); feats["missing_rate"] = feats["missing_rate"].astype(float)
    cost, epochs = per_model_cost_epochs(str(root))
    models = [m for m in ORDER if m in df.model.unique()]
    piv = df.pivot_table(index=["dataset","missing_rate","seed"], columns="model", values="MAE")
    datasets = sorted(df.dataset.unique())

    # ---- F: cost-accuracy pareto (the cost-aware headline) ----
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in models:
        x, y = cost.get(m, np.nan), df[df.model == m]["MAE"].mean()
        ax.scatter(x, y, s=140, color=FAMILY[m], edgecolor="black", zorder=3)
        ax.annotate(PRETTY[m], (x, y), xytext=(6, 5), textcoords="offset points", fontsize=10, fontweight="bold")
    rr = router_eval(df, feats, models)
    ax.axhline(rr["oracle"], ls=":", color=C_ORACLE, lw=1.8, label="Oracle")
    ax.axhline(rr["logo"], ls="--", color=C_ROUTER, lw=1.8, label="Router (LOGO)")
    ax.set_xscale("log"); ax.set_xlabel("Training+inference time per scenario (s, log scale)")
    ax.set_ylabel("Mean MAE"); ax.set_title("Accuracy versus computation cost")
    from matplotlib.lines import Line2D
    fams = [Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markeredgecolor='k',markersize=10,label=l)
            for c,l in [(C_BASE,"Classical"),(C_DEEP,"Deep"),(C_LLM,"LLM")]]
    ax.legend(handles=fams+ax.get_legend_handles_labels()[0], frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(figs/"fig_router_vs_oracle.pdf"); plt.close()

    # ---- F: per-model computation cost + epochs ----
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = [PRETTY[m] for m in models]; ys = [cost.get(m, 0) for m in models]
    bars = ax.bar(xs, ys, color=[FAMILY[m] for m in models], edgecolor="black")
    ax.set_yscale("log"); ax.set_ylabel("Training+inference time (s, log)")
    ax.set_title("Per-model computation cost")
    for m, b in zip(models, bars):
        e = epochs.get(m, 0)
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.05, (f"{e:.0f} ep" if e else "0 ep"),
                ha="center", fontsize=8.5, fontweight="bold")
    plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(figs/"fig_computation_cost.pdf"); plt.close()

    # ---- F: heterogeneity heatmap (per-model mean MAE by dataset) ----
    mat = np.full((len(models), len(datasets)), np.nan)
    for i, m in enumerate(models):
        for j, ds in enumerate(datasets):
            v = df[(df.model == m) & (df.dataset == ds)]["MAE"].mean()
            mat[i, j] = v
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
    # mark per-dataset best model
    for j in range(len(datasets)):
        i = int(np.nanargmin(mat[:, j])); ax.scatter(j, i, marker="*", s=120, color="white", edgecolor="black", zorder=3)
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels([SHORT_DS.get(d, d) for d in datasets], rotation=30, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels([PRETTY[m] for m in models])
    ax.set_title("Per-model MAE by dataset (white marker = best per dataset)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Mean MAE")
    ax.grid(False); plt.tight_layout(); plt.savefig(figs/"fig_heterogeneity.pdf"); plt.close()

    # ---- F: Friedman mean ranks (critical-difference style) ----
    comp = piv[models].dropna(how="any"); ranks = comp.rank(axis=1).mean(axis=0).sort_values()
    fr = stats.friedmanchisquare(*[comp[m].values for m in comp.columns])
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.barh([PRETTY[m] for m in ranks.index][::-1], ranks.values[::-1],
            color=[FAMILY[m] for m in ranks.index][::-1], edgecolor="black")
    ax.set_xlabel("Mean rank (lower is better)")
    ax.set_title(f"Friedman mean ranks ($p={fr.pvalue:.1e}$)")
    plt.tight_layout(); plt.savefig(figs/"fig_critical_difference.pdf"); plt.close()

    print("wrote figures to", figs, "| models:", models)
    print("cost(s):", {m: round(cost.get(m,0),1) for m in models}, "| epochs:", {m: round(epochs.get(m,0)) for m in models})
    print("router:", {k: round(v,3) for k,v in rr.items()})


if __name__ == "__main__":
    main()
