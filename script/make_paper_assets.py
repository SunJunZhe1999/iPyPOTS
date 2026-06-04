#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper assets (LaTeX tables + figures + significance tests) for the
InterveneNet routing study from the benchmark output.

Self-contained: reads only metrics_summary.csv (per model/dataset/missing_rate/
seed MAE...) and, if present, ctcar_features.csv (80-dim per dataset/missing_rate)
to evaluate the router. Robust to partial data so it can be smoke-tested while the
benchmark is still running, then re-run unchanged at the end.

Outputs LaTeX tables to <paper>/tables and PDF figures to <paper>/figures, with
names matching the labels used in intervenenet.tex.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Dataset metadata mirrors run_paper_campaign.sh / set_dims_for_dataset.
DATASET_META = {
    "household_power":   (168, 11,  "Energy / residential load"),
    "appliances_energy": (72,  32,  "Energy / appliances"),
    "opsd_germany":      (168, 10,  "Energy / national grid"),
    "citylearn_zone5":   (168, 20,  "Energy / building"),
    "etth1":             (96,  7,   "Energy / transformer (hourly)"),
    "ettm1":             (96,  7,   "Energy / transformer (15-min)"),
    "solar":             (24,  137, "Energy / solar PV"),
    "eld":               (24,  370, "Energy / load diagrams"),
    "physionet_2012":    (48,  37,  "Clinical / ICU (cross-domain)"),
}
PRETTY_MODEL = {
    "mean": "Mean", "median": "Median", "locf": "LOCF", "saits": "SAITS",
    "tefn": "TEFN", "timemixerpp": "TimeMixer++", "tslanet": "TSLANet",
    "gpt4ts": "GPT4TS", "moment": "MOMENT", "uniformtsv": "UniFormTSV",
}
SHORT_DS = {
    "household_power": "HHP", "appliances_energy": "App", "opsd_germany": "OPSD",
    "citylearn_zone5": "City", "etth1": "ETTh1", "ettm1": "ETTm1", "solar": "Solar",
    "eld": "ELD", "physionet_2012": "Phys",
}
SCENARIO_KEYS = ["dataset", "missing_rate", "seed"]


def _fit(tex: str) -> str:
    """Wrap a table's tabular in adjustbox so it never overflows its container."""
    return re.sub(
        r"(\\begin\{tabular\}.*?\\end\{tabular\})",
        r"\\adjustbox{max width=\\linewidth}{%\n\1\n}",
        tex, flags=re.S,
    )


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["model"] = df["model"].str.lower()
    return df


def per_model_cost(root: Path) -> dict:
    """Wall-clock cost proxy per model = mean(.done mtime - run.log mtime)."""
    cost = {}
    for done in root.glob("**/.done"):
        run_log = done.parent / "run.log"
        if not run_log.exists():
            continue
        model = done.parent.parts[-4]  # <root>/model/dataset/mrX/seedY/.done
        dt = done.stat().st_mtime - run_log.stat().st_mtime
        if dt >= 0:
            cost.setdefault(model, []).append(dt)
    return {m: float(np.mean(v)) for m, v in cost.items() if v}


def scenario_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario, one column per model holding MAE."""
    piv = df.pivot_table(index=SCENARIO_KEYS, columns="model", values="MAE", aggfunc="mean")
    return piv


def oracle_single_router(piv: pd.DataFrame):
    models = list(piv.columns)
    complete = piv.dropna(axis=0, how="any")
    oracle = complete.min(axis=1)
    single_model = piv.mean(axis=0).idxmin()
    single = complete[single_model]
    return models, complete, oracle, single, single_model


def fmt(x, nd=4):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ---------------------------------------------------------------- tables
def table_datasets(present_datasets, out: Path):
    rows = []
    for ds in [d for d in DATASET_META if d in present_datasets]:
        ns, nf, dom = DATASET_META[ds]
        rows.append(f"\\texttt{{{ds.replace('_', chr(92)+'_')}}} & {ns} & {nf} & {dom} \\\\")
    body = "\n".join(rows)
    tex = (
        "\\begin{table}[t]\n\\centering\n\\caption{Datasets used in the routing study.}\n"
        "\\label{tab:datasets}\n\\begin{tabular}{lccl}\n\\toprule\n"
        "Dataset & $n_\\text{steps}$ & $n_\\text{feat}$ & Domain \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    out.write_text(_fit(tex))


def table_main_results(df: pd.DataFrame, piv: pd.DataFrame, out: Path):
    """Per-model mean MAE per dataset + Oracle + Single-best row."""
    models, complete, oracle, single, single_model = oracle_single_router(piv)
    datasets = sorted(df["dataset"].unique())
    order = [m for m in PRETTY_MODEL if m in models]
    esc = lambda s: s.replace("_", "\\_")
    header = "Model & " + " & ".join(SHORT_DS.get(d, esc(d)) for d in datasets) + " & Mean \\\\"
    lines = []
    for m in order:
        per_ds = [df[(df.model == m) & (df.dataset == d)]["MAE"].mean() for d in datasets]
        mean_all = np.nanmean(per_ds)
        lines.append(f"{PRETTY_MODEL[m]} & " + " & ".join(fmt(v, 3) for v in per_ds) + f" & {fmt(mean_all,3)} \\\\")
    # oracle / router rows (router added later when features available)
    orc_per_ds = []
    for d in datasets:
        sub = piv.loc[piv.index.get_level_values("dataset") == d].dropna(axis=0, how="any")
        orc_per_ds.append(sub.min(axis=1).mean() if len(sub) else np.nan)
    lines.append("\\midrule")
    lines.append(f"\\textit{{Oracle}} & " + " & ".join(fmt(v, 3) for v in orc_per_ds) + f" & {fmt(np.nanmean(orc_per_ds),3)} \\\\")
    colspec = "l" + "c" * (len(datasets) + 1)
    tex = (
        "\\begin{table*}[t]\n\\centering\n\\caption{Imputation MAE per model and dataset "
        "(mean over missing rates and seeds; dataset abbreviations per Table~\\ref{tab:datasets}). "
        "Lower is better; \\textit{Oracle} is the per-scenario best.}\n\\label{tab:main-results}\n"
        f"\\footnotesize\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header}\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )
    out.write_text(_fit(tex))
    return models, single_model


def table_router_cv(router: dict | None, single_mae: float, out: Path):
    if not router and out.exists():
        return  # keep a hand-populated table rather than clobbering it with placeholders
    def f(v):
        return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    if router:
        rows = (f"LOO  & {f(router['loo_acc'])} & {f(router['loo_routed_mae'])} & {f(router['loo_oracle_mae'])} & {f(single_mae)} \\\\\n"
                f"LOGO & {f(router['logo_acc'])} & {f(router['logo_routed_mae'])} & {f(router['logo_oracle_mae'])} & {f(single_mae)} \\\\")
        note = ""
    else:
        rows = "LOO  & -- & -- & -- & -- \\\\\nLOGO & -- & -- & -- & -- \\\\"
        note = " \\textit{Preliminary; refreshed when C-TCAR features are available.}"
    tex = (
        "\\begin{table}[t]\n\\centering\n\\caption{Router cross-validation: selection accuracy "
        "against the oracle and routed MAE, versus oracle and single-best references, under "
        "leave-one-out (LOO) and leave-one-group-out by dataset (LOGO)." + note + "}\n"
        "\\label{tab:router-cv}\n\\begin{tabular}{lcccc}\n\\toprule\n"
        "Protocol & Acc.\\ vs oracle & Routed MAE & Oracle MAE & Single-best MAE \\\\\n\\midrule\n"
        + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    out.write_text(_fit(tex))


# ---------------------------------------------------------------- significance
def significance_report(piv: pd.DataFrame) -> dict:
    complete = piv.dropna(axis=0, how="any")
    res = {"n_complete_scenarios": int(len(complete)), "models": list(complete.columns)}
    if len(complete) >= 3 and complete.shape[1] >= 3:
        fr = stats.friedmanchisquare(*[complete[m].values for m in complete.columns])
        res["friedman_stat"] = float(fr.statistic)
        res["friedman_p"] = float(fr.pvalue)
        ranks = complete.rank(axis=1).mean(axis=0)  # mean rank (1=best)
        res["mean_ranks"] = {m: float(r) for m, r in ranks.sort_values().items()}
    return res


# ---------------------------------------------------------------- router
def router_eval(piv: pd.DataFrame, feats_path: Path) -> dict | None:
    if not feats_path.exists():
        return None
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
    try:
        from script.ctcar_features import CTCAR_FEATURE_NAMES
    except Exception:
        CTCAR_FEATURE_NAMES = None
    feats = pd.read_csv(feats_path)
    complete = piv.dropna(axis=0, how="any").reset_index()
    if not len(complete):
        return None
    complete["best"] = complete[[c for c in piv.columns]].idxmin(axis=1)
    feat_cols = [c for c in feats.columns if (CTCAR_FEATURE_NAMES is None and c not in ("dataset", "missing_rate")) or (CTCAR_FEATURE_NAMES and c in CTCAR_FEATURE_NAMES)]
    merged = complete.merge(feats, on=["dataset", "missing_rate"], how="inner")
    if len(merged) < 5 or merged["best"].nunique() < 2:
        return None
    X = merged[feat_cols].fillna(0.0).values
    y = merged["best"].values
    groups = merged["dataset"].values
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)

    def cv_acc_and_error(splitter, gs=None):
        correct, routed_err, oracle_err = 0, [], []
        for tr, te in (splitter.split(X, y, gs) if gs is not None else splitter.split(X)):
            if len(np.unique(y[tr])) < 2:
                continue
            rf.fit(X[tr], y[tr])
            pred = rf.predict(X[te])
            for i, ti in enumerate(te):
                row = merged.iloc[ti]
                correct += int(pred[i] == y[ti])
                routed_err.append(row[pred[i]])
                oracle_err.append(row[[c for c in piv.columns]].min())
        n = len(routed_err)
        return (correct / n if n else float("nan"),
                float(np.mean(routed_err)) if n else float("nan"),
                float(np.mean(oracle_err)) if n else float("nan"))

    loo_acc, loo_err, loo_orc = cv_acc_and_error(LeaveOneOut())
    logo_acc, logo_err, logo_orc = cv_acc_and_error(LeaveOneGroupOut(), groups)
    return {"loo_acc": loo_acc, "loo_routed_mae": loo_err, "loo_oracle_mae": loo_orc,
            "logo_acc": logo_acc, "logo_routed_mae": logo_err, "logo_oracle_mae": logo_orc,
            "n": int(len(merged))}


# ---------------------------------------------------------------- figures
def fig_mean_rank(sig: dict, out: Path):
    if "mean_ranks" not in sig:
        return
    items = list(sig["mean_ranks"].items())
    names = [PRETTY_MODEL.get(m, m) for m, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(6, 3.2))
    plt.barh(names[::-1], vals[::-1], color="#4C72B0")
    plt.xlabel("Mean rank (lower is better)")
    plt.title(f"Friedman mean ranks (p={sig.get('friedman_p', float('nan')):.1e})")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def fig_cost_quality(df: pd.DataFrame, cost: dict, router: dict | None, out: Path):
    models = [m for m in PRETTY_MODEL if m in df["model"].unique() and m in cost]
    xs = [cost[m] for m in models]
    ys = [df[df.model == m]["MAE"].mean() for m in models]
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, c="#55A868", s=60)
    for m, x, y in zip(models, xs, ys):
        plt.annotate(PRETTY_MODEL.get(m, m), (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    if router and not np.isnan(router.get("logo_routed_mae", float("nan"))):
        plt.axhline(router["logo_routed_mae"], ls="--", c="#C44E52", label="Router (LOGO) MAE")
        plt.axhline(router["logo_oracle_mae"], ls=":", c="black", label="Oracle MAE")
        plt.legend(fontsize=8)
    plt.xscale("log")
    plt.xlabel("Compute cost per scenario (s, log scale)")
    plt.ylabel("Mean MAE")
    plt.title("Accuracy--cost trade-off")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/imputation/cuda/paper_campaign")
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--paper", default="../intervenenet")
    args = ap.parse_args()

    root = Path(args.root)
    metrics_path = Path(args.metrics) if args.metrics else root / "metrics_summary.csv"
    feats_path = Path(args.features) if args.features else root / "ctcar_features.csv"
    paper = Path(args.paper)
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    (paper / "figures").mkdir(parents=True, exist_ok=True)

    df = load_metrics(metrics_path)
    piv = scenario_pivot(df)
    cost = per_model_cost(root)
    sig = significance_report(piv)
    router = router_eval(piv, feats_path)

    _, _single_complete, _, _single_series, _ = oracle_single_router(piv)
    single_mae = float(_single_series.mean()) if len(_single_series) else float("nan")

    table_datasets(set(df["dataset"].unique()), paper / "tables" / "table_datasets.tex")
    table_main_results(df, piv, paper / "tables" / "table_main_results.tex")
    table_router_cv(router, single_mae, paper / "tables" / "table_router_cv.tex")
    fig_mean_rank(sig, paper / "figures" / "fig_critical_difference.pdf")
    fig_cost_quality(df, cost, router, paper / "figures" / "fig_router_vs_oracle.pdf")

    print("=== SUMMARY ===")
    print(f"rows={len(df)} scenarios={len(piv)} complete={sig['n_complete_scenarios']} models={sig['models']}")
    print(f"cost(s)={ {k: round(v,2) for k,v in cost.items()} }")
    if "friedman_p" in sig:
        print(f"Friedman p={sig['friedman_p']:.2e}; mean_ranks={ {k: round(v,2) for k,v in sig['mean_ranks'].items()} }")
    print(f"router={router}")
    print("wrote tables/figures to", paper)


if __name__ == "__main__":
    main()
