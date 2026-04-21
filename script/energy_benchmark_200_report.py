#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write a compact report for the expanded energy benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in columns) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expanded benchmark markdown report.")
    parser.add_argument("--root", default="output/imputation/mps/energy_benchmark")
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--out", default="script/energy_benchmark_200_report.md")
    args = parser.parse_args()

    root = Path(args.root)
    metrics_path = Path(args.metrics) if args.metrics else root / "metrics_summary.csv"
    metrics = pd.read_csv(metrics_path)
    metrics["missing_rate"] = metrics["missing_rate"].astype(float)
    metrics["seed"] = metrics["seed"].astype(str)

    completed = len(metrics)
    datasets = sorted(metrics["dataset"].dropna().unique())
    models = sorted(metrics["model"].dropna().unique())
    missing_rates = sorted(metrics["missing_rate"].dropna().unique())
    seeds = sorted(metrics["seed"].dropna().unique())
    scenarios = metrics[["dataset", "missing_rate"]].drop_duplicates().shape[0]

    grouped = (
        metrics.groupby(["dataset", "missing_rate", "model"], as_index=False)
        .agg(MAE=("MAE", "mean"), MSE=("MSE", "mean"), RMSE=("RMSE", "mean"), MRE=("MRE", "mean"), seeds=("seed", "nunique"))
        .sort_values(["dataset", "missing_rate", "MAE"])
    )
    winners = grouped.loc[grouped.groupby(["dataset", "missing_rate"])["MAE"].idxmin()]
    winners = winners.rename(columns={"model": "best_model", "seeds": "seeds_used"})
    winners = winners[["dataset", "missing_rate", "best_model", "MAE", "MSE", "RMSE", "MRE", "seeds_used"]]

    model_counts = (
        metrics.groupby("model", as_index=False)
        .agg(runs=("run", "count"), datasets=("dataset", "nunique"), mean_MAE=("MAE", "mean"))
        .sort_values(["runs", "mean_MAE"], ascending=[False, True])
    )
    incomplete_models = model_counts.loc[model_counts["runs"] < scenarios, ["model", "runs"]]

    dataset_counts = (
        metrics.groupby("dataset", as_index=False)
        .agg(runs=("run", "count"), models=("model", "nunique"), missing_rates=("missing_rate", "nunique"), mean_MAE=("MAE", "mean"))
        .sort_values("dataset")
    )

    router_diag = _read_json(root / "routing" / "router_diagnostics.json")
    ctcar_diag = _read_json(root / "ctcar_routing" / "ctcar_router_diagnostics.json")
    ctcar_path = root / "ctcar_features.csv"
    ctcar_rows = None
    if ctcar_path.exists():
        ctcar_rows = len(pd.read_csv(ctcar_path))

    lines = [
        "# Expanded Energy C-TCAR Benchmark Report",
        "",
        "This report summarizes the current expanded benchmark after adding more energy datasets and a wider model matrix.",
        "",
        "## Scale",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Completed metric rows | {completed} |",
        f"| Distinct datasets | {len(datasets)} |",
        f"| Distinct models | {len(models)} |",
        f"| Distinct missing rates | {len(missing_rates)} |",
        f"| Distinct seeds | {len(seeds)} |",
        f"| Dataset/missing-rate scenarios | {scenarios} |",
    ]
    if ctcar_rows is not None:
        lines.append(f"| C-TCAR rows | {ctcar_rows} |")

    lines.extend(
        [
            "",
            "## Datasets",
            "",
            ", ".join(f"`{item}`" for item in datasets),
            "",
            "## Models",
            "",
            ", ".join(f"`{item}`" for item in models),
            "",
            "## Coverage By Dataset",
            "",
            *_markdown_table(dataset_counts, ["dataset", "runs", "models", "missing_rates", "mean_MAE"]),
            "",
            "## Coverage By Model",
            "",
            *_markdown_table(model_counts, ["model", "runs", "datasets", "mean_MAE"]),
            "",
            "## Current Winners By Scenario",
            "",
            *_markdown_table(winners, ["dataset", "missing_rate", "best_model", "MAE", "MSE", "RMSE", "MRE", "seeds_used"]),
            "",
            "## Router Diagnostics",
            "",
            "| Router | Training | Leave-one-scenario | Leave-one-dataset |",
            "| --- | ---: | ---: | ---: |",
            (
                "| Metadata router | "
                f"{router_diag.get('training_accuracy', '')} | "
                f"{router_diag.get('leave_one_scenario_out_accuracy', '')} | "
                f"{router_diag.get('leave_one_dataset_out_accuracy', '')} |"
            ),
            (
                "| C-TCAR router | "
                f"{ctcar_diag.get('training_accuracy', '')} | "
                f"{ctcar_diag.get('leave_one_scenario_out_accuracy', '')} | "
                f"{ctcar_diag.get('leave_one_dataset_out_accuracy', '')} |"
            ),
            "",
            "## Notes",
            "",
            "Experiments were run on the local Apple Silicon environment, so the benchmark prioritizes broad coverage over very large backbones or multi-seed saturation.",
        ]
    )
    if not incomplete_models.empty:
        shortfall = ", ".join(f"`{row.model}` ({int(row.runs)}/{scenarios})" for row in incomplete_models.itertuples())
        lines.append(
            f"The following heavier models have partial local coverage because they are expensive on this machine: {shortfall}."
        )

    metadata_loso = router_diag.get("leave_one_scenario_out_accuracy")
    ctcar_loso = ctcar_diag.get("leave_one_scenario_out_accuracy")
    if metadata_loso is not None and ctcar_loso is not None:
        if ctcar_loso >= metadata_loso:
            router_note = (
                "On the current runs, the C-TCAR router matches or exceeds the metadata router under leave-one-scenario validation."
            )
        else:
            router_note = (
                "On the current small local run, the metadata router has higher leave-one-scenario accuracy; "
                "C-TCAR still adds interpretable causal, temporal, and spectral descriptors that can improve as more scenarios and intervention labels are added."
            )
        lines.append(router_note)

    lines.extend(
        [
            "",
            "## Important Caveat",
            "",
            "The current C-TCAR implementation is a causal-aware routing proxy. True intervention labels, ATE ground truth, and full do-calculus evaluation are still future work.",
            "",
        ]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote expanded benchmark report to {out_path}")


if __name__ == "__main__":
    main()
