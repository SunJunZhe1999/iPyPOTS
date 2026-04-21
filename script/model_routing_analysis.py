#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a compact model-routing report from benchmark metrics."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


DATASET_META = {
    "physionet_2012": {"n_steps": 48, "n_features": 37, "domain": "clinical", "is_energy": 0},
    "appliances_energy": {"n_steps": 72, "n_features": 32, "domain": "energy", "is_energy": 1},
    "household_power": {"n_steps": 168, "n_features": 11, "domain": "energy", "is_energy": 1},
    "citylearn_zone5": {"n_steps": 168, "n_features": 20, "domain": "energy", "is_energy": 1},
    "opsd_germany": {"n_steps": 168, "n_features": 10, "domain": "energy", "is_energy": 1},
    "etth1": {"n_steps": 96, "n_features": 7, "domain": "energy_transformer", "is_energy": 1},
    "etth2": {"n_steps": 96, "n_features": 7, "domain": "energy_transformer", "is_energy": 1},
    "ettm1": {"n_steps": 96, "n_features": 7, "domain": "energy_transformer", "is_energy": 1},
    "ettm2": {"n_steps": 96, "n_features": 7, "domain": "energy_transformer", "is_energy": 1},
    "solar": {"n_steps": 24, "n_features": 137, "domain": "solar_energy", "is_energy": 1},
    "solar_alabama": {"n_steps": 24, "n_features": 137, "domain": "solar_energy", "is_energy": 1},
    "eld": {"n_steps": 24, "n_features": 370, "domain": "electricity_load", "is_energy": 1},
    "electricity_load_diagrams": {"n_steps": 24, "n_features": 370, "domain": "electricity_load", "is_energy": 1},
}


def _add_metadata(rows: pd.DataFrame) -> pd.DataFrame:
    meta = pd.DataFrame.from_dict(DATASET_META, orient="index").reset_index(names="dataset")
    enriched = rows.merge(meta, on="dataset", how="left")
    if enriched[["n_steps", "n_features", "domain", "is_energy"]].isnull().any().any():
        missing = sorted(enriched.loc[enriched["n_steps"].isna(), "dataset"].unique())
        raise ValueError(f"Missing dataset metadata for: {missing}")
    return enriched


def build_routing_table(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics.groupby(["dataset", "missing_rate", "model"], as_index=False)
        .agg(MAE=("MAE", "mean"), MSE=("MSE", "mean"), RMSE=("RMSE", "mean"), MRE=("MRE", "mean"), seeds=("seed", "nunique"))
        .sort_values(["dataset", "missing_rate", "MAE"])
    )

    rows = []
    for (dataset, missing_rate), scenario in grouped.groupby(["dataset", "missing_rate"], sort=True):
        scenario = scenario.sort_values("MAE").reset_index(drop=True)
        best = scenario.iloc[0]
        runner_up = scenario.iloc[1] if len(scenario) > 1 else scenario.iloc[0]
        mae_gap = float(runner_up["MAE"] - best["MAE"])
        relative_gain = mae_gap / float(runner_up["MAE"]) if runner_up["MAE"] else 0.0
        rows.append(
            {
                "dataset": dataset,
                "missing_rate": missing_rate,
                "best_model": best["model"],
                "best_MAE": best["MAE"],
                "best_MSE": best["MSE"],
                "best_RMSE": best["RMSE"],
                "best_MRE": best["MRE"],
                "runner_up_model": runner_up["model"],
                "runner_up_MAE": runner_up["MAE"],
                "MAE_gap_to_runner_up": mae_gap,
                "relative_gain_vs_runner_up": relative_gain,
                "candidate_models": len(scenario),
                "seeds_per_candidate": int(scenario["seeds"].max()),
            }
        )

    return _add_metadata(pd.DataFrame(rows))


def train_router(routing_table: pd.DataFrame, max_depth: int, random_state: int):
    feature_cols = ["dataset", "domain", "is_energy", "n_steps", "n_features", "missing_rate"]
    X = routing_table[feature_cols]
    y = routing_table["best_model"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["dataset", "domain"]),
            ("num", "passthrough", ["is_energy", "n_steps", "n_features", "missing_rate"]),
        ]
    )
    router = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("tree", DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=1, random_state=random_state)),
        ]
    )
    router.fit(X, y)

    diagnostics = {"training_accuracy": accuracy_score(y, router.predict(X))}
    if len(routing_table) > 1 and y.nunique() > 1:
        loo_pred = cross_val_predict(router, X, y, cv=LeaveOneOut())
        diagnostics["leave_one_scenario_out_accuracy"] = accuracy_score(y, loo_pred)

        logo = LeaveOneGroupOut()
        groups = routing_table["dataset"]
        logo_pred = cross_val_predict(router, X, y, cv=logo.split(X, y, groups=groups))
        diagnostics["leave_one_dataset_out_accuracy"] = accuracy_score(y, logo_pred)

    fitted_preprocessor = router.named_steps["preprocess"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    rules = export_text(router.named_steps["tree"], feature_names=list(feature_names))
    return router, diagnostics, rules


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze model routing from benchmark metrics.")
    parser.add_argument("--metrics", default="output/imputation/mps/energy_benchmark/metrics_summary.csv")
    parser.add_argument("--out-dir", default="output/imputation/mps/energy_benchmark/routing")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(metrics_path)
    routing_table = build_routing_table(metrics)
    router, diagnostics, rules = train_router(routing_table, args.max_depth, args.random_state)

    routing_csv = out_dir / "routing_table.csv"
    model_path = out_dir / "router.joblib"
    diagnostics_path = out_dir / "router_diagnostics.json"
    rules_path = out_dir / "router_rules.txt"

    routing_table.to_csv(routing_csv, index=False)
    joblib.dump(router, model_path)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    rules_path.write_text(rules, encoding="utf-8")

    print(f"Wrote routing table to {routing_csv}")
    print(f"Wrote router model to {model_path}")
    print(f"Wrote diagnostics to {diagnostics_path}")
    print(f"Wrote rules to {rules_path}")
    print(json.dumps(diagnostics, indent=2))
    print(routing_table[["dataset", "missing_rate", "best_model", "best_MAE", "runner_up_model", "runner_up_MAE", "relative_gain_vs_runner_up"]].to_string(index=False))


if __name__ == "__main__":
    main()
