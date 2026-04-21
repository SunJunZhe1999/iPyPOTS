#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and validate a C-TCAR-based model router."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script.ctcar_features import CTCAR_FEATURE_NAMES
from script.model_routing_analysis import build_routing_table


def _ctcar_columns(table: pd.DataFrame) -> list[str]:
    columns = [f"ctcar_{name}" for name in CTCAR_FEATURE_NAMES]
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"C-TCAR table is missing {len(missing)} columns, e.g. {missing[:5]}")
    return columns


def _load_training_frame(metrics_path: Path, ctcar_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    ctcar = pd.read_csv(ctcar_path)
    routing = build_routing_table(metrics)
    merged = routing.merge(ctcar, on=["dataset", "missing_rate"], how="inner", suffixes=("", "_ctcar"))
    if len(merged) != len(routing):
        missing = routing.merge(ctcar, on=["dataset", "missing_rate"], how="left", indicator=True)
        missing = missing.loc[missing["_merge"] == "left_only", ["dataset", "missing_rate"]]
        raise ValueError(f"Missing C-TCAR rows for routing scenarios:\n{missing}")
    return merged


def _train_router(frame: pd.DataFrame, random_state: int):
    feature_cols = _ctcar_columns(frame) + ["missing_rate"]
    X = frame[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = frame["best_model"]

    router = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
    )
    router.fit(X, y)

    diagnostics = {
        "training_accuracy": accuracy_score(y, router.predict(X)),
        "n_scenarios": int(len(frame)),
        "n_features": int(X.shape[1]),
        "classes": sorted(y.unique().tolist()),
    }

    if len(frame) > 1 and y.nunique() > 1:
        loo_pred = cross_val_predict(router, X, y, cv=LeaveOneOut())
        diagnostics["leave_one_scenario_out_accuracy"] = accuracy_score(y, loo_pred)
        diagnostics["leave_one_scenario_out_report"] = classification_report(y, loo_pred, output_dict=True, zero_division=0)

        logo = LeaveOneGroupOut()
        groups = frame["dataset"]
        logo_pred = cross_val_predict(router, X, y, cv=logo.split(X, y, groups=groups))
        diagnostics["leave_one_dataset_out_accuracy"] = accuracy_score(y, logo_pred)
        diagnostics["leave_one_dataset_out_report"] = classification_report(y, logo_pred, output_dict=True, zero_division=0)

    importances = (
        pd.DataFrame({"feature": feature_cols, "importance": router.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    predictions = frame[
        [
            "dataset",
            "missing_rate",
            "best_model",
            "best_MAE",
            "runner_up_model",
            "runner_up_MAE",
            "relative_gain_vs_runner_up",
            "action_feature_name",
            "outcome_feature_name",
        ]
    ].copy()
    predictions["router_prediction"] = router.predict(X)
    predictions["router_correct"] = predictions["router_prediction"] == predictions["best_model"]

    return router, diagnostics, importances, predictions


def _write_markdown_report(
    path: Path,
    diagnostics: dict,
    importances: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    top_importances = importances.head(15)
    lines = [
        "# C-TCAR Routing Report",
        "",
        "This report trains a RandomForest router over the 80-dimensional C-TCAR representation.",
        "The current target is the empirically best imputation model per dataset and missing-rate scenario.",
        "It is a causal-aware routing proxy, not yet a full interventional ATE benchmark.",
        "",
        "## Diagnostics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in diagnostics.items():
        if isinstance(value, (int, float, str)):
            lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Top C-TCAR Feature Importances",
            "",
            "| Rank | Feature | Importance |",
            "| ---: | --- | ---: |",
        ]
    )
    for idx, row in top_importances.iterrows():
        lines.append(f"| {idx + 1} | `{row['feature']}` | {row['importance']:.6f} |")

    lines.extend(
        [
            "",
            "## Scenario Routing Results",
            "",
            "| Dataset | Missing rate | Best | Router | Correct | Action proxy | Outcome proxy |",
            "| --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in predictions.sort_values(["dataset", "missing_rate"]).iterrows():
        lines.append(
            "| "
            f"{row['dataset']} | {row['missing_rate']:.1f} | {row['best_model']} | "
            f"{row['router_prediction']} | {bool(row['router_correct'])} | "
            f"{row['action_feature_name']} | {row['outcome_feature_name']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a C-TCAR RandomForest router.")
    parser.add_argument("--metrics", default="output/imputation/mps/energy_benchmark/metrics_summary.csv")
    parser.add_argument("--ctcar", default="output/imputation/mps/energy_benchmark/ctcar_features.csv")
    parser.add_argument("--out-dir", default="output/imputation/mps/energy_benchmark/ctcar_routing")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_training_frame(Path(args.metrics), Path(args.ctcar))
    router, diagnostics, importances, predictions = _train_router(frame, args.random_state)

    training_frame_path = out_dir / "ctcar_router_training_frame.csv"
    model_path = out_dir / "ctcar_router.joblib"
    diagnostics_path = out_dir / "ctcar_router_diagnostics.json"
    importances_path = out_dir / "ctcar_feature_importances.csv"
    predictions_path = out_dir / "ctcar_router_predictions.csv"
    report_path = out_dir / "ctcar_routing_report.md"

    frame.to_csv(training_frame_path, index=False)
    joblib.dump(router, model_path)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    importances.to_csv(importances_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    _write_markdown_report(report_path, diagnostics, importances, predictions)

    print(f"Wrote training frame to {training_frame_path}")
    print(f"Wrote router model to {model_path}")
    print(f"Wrote diagnostics to {diagnostics_path}")
    print(f"Wrote feature importances to {importances_path}")
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote report to {report_path}")
    print(json.dumps({k: v for k, v in diagnostics.items() if not isinstance(v, dict)}, indent=2))
    print(predictions.sort_values(["dataset", "missing_rate"]).to_string(index=False))


if __name__ == "__main__":
    main()
