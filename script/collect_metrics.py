#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect per-run metric JSON files into a single CSV table."""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PyPOTS metric JSON files.")
    parser.add_argument("--root", default="output/imputation", help="Root directory to scan")
    parser.add_argument("--out", default="output/imputation/metrics_summary.csv", help="CSV path to write")
    args = parser.parse_args()

    root = Path(args.root)
    metric_files = sorted(root.glob("**/metrics/*_metrics.json"))
    rows = []
    for metric_file in metric_files:
        with metric_file.open() as f:
            metrics = json.load(f)
        relative = metric_file.relative_to(root)
        run_parts = relative.parent.parent.parts
        model_from_path = run_parts[0] if len(run_parts) > 0 else ""
        dataset = run_parts[1] if len(run_parts) > 1 else ""
        missing_rate = run_parts[2].removeprefix("mr") if len(run_parts) > 2 else ""
        seed = run_parts[3].removeprefix("seed") if len(run_parts) > 3 else ""
        rows.append(
            {
                "run": str(relative.parent.parent),
                "model": metric_file.name[: -len("_metrics.json")] or model_from_path,
                "dataset": dataset,
                "missing_rate": missing_rate,
                "seed": seed,
                "MAE": metrics.get("MAE"),
                "MSE": metrics.get("MSE"),
                "RMSE": metrics.get("RMSE"),
                "MRE": metrics.get("MRE"),
                "metrics_file": str(metric_file),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "model",
        "dataset",
        "missing_rate",
        "seed",
        "MAE",
        "MSE",
        "RMSE",
        "MRE",
        "metrics_file",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
