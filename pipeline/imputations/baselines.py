#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Naive imputation baselines for model selection experiments."""

import numpy as np
from pypots.imputation import LOCF, Mean, Median
from pypots.nn.functional import calc_mae, calc_mre, calc_mse, calc_rmse


def _evaluate_baseline(dataset: dict, model, model_label: str):
    test_set = {"X": dataset["test_X"]}
    test_X_ori = np.nan_to_num(dataset["test_X_ori"])
    test_X_indicating_mask = np.isnan(dataset["test_X_ori"]) ^ np.isnan(dataset["test_X"])

    results = model.predict(test_set)
    imputations = results["imputation"]

    mae = calc_mae(imputations, test_X_ori, test_X_indicating_mask)
    mse = calc_mse(imputations, test_X_ori, test_X_indicating_mask)
    rmse = calc_rmse(imputations, test_X_ori, test_X_indicating_mask)
    mre = calc_mre(imputations, test_X_ori, test_X_indicating_mask)

    print(f"[{model_label}] Testing —— MAE: {mae:.4f}| MSE: {mse:.4f}| RMSE: {rmse:.4f}| MRE: {mre:.4f}| ")
    return mae, mse, rmse, mre


def train_and_evaluate_mean(dataset: dict, args):
    print("🚀 Starting Mean imputation baseline...")
    return _evaluate_baseline(dataset, Mean(), "Mean")


def train_and_evaluate_median(dataset: dict, args):
    print("🚀 Starting Median imputation baseline...")
    return _evaluate_baseline(dataset, Median(), "Median")


def train_and_evaluate_locf(dataset: dict, args):
    print("🚀 Starting LOCF imputation baseline...")
    return _evaluate_baseline(dataset, LOCF(first_step_imputation="backward", device=args.device), "LOCF")
