#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025/07/24

🚀 Welcome to the Awesome Python Script 🚀

User: Messou Franck Junior Aboya
Email: messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University - IIST - (Tokyo, Japan)
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import numpy as np
from typing import Tuple

from pypots.optim import Adam
from pypots.imputation.tefn import TEFN
from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre


def train_and_evaluate_tefn(dataset: dict, args) -> Tuple[float, float, float, float]:
    print("🚀 Starting TEFN training pipeline...")

    # 1. Assemble train/val/test datasets
    train_set = {"X": dataset["train_X"]}
    val_set = {
        "X": dataset["val_X"],
        "X_ori": dataset["val_X_ori"],
    }
    test_set = {"X": dataset["test_X"]}
    test_X_ori = np.nan_to_num(dataset["test_X_ori"])
    test_X_indicating_mask = np.isnan(dataset["test_X_ori"]) ^ np.isnan(dataset["test_X"])

    # 2. Initialize model
    tefn = TEFN(
        n_steps=dataset["n_steps"],
        n_features=dataset["n_features"],
        n_fod=args.n_fod,
        apply_nonstationary_norm=args.apply_nonstationary_norm,
        ORT_weight=args.ORT_weight,
        MIT_weight=args.MIT_weight,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        optimizer=Adam(lr=1e-3),
        device=args.device,
        saving_path=args.saving_path,
        model_saving_strategy="best",
        verbose=True,
    )

    # 3. Fit
    tefn.fit(train_set=train_set, val_set=val_set)

    # 4. Predict
    results = tefn.predict(test_set)
    imputations = results["imputation"]

    # 5. Evaluate
    mae = calc_mae(imputations, test_X_ori, test_X_indicating_mask)
    mse = calc_mse(imputations, test_X_ori, test_X_indicating_mask)
    rmse = calc_rmse(imputations, test_X_ori, test_X_indicating_mask)
    mre = calc_mre(imputations, test_X_ori, test_X_indicating_mask)

    print(f"[TEFN] Testing —— MAE: {mae:.4f}| MSE: {mse:.4f}| RMSE: {rmse:.4f}| MRE: {mre:.4f}| ")
    return mae, mse, rmse, mre