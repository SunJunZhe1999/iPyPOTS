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
from pypots.imputation.gpt4ts import GPT4TS
from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre


def train_and_evaluate_gpt4ts(dataset: dict, args) -> Tuple[float, float, float, float]:
    print("🚀 Starting GPT4TS training pipeline...")

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
    gpt4ts = GPT4TS(
        n_steps=dataset["n_steps"],
        n_features=dataset["n_features"],
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        n_layers=args.n_layers,
        train_gpt_mlp=args.train_gpt_mlp,
        d_ffn=args.d_ffn,
        dropout=args.dropout,
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
    gpt4ts.fit(train_set=train_set, val_set=val_set)

    # 4. Predict
    results = gpt4ts.predict(test_set)
    imputations = results["imputation"]

    # 5. Evaluate
    mae = calc_mae(imputations, test_X_ori, test_X_indicating_mask)
    mse = calc_mse(imputations, test_X_ori, test_X_indicating_mask)
    rmse = calc_rmse(imputations, test_X_ori, test_X_indicating_mask)
    mre = calc_mre(imputations, test_X_ori, test_X_indicating_mask)

    print(f"[GPT4TS] Testing —— MAE: {mae:.4f}| MSE: {mse:.4f}| RMSE: {rmse:.4f}| MRE: {mre:.4f}| ")
    return mae, mse, rmse, mre