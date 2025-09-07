#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025/07/21

🚀 Welcome to the Awesome Python Script 🚀

User: Messou Franck Junior Aboya
Email: messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University - IIST - (Tokyo, Japan)
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

# pipeline/imputations/saits.py

import numpy as np
from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre
from pypots.optim import Adam
from pypots.imputation import SAITS


def train_and_evaluate_saits(dataset, args):
    print("🚀 Starting SAITS training pipeline...")

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
    saits = SAITS(
        n_steps=dataset["n_steps"],
        n_features=dataset["n_features"],
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        n_heads=args.n_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        dropout=args.dropout ,
        ORT_weight=args.ORT_weight ,
        MIT_weight=args.MIT_weight ,
        batch_size=args.batch_size ,
        epochs=args.epochs ,
        patience=args.patience ,
        optimizer=Adam(lr=1e-3),
        num_workers=0,
        device=args.device,
        saving_path=args.saving_path,
        model_saving_strategy="best",
    )

    # 3. Fit
    saits.fit(train_set=train_set, val_set=val_set)

    # 4. Predict
    results = saits.predict(test_set)
    imputations = results["imputation"]

    # 5. Evaluate
    mae = calc_mae(imputations, test_X_ori, test_X_indicating_mask)
    mse = calc_mse(imputations, test_X_ori, test_X_indicating_mask)
    rmse = calc_rmse(imputations, test_X_ori, test_X_indicating_mask)
    mre = calc_mre(imputations, test_X_ori, test_X_indicating_mask)
    print(f"[LLM4IMP] Testing —— MAE: {mae:.4f}| MSE: {mse:.4f}| RMSE: {rmse:.4f}| MRE: {mre:.4f}| ")
    return mae, mse, rmse, mre
