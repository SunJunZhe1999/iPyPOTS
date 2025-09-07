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
from pypots.imputation.moment import MOMENT
from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre


def train_and_evaluate_moment(dataset: dict, args) -> Tuple[float, float, float, float]:
    print("🚀 Starting MOMENT training pipeline...")

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
    model = MOMENT(
        n_steps=dataset["n_steps"],
        n_features=dataset["n_features"],
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        transformer_backbone=args.transformer_backbone,
        transformer_type=args.transformer_type,
        n_layers=args.n_layers,
        d_ffn=args.d_ffn,
        d_model=args.d_model,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        finetuning_mode=args.finetuning_mode,
        revin_affine=args.revin_affine,
        add_positional_embedding=args.add_positional_embedding,
        value_embedding_bias=args.value_embedding_bias,
        orth_gain=args.orth_gain,
        mask_ratio=args.mask_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        optimizer=Adam(lr=1e-3),
        device=args.device,
        saving_path=args.saving_path,
        model_saving_strategy=args.model_saving_strategy,
        verbose=args.verbose,
    )

    # 3. Fit model
    model.fit(train_set=train_set, val_set=val_set)

    # 4. Predict
    results = model.predict(test_set)
    imputations = results["imputation"]

    # 5. Evaluate
    mae = calc_mae(imputations, test_X_ori, test_X_indicating_mask)
    mse = calc_mse(imputations, test_X_ori, test_X_indicating_mask)
    rmse = calc_rmse(imputations, test_X_ori, test_X_indicating_mask)
    mre = calc_mre(imputations, test_X_ori, test_X_indicating_mask)

    print(f"[MOMENT] Testing —— MAE: {mae:.4f}| MSE: {mse:.4f}| RMSE: {rmse:.4f}| MRE: {mre:.4f}| ")
    return mae, mse, rmse, mre